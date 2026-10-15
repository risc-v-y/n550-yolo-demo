"""Validate the shipped package from actual Git bytes in a clean Linux directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',default='HEAD',help='Commit or staged tree to validate')
    p.add_argument('--output',type=Path,default=ROOT/'yolo26_pc/outputs/checkout-release-validation')
    args=p.parse_args()
    if sys.platform!='linux': p.error('Run the clean package check on Linux')
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    report={'status':'running','revision':args.revision}; started=time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='yolo-release-') as name:
            temporary=Path(name); archive=temporary/'package.tar'; checkout=temporary/'checkout'; checkout.mkdir()
            subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'archive','--format=tar',
                            '--output='+str(archive),args.revision],cwd=ROOT,check=True)
            with tarfile.open(str(archive)) as tar:
                for member in tar.getmembers():
                    (checkout/member.name).resolve().relative_to(checkout.resolve())
                    if member.issym() or member.islnk(): raise ValueError('Unexpected archive link')
                tar.extractall(str(checkout))
            archive.unlink()
            subprocess.run(['sha256sum','-c','firmware/SHA256SUMS'],cwd=checkout,check=True)
            sys.path[:0]=[str(checkout/'tools'),str(checkout/'yolo26_pc')]
            from full_replay import Replay,source_sha
            from pcie_prepared import checked_file,detections,save_annotation
            native=Replay(); native.close()
            manifest=checkout/'reference/nodes/manifest.json'
            report.update(reference_crlf_count=manifest.read_bytes().count(b'\r\n'),
                          reference_manifest_sha256=source_sha(manifest),images={})
            samples=json.loads((checkout/'reference/prepared/manifest.json').read_text())['samples']
            for sample in ('bus',):
                record=samples[sample]
                source=checked_file(record['source'],record['source_sha256'])
                raw=checked_file(record['reference'],record['reference_sha256'])[32:]
                found=detections(raw,record['transform']); path=output/(sample+'-detected.png')
                dimensions=save_annotation(path,source,record['transform'],found)
                report['images'][sample]={'detections':len(found),'size':list(dimensions),
                                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            gate=output/'full-replay-gate.json'
            subprocess.run([sys.executable,str(checkout/'tools/validate_full_replay.py'),
                            '--binary',str(checkout/'firmware/host-verify'),'--output',str(gate)],cwd=checkout,check=True)
            report['qemu_gate']=json.loads(gate.read_text())['status']
            report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=str(exc)); raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print('Clean Linux package: '+report['status'],flush=True)


if __name__=='__main__': main()
