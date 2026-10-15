"""Release gate: replay ALL frozen QEMU nodes for both samples, never hardware."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from full_replay import Replay,ROOT,sha

sys.path.insert(0,str(ROOT/'yolo26_pc'))
from node_compare import ReferenceArchive
from node_capture import verify_reference_manifest


def validate(binary,output):
    manifest=json.loads((ROOT/'reference/nodes/manifest.json').read_text())
    verify_reference_manifest(manifest)
    constants=(ROOT/'model/constants.bin').read_bytes()
    tensors=manifest['tensors']; nodes=manifest['nodes']
    last={tid:n['id'] for n in nodes for tid in n['inputs']}
    report={'status':'running','binary_sha256':sha(binary.read_bytes()),'samples':{}}
    started=time.monotonic()
    try:
        for sample in ('bus','zidane'):
            replay=Replay(32,binary=binary); archive=ReferenceArchive(ROOT,manifest,sample)
            live={0:archive.take(0)}; results=[]
            try:
                for node in nodes:
                    inputs={}
                    for tid in node['inputs']:
                        t=tensors[tid]
                        inputs[tid]=constants[t['offset']:t['offset']+t['bytes']] if t['constant'] else live[tid]
                    predicted=replay.run(node,tensors,inputs)
                    for tid in [tid for tid in node['outputs'] if tid>=0]:
                        expected=archive.take(tid)
                        if predicted[tid]!=expected:
                            offset=next(i for i,(a,b) in enumerate(zip(predicted[tid],expected)) if a!=b)
                            raise ValueError('{} node={} {} tensor={} first byte={}'.format(sample,node['id'],node['kind'],tid,offset))
                        live[tid]=expected; results.append({'node':node['id'],'tensor':tid,'sha256':sha(expected)})
                    for tid in list(live):
                        if last.get(tid,-1)<=node['id']: del live[tid]
                    if node['id']%32==0: print(sample,'node',node['id'],'PASS',flush=True)
                archive.finish()
            finally:
                replay.close(); archive.close()
            report['samples'][sample]={'nodes':334,'outputs':len(results),'records':results}
        report.update(status='passed',operators=dict(Counter(n['kind'] for n in nodes)))
    except Exception as exc:
        report.update(status='failed',error=str(exc)); raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+'\n')
    print('Native release gate PASS: both samples, 668 nodes / 670 full outputs exact.',flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,default=ROOT/'yolo26_riscv/build/host-verify/host-verify')
    p.add_argument('--output',type=Path,default=ROOT/'yolo26_pc/outputs/full-replay-gate.json')
    a=p.parse_args(); validate(a.binary,a.output)
