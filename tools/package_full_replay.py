"""Package a static native auditor only after its full frozen-QEMU release gate."""
import argparse
import json
from pathlib import Path
import struct
from full_replay import ROOT,sha,source_sha

SOURCES=('tools/full_replay.c','yolo26_riscv/scalar_model.c','yolo26_riscv/scalar_model.h',
         'yolo26_riscv/basic.c','yolo26_riscv/basic.h','yolo26_riscv/activation.c',
         'yolo26_riscv/conv2d.h','yolo26_riscv/matmul.h','model/model_generated.c','model/model_generated.h',
         'model/constants.bin')


def package(binary,gate_path):
    raw=binary.read_bytes(); gate=json.loads(gate_path.read_text())
    if (raw[:6]!=b'\x7fELF\x02\x01' or struct.unpack_from('<H',raw,18)[0]!=62 or
            gate.get('status')!='passed' or gate.get('binary_sha256')!=sha(raw)):
        raise ValueError('Need the matching Linux x86_64 ELF and passed release gate')
    phoff=struct.unpack_from('<Q',raw,32)[0]; stride,count=struct.unpack_from('<HH',raw,54)
    if any(struct.unpack_from('<I',raw,phoff+i*stride)[0] in (2,3) for i in range(count)):
        raise ValueError('Native auditor has a dynamic loader/dependency')
    manifest=json.loads((ROOT/'reference/nodes/manifest.json').read_text())
    if len(gate['operators'])!=21: raise ValueError('Not all operator types were validated')
    for sample in ('bus','zidane'):
        expected=[{'node':n['id'],'tensor':tid,'sha256':next(r['sha256'] for r in manifest['samples'][sample]['records'] if r['id']==tid)}
                  for n in manifest['nodes'] for tid in n['outputs'] if tid>=0]
        actual=gate['samples'][sample]
        if actual['nodes']!=334 or actual['outputs']!=335 or actual['records']!=expected:
            raise ValueError('Incomplete/incompatible frozen QEMU gate: '+sample)
    record={'version':1,'arch':'linux-x86_64','static':True,'binary':'firmware/host-verify','sha256':sha(raw),
            'sources':{name:source_sha(ROOT/name) for name in SOURCES},
            'source_normalization':'CRLF-to-LF','reference_manifest_sha256':sha((ROOT/'reference/nodes/manifest.json').read_bytes()),
            'qemu_gate':{'status':'passed','nodes':668,'outputs':670,'operators':gate['operators'],
                         'full_report_sha256':sha(gate_path.read_bytes())}}
    (ROOT/'firmware/host-verify').write_bytes(raw)
    (ROOT/'firmware/host-verify').chmod(0o755)
    (ROOT/'firmware/host-verify.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    # SHA256SUMS is updated in the separate artifact batch, preserving LF.
    print('Packaged static auditor:',record['sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,default=ROOT/'yolo26_riscv/build/host-verify/host-verify')
    p.add_argument('--gate',type=Path,default=ROOT/'yolo26_pc/outputs/full-replay-gate.json')
    a=p.parse_args(); package(a.binary,a.gate)
