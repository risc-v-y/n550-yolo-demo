"""Run actual board boot, CLP selftest, PCIe mailbox and all 334 model nodes.
QEMU uses unified RAM at 0x84000000, not physical CLP remapping. No hardware I/O.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import secrets
import math
import struct
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--capture',action='store_true',help='Exercise node snapshots/ACK and export all tensors')
parser.add_argument('--manifest',type=Path,default=root/'reference/prepared/manifest.json',
                    help='Validate the first prepared sample (normally BUS)')
args=parser.parse_args()
manifest_path=args.manifest.resolve()
manifest=json.loads(manifest_path.read_text())
samples=list(manifest['samples'])[:1]
if not samples:
    parser.error('A prepared frame is required')
src=root/'yolo26_riscv'
out=root/'yolo26_pc/outputs/clp-model-validation'/('run-'+secrets.token_hex(6))
out.mkdir(parents=True)
sys.path.insert(0,str(root/'yolo26_pc'))
from pcie_prepared import checked_file,reference_result,detections,annotated_svg,INPUT_BYTES,OUTPUT_BYTES
tc=os.environ.get('TOOLCHAIN_PREFIX',str(root/'toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf'))
qemu=os.environ.get('QEMU',str(root/'toolchain/qemu_linux/bin/qemu-system-riscv64'))
flags=['-march=rv64gcv_zicbom_zifencei_xewmatrix1p0','-mabi=lp64d','-mcmodel=medany','-mno-relax',
       '-O2','-g','-std=c11','-Wall','-Wextra','-Werror','-ffreestanding','-fno-builtin',
       '-fno-strict-aliasing','-fno-tree-vectorize','-mstringop-strategy=scalar','-ffp-contract=off',
       '-fno-stack-protector','-fno-pic','-fno-pie','-ffunction-sections','-fdata-sections',
       '-fno-asynchronous-unwind-tables','-DYOLO_N550_BOARD=1','-DMODEL_PRECISION=1','-DMODEL_PROFILE=1',
       '-DBOARD_CLP_SIM=1','-DCLP_PROBE_EMBEDDED=1','-DBOARD_UART_TEST=1','-DCLP_MODEL_QEMU=1',
       '-I',str(src),'-I',str(root/'model')]
if args.capture:
    flags+=['-DMODEL_CAPTURE_TEST=1','-Wl,--wrap=board_pcie_node_end']
subprocess.run([tc+'-gcc',*flags,'-c','board/start.S','-o',str(out/'start.o')],cwd=src,check=True)
(out/'weights.S').write_text('.section .rodata.weights,"a",@progbits\n.balign 64\n'
                            '.global model_weights,model_weights_end\nmodel_weights:\n'
                            f'.incbin "{root}/model/constants.bin"\nmodel_weights_end:\n.balign 64\n')
subprocess.run([tc+'-gcc',*flags,'-c',str(out/'weights.S'),'-o',str(out/'weights.o')],cwd=src,check=True)
sources=['matmul.c','conv2d.c','activation.c','basic.c','basic_rvv.c','scalar_model.c',
         'rvv_model_ops.c','amu.c','amu_input_rvv.c','accel_model.c','board/uart.c','board/model_io.c',
         'board/main.c','board/clp_selftest.c','board/diagnostics.c','board/pcie.c',
         'board/tests/model_qemu.c','board/tests/clp_qemu.c',str(root/'model/model_generated.c')]
elf=out/'model-qemu.elf'
subprocess.run([tc+'-gcc',*flags,'-include','board/coherent_rvv.h','-nostdlib','-nostartfiles','-Wl,--gc-sections',
                '-Wl,--defsym=board_clp_sim=1','-Wl,--wrap=board_pcie_loop','-T','board/ddr.ld',
                str(out/'start.o'),str(out/'weights.o'),*sources,'-o',str(elf)],cwd=src,check=True)
for index,sample in enumerate(samples):
    record=manifest['samples'][sample]
    (out/f'frame-{index}-input.bin').write_bytes(checked_file(record['input'],record['input_sha256'],INPUT_BYTES))
    (out/f'{sample}-original.jpg').write_bytes(checked_file(record['source'],record['source_sha256']))
report={'status':'running','mode':'QEMU unified RAM','nodes_per_frame':334,
        'physical_clp_cache_verified':False,'samples':{},'output':str(out),
        'manifest':manifest_path.relative_to(root).as_posix(),'node_capture':args.capture}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'QEMU model started; UART/progress: {out}/uart.log',flush=True)
started=time.monotonic()
try:
    with (out/'uart.log').open('wb') as log:
        result=subprocess.run([qemu,'-machine','virt','-cpu',
                               'rv64,v=true,vlen=512,matrix=on,xtlen=65536,xtrlen=512,xelen=32',
                               '-m','128M','-smp','1','-nographic','-monitor','none','-serial','none',
                               '-bios','none','-kernel',str(elf),'-semihosting-config',
                               'enable=on,target=native','-no-reboot'],cwd=out,stdout=log,stderr=subprocess.STDOUT,
                              timeout=1800)
except BaseException as exc:
    report.update(status='failed',error=str(exc),elapsed_seconds=time.monotonic()-started)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    raise
report['elapsed_seconds']=time.monotonic()-started
report['qemu_exit']=result.returncode
text=(out/'uart.log').read_text(errors='replace')
if result.returncode or 'QEMU MODEL BUS PASS' not in text:
    report['status']='failed'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    raise RuntimeError(f'QEMU model failed; inspect {out}/uart.log')
from board_profile import decode_profile
profile=decode_profile((out/'frame-0-profile.bin').read_bytes(),1,0x12345678)
report.update(profile_version=profile['version'],input_path=profile['input_path'],
              profile_sha256=profile['sha256'],phases=profile['phases'])
for index,sample in enumerate(samples):
    record=manifest['samples'][sample]
    raw=(out/f'frame-{index}-candidates.bin').read_bytes()
    (out/f'{sample}-candidates.bin').write_bytes(raw)
    if len(raw)!=OUTPUT_BYTES or any(not all(math.isfinite(x) for x in row) or
            row[5]!=int(row[5]) or not 0<=row[5]<80 for row in struct.iter_unpack('<6f',raw)):
        raise ValueError('Invalid result length, finite values or class range')
    comparison=None
    if args.capture:
        from node_compare import ReferenceArchive
        capture_manifest=json.loads((root/'reference/nodes/manifest.json').read_text(encoding='utf-8'))
        archive=ReferenceArchive(root,capture_manifest,sample)
        try:
            with (out/f'frame-{index}-tensors.bin').open('rb') as stream:
                if struct.unpack('<8sII',stream.read(16))!=(b'Y26SCAL1',1,336):
                    raise ValueError('Invalid actual capture header')
                for tensor_record in capture_manifest['samples'][sample]['records']:
                    tid,dtype,size=struct.unpack('<3I',stream.read(12))
                    if (tid,dtype,size)!=(tensor_record['id'],capture_manifest['tensors'][tid]['dtype'],
                                          capture_manifest['tensors'][tid]['bytes']) or stream.read(size)!=archive.take(tid):
                        raise ValueError(f'Actual model capture differs at tensor {tid}')
                if stream.read(1):
                    raise ValueError('Unexpected actual capture tail')
            archive.finish()
        finally:
            archive.close()
    if 'reference' in record:
        reference=checked_file(record['reference'],record['reference_sha256'],32+OUTPUT_BYTES)
        comparison=reference_result(raw,reference)
    found=detections(raw,record['transform'])
    source=(out/f'{sample}-original.jpg').read_bytes()
    (out/f'{sample}-detected.svg').write_text(annotated_svg(source,record['transform'],found))
    report['samples'][sample]={'reference':comparison,'detections':found,
                                'sha256':hashlib.sha256(raw).hexdigest()}
    print(f'{sample}: {len(found)} detections; reference={comparison}',flush=True)
comparisons=[item['reference'] for item in report['samples'].values() if item['reference'] is not None]
report['status']='completed' if not comparisons else ('passed' if all(item['byte_identical'] for item in comparisons) else 'differences')
(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(f'Model validation {report["status"]}, {report["elapsed_seconds"]:.1f}s. Artifacts: {out}',flush=True)
if report['status']=='differences':
    raise RuntimeError('Numerical differences recorded; no tolerance was used to hide them')
