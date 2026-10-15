"""Build and execute actual diagnostic firmware twice in QEMU unified RAM."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

root=Path(__file__).resolve().parents[3]
subprocess.run(['bash','yolo26_riscv/build_board.sh','--tile-probe-qemu'],cwd=root,check=True)
base=root/'yolo26_pc/outputs/tile-probe-validation'; base.mkdir(parents=True,exist_ok=True)
out=Path(tempfile.mkdtemp(prefix='run-',dir=str(base)))
sys.path.insert(0,str(root/'tools'))
import tile_probe
fixture=tile_probe.assets()
(out/'input.bin').write_bytes(fixture[1][0])
qemu=os.environ.get('QEMU',str(root/'toolchain/qemu_linux/bin/qemu-system-riscv64'))
started=time.monotonic()
with (out/'uart.log').open('wb') as log:
    subprocess.run([qemu,'-machine','virt','-cpu',
        'rv64,v=true,vlen=512,matrix=on,xtlen=65536,xtrlen=512,xelen=32',
        '-m','128M','-smp','1','-nographic','-monitor','none','-serial','none','-bios','none',
        '-kernel',str(root/'yolo26_riscv/build/n550-board/tile-probe-qemu.elf'),
        '-semihosting-config','enable=on,target=native','-no-reboot'],cwd=out,
        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
assert 'QEMU TILE PROBE TWO RUNS PASS' in (out/'uart.log').read_text()
results=[]
for i in (1,2):
    result=tile_probe.analyze((out/('probe-{}.bin'.format(i))).read_bytes(),fixture)
    for t in result['tiles']:
        assert all(t[k]==0 for k in ('weight_bad','input_bad','bias_bad','pre_vs_exact_RNE',
                                     'post_vs_actual_pre_plus_bias','post_vs_qemu')),t
    results.append(result)
assert results[0]['sha256']==results[1]['sha256']
report={'status':'passed','mode':'QEMU unified RAM (physical cache unverified)',
        'elapsed_seconds':time.monotonic()-started,'runs':results}
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('QEMU probe PASS: 2 identical runs, 1536 outputs/run, all active operands and stages exact.')
print(out)
