"""Real AMU/RVV handoff regression; QEMU proves calls/results, not cache hardware."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT/'yolo26_riscv'
OUT = SRC/'build/ownership-tests'
OUT.mkdir(parents=True, exist_ok=True)
TC = str(ROOT/'toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf')
QEMU = str(ROOT/'toolchain/qemu_linux/bin/qemu-system-riscv64')
flags = ['-march=rv64gcv_zicbom_zifencei_xewmatrix1p0', '-mabi=lp64d', '-mcmodel=medany',
         '-mno-relax', '-O2', '-g', '-std=c11', '-Wall', '-Wextra', '-Werror',
         '-ffreestanding', '-fno-builtin', '-fno-strict-aliasing', '-fno-tree-vectorize',
         '-mstringop-strategy=scalar', '-ffp-contract=off', '-ffunction-sections',
         '-fdata-sections', '-DYOLO_N550_BOARD=1', '-DBOARD_CLP_SIM=1',
         '-DMODEL_PROFILE=0', '-I', str(SRC)]
subprocess.run([TC+'-gcc', *flags, '-c', 'board/start.S', '-o', str(OUT/'start.o')], cwd=SRC, check=True)
for fast in (0, 1):
    elf = OUT/('ownership-{}.elf'.format(fast))
    subprocess.run([TC+'-gcc', *flags, '-DYOLO_FAST='+str(fast), '-include', 'board/tests/cache_audit.h',
                    '-nostdlib', '-nostartfiles', '-Wl,--gc-sections', '-Wl,--defsym=board_clp_sim=1',
                    '-T', 'board/ddr.ld', str(OUT/'start.o'), 'amu.c', 'amu_input_rvv.c',
                    'board/tests/ownership_qemu.c', '-o', str(elf)], cwd=SRC, check=True)
    for vlen in (128, 512):
        result = subprocess.run([QEMU, '-machine', 'virt', '-cpu',
            'rv64,v=true,vlen={},matrix=on,xtlen=65536,xtrlen=512,xelen=32'.format(vlen),
            '-m', '128M', '-nographic', '-monitor', 'none', '-serial', 'none', '-bios', 'none',
            '-kernel', str(elf), '-semihosting-config', 'enable=on,target=native', '-no-reboot'],
            capture_output=True, timeout=60)
        text = (result.stdout+result.stderr).decode(errors='replace')
        (OUT/'fast-{}-vlen-{}.log'.format(fast,vlen)).write_text(text)
        print('fast={} vlen={}: {}'.format(fast,vlen,text), flush=True)
        if result.returncode or 'OWNERSHIP PASS' not in text:
            raise RuntimeError('Ownership test failed: {}'.format(result.returncode))
