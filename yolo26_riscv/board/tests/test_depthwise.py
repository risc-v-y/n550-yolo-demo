"""Actual old/scalar/direct DW3x3 kernels under two QEMU vector lengths."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'yolo26_riscv'
OUT = SRC / 'build/depthwise-tests'
OUT.mkdir(parents=True, exist_ok=True)
TC = str(ROOT / 'toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf')
QEMU = str(ROOT / 'toolchain/qemu_linux/bin/qemu-system-riscv64')
ELF = OUT / 'depthwise.elf'
flags = ['-march=rv64gcv_zicbom_zifencei_xewmatrix1p0', '-mabi=lp64d', '-mcmodel=medany',
         '-mno-relax', '-O2', '-g', '-std=c11', '-Wall', '-Wextra', '-Werror', '-ffreestanding',
         '-fno-builtin', '-fno-strict-aliasing', '-fno-tree-vectorize', '-mstringop-strategy=scalar',
         '-ffp-contract=off', '-frounding-math', '-ffunction-sections', '-fdata-sections',
         '-DYOLO_N550_BOARD=1', '-DBOARD_CLP_SIM=1', '-DMODEL_PROFILE=0',
         '-I', str(SRC), '-I', str(ROOT / 'model'), '-include', 'board/coherent_rvv.h',
         '-nostdlib', '-nostartfiles', '-Wl,--gc-sections', '-Wl,--defsym=board_clp_sim=1',
         '-T', str(SRC / 'board/ddr.ld')]
subprocess.run([TC + '-gcc', '-march=rv64gcv_zicbom_zifencei_xewmatrix1p0', '-mabi=lp64d',
                '-mcmodel=medany', '-mno-relax', '-c', 'board/start.S', '-o', str(OUT / 'start.o')],
               cwd=SRC, check=True)
subprocess.run([TC + '-gcc', *flags, str(OUT / 'start.o'), 'rvv_model_ops.c', 'conv2d.c',
                'board/tests/depthwise_qemu.c', str(ROOT / 'model/model_generated.c'),
                '-o', str(ELF)], cwd=SRC, check=True)
assembly = subprocess.check_output([TC + '-objdump', '-d', '--disassemble=rvv_depthwise3x3', str(ELF)], text=True)
(OUT / 'depthwise.disasm').write_text(assembly)
assert all(op in assembly for op in ('vle32.v', 'vse32.v', 'vfmul.vf', 'vfadd.vv', 'vfadd.vf'))
assert 'vfmacc' not in assembly and 'vfmadd' not in assembly
for vlen in (128, 512):
    result = subprocess.run([QEMU, '-machine', 'virt', '-cpu',
                            'rv64,v=true,vlen={},matrix=on,xtlen=65536,xtrlen=512,xelen=32'.format(vlen),
                            '-m', '128M', '-smp', '1', '-nographic', '-monitor', 'none', '-serial', 'none',
                            '-bios', 'none', '-kernel', str(ELF), '-semihosting-config',
                            'enable=on,target=native', '-no-reboot'], capture_output=True, timeout=300)
    text = (result.stdout + result.stderr).decode(errors='replace')
    (OUT / 'vlen-{}.log'.format(vlen)).write_text(text)
    print('VLEN={}: {}'.format(vlen, text), end='', flush=True)
    if result.returncode or 'DEPTHWISE PASS' not in text:
        raise RuntimeError('Depthwise QEMU failed: {}'.format(result.returncode))
