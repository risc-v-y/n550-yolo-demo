"""Exercise real probe code in unified QEMU RAM, including injected failures.

This does NOT validate S2C CLP remapping or physical cache behaviour.
Run on Linux/WSL: python3 yolo26_riscv/board/tests/test_clp.py
"""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
src = root / 'yolo26_riscv'
out = src / 'build/clp-qemu-tests'
out.mkdir(parents=True, exist_ok=True)
prefix = os.environ.get('TOOLCHAIN_PREFIX', str(root / 'toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf'))
qemu = os.environ.get('QEMU', str(root / 'toolchain/qemu_linux/bin/qemu-system-riscv64'))
flags = ['-march=rv64gcv_zicbom_zifencei_xewmatrix1p0', '-mabi=lp64d',
         '-mcmodel=medany', '-mno-relax', '-O2', '-g', '-std=c11',
         '-Wall', '-Wextra', '-Werror', '-ffreestanding', '-fno-builtin',
         '-fno-strict-aliasing', '-fno-tree-vectorize', '-mstringop-strategy=scalar',
         '-ffp-contract=off', '-fno-stack-protector', '-fno-pic', '-fno-pie',
         '-ffunction-sections', '-fdata-sections', '-fno-asynchronous-unwind-tables',
         '-DYOLO_N550_BOARD=1', '-DMODEL_PROFILE=0', '-DCLP_PROBE_SIM=1',
         '-DBOARD_UART_TEST=1', '-I', str(src), '-I', str(src / 'board')]
subprocess.run([prefix + '-gcc', *flags, '-c', 'board/start.S', '-o', str(out / 'start.o')],
               cwd=src, check=True, timeout=60)
normal = (src / 'board/clp_selftest.c').read_text(encoding='utf-8')
variants = [
    ('normal', None, None, 0, 'CLP RVV->AMU->RVV REUSE 32 ROUNDS PASS'),
    ('tag', 'add_tag((float)(round+1));', 'add_tag((float)(round==7 ? 0 : round+1));',
     4, 'round=0x0000000000000007 buffer=0x0000000000000003'),
    ('guard', '        (void)old;', '        write16(256,0);\n        (void)old;',
     4, 'round=0x000000000000001f buffer=0x0000000000000004'),
    ('alias', '#define CLP_DDR_BASE ((uintptr_t)0x80100000u)',
     '#define CLP_DDR_BASE ((uintptr_t)0x80102000u)',
     4, 'round=0x0000000000000000 buffer=0x0000000000000002'),
]
for name, anchor, replacement, exit_code, expected in variants:
    code = src / 'board/clp_selftest.c'
    if anchor is not None:
        if normal.count(anchor) != 1:
            raise RuntimeError('Fault injection anchor changed: ' + name)
        code = out / (name + '.c')
        code.write_text(normal.replace(anchor, replacement), encoding='utf-8')
    elf = out / (name + '.elf')
    subprocess.run([prefix + '-gcc', *flags, '-nostdlib', '-nostartfiles',
                    '-Wl,--gc-sections', '-T', 'board/ddr.ld', str(out / 'start.o'),
                    str(code), 'board/uart.c', 'board/diagnostics.c', 'amu.c',
                    'board/tests/clp_qemu.c', '-o', str(elf)], cwd=src, check=True, timeout=60)
    for vlen in (128, 256, 512):
        result = subprocess.run([qemu, '-machine', 'virt', '-cpu',
                                 f'rv64,v=true,vlen={vlen},matrix=on,xtlen=65536,xtrlen=512,xelen=32',
                                 '-m', '64M', '-smp', '1', '-nographic', '-monitor', 'none',
                                 '-serial', 'none', '-bios', 'none', '-kernel', str(elf),
                                 '-semihosting-config', 'enable=on,target=native', '-no-reboot'],
                                cwd=src, capture_output=True, text=True, timeout=30)
        text = result.stdout + result.stderr
        (out / f'{name}-vlen{vlen}.log').write_text(text, encoding='utf-8')
        if result.returncode != exit_code or expected not in text:
            raise RuntimeError(f'{name} vlen={vlen} exit={result.returncode}:\n{text}')
        print(f'PASS {name} vlen={vlen}', flush=True)
print('12 QEMU cases passed; physical remapping/cache/completion still require board validation.')
