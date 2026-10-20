"""Compare actual v5/v6 kernels under VLEN 128/512, not hardware timing."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'yolo26_riscv'
OUT = SRC / 'build/optimized-tests'
OUT.mkdir(parents=True, exist_ok=True)
TC = str(ROOT / 'toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf')
QEMU = str(ROOT / 'toolchain/qemu_linux/bin/qemu-system-riscv64')
BASELINE = '0bad6ce'
flags = ['-march=rv64gcv_zicbom_zifencei_xewmatrix1p0', '-mabi=lp64d', '-mcmodel=medany',
         '-mno-relax', '-O2', '-g', '-std=c11', '-Wall', '-Wextra', '-Werror', '-ffreestanding',
         '-fno-builtin', '-fno-strict-aliasing', '-fno-tree-vectorize', '-mstringop-strategy=scalar',
         '-ffp-contract=off', '-frounding-math', '-ffunction-sections', '-fdata-sections',
         '-DYOLO_N550_BOARD=1', '-DBOARD_CLP_SIM=1', '-DMODEL_PROFILE=0',
         '-I', str(SRC), '-I', str(ROOT/'model'), '-include', 'board/coherent_rvv.h']
renamed = ['rvv_depthwise3x3', 'rvv_depthwise3x3_calls', 'rvv_matmul_ordered', 'rvv_copy_bytes',
           'rvv_nonlinear', 'rvv_softmax', 'maxpool_fp32_rvv', 'nearest_fp32_rvv', 'add_fp32_rvv',
           'concat_fp32_rvv', 'slice_fp32_rvv', 'accel_model_node', 'accel_model_prepare',
           'accel_rvv_nodes', 'accel_amu_nodes', 'accel_scalar_nodes', 'accel_alias_nodes',
           'accel_weight_bytes', 'accel_weight_values', 'accel_weight_cycles', 'accel_weight_layers',
           'amu_pack_conv_input_rvv', 'amu_rvv_input_blocks', 'board_crc32']
old_objects = []
for name in ('rvv_model_ops.c', 'basic_rvv.c', 'accel_model.c', 'amu_input_rvv.c', 'board/model_io.c'):
    target = OUT/(Path(name).stem+'-v5.c')
    target.write_bytes(subprocess.check_output(['git', '-c', 'safe.directory='+str(ROOT), 'show',
                                               BASELINE+':yolo26_riscv/'+name], cwd=ROOT))
    # Snapshot uses original relative headers, not a second source tree.
    if name.startswith('board/'):
        target.write_text(target.read_text().replace('"model_io.h"', '"board/model_io.h"'))
    obj = target.with_suffix('.o')
    subprocess.run([TC+'-gcc', *flags, *['-D'+n+'='+n+'_v5' for n in renamed],
                    '-c', str(target), '-o', str(obj)], cwd=SRC, check=True)
    old_objects.append(str(obj))
start = OUT/'start.o'
subprocess.run([TC+'-gcc', *flags[:4], '-c', 'board/start.S', '-o', str(start)], cwd=SRC, check=True)
elf = OUT/'optimized.elf'
sources = ['rvv_model_ops.c', 'basic_rvv.c', 'accel_model.c', 'amu.c', 'amu_input_rvv.c',
           'scalar_model.c', 'basic.c', 'conv2d.c', 'matmul.c', 'activation.c', 'board/model_io.c',
           'board/tests/optimized_qemu.c', str(ROOT/'model/model_generated.c')]
subprocess.run([TC+'-gcc', *flags, '-nostdlib', '-nostartfiles', '-Wl,--gc-sections',
                '-Wl,--defsym=board_clp_sim=1', '-T', 'board/ddr.ld', str(start),
                *old_objects, *sources, '-o', str(elf)], cwd=SRC, check=True)
for name in ('rvv_matmul_ordered', 'rvv_nonlinear', 'rvv_softmax'):
    assembly = subprocess.check_output([TC+'-objdump', '-d', '--disassemble='+name, str(elf)], text=True)
    (OUT/(name+'.disasm')).write_text(assembly)
    if 'vfmacc' in assembly or 'vfmadd' in assembly:
        raise RuntimeError('Unexpected FMA: '+name)
for vlen in (128, 512):
    result = subprocess.run([QEMU, '-machine', 'virt', '-cpu',
                             'rv64,v=true,vlen={},matrix=on,xtlen=65536,xtrlen=512,xelen=32'.format(vlen),
                             '-m', '128M', '-smp', '1', '-nographic', '-monitor', 'none', '-serial', 'none',
                             '-bios', 'none', '-kernel', str(elf), '-semihosting-config',
                             'enable=on,target=native', '-no-reboot'], capture_output=True, timeout=900)
    text = (result.stdout+result.stderr).decode(errors='replace')
    (OUT/('vlen-{}.log'.format(vlen))).write_text(text)
    print('VLEN={}: {}'.format(vlen,text), flush=True)
    if result.returncode or 'OPTIMIZED PASS' not in text:
        raise RuntimeError('Optimized QEMU failed: {}'.format(result.returncode))
