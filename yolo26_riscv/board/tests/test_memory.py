"""Check physical pointer mapping and production RVV cache handoff choices."""
import os
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
src=root/'yolo26_riscv'
out=src/'build/memory-tests'
out.mkdir(parents=True,exist_ok=True)
subprocess.run(['gcc','-std=c11','-Wall','-Wextra','-Werror','board/tests/memory_host.c',
                '-o',str(out/'memory-host')],cwd=src,check=True)
subprocess.run([str(out/'memory-host')],check=True)
tc=os.environ.get('TOOLCHAIN_PREFIX',str(root/'toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf'))
flags=['-march=rv64gcv_zicbom_zifencei_xewmatrix1p0','-mabi=lp64d','-mcmodel=medany','-mno-relax',
       '-O2','-std=c11','-Wall','-Wextra','-Werror','-ffreestanding','-fno-builtin','-fno-tree-vectorize',
       '-mstringop-strategy=scalar','-ffunction-sections','-fdata-sections','-DBOARD_CLP_SIM=1',
       '-DCLP_MODEL_QEMU=1','-I',str(src)]
subprocess.run([tc+'-gcc',*flags,'-c','board/start.S','-o',str(out/'start.o')],cwd=src,check=True)
elf=out/'coherent.elf'
subprocess.run([tc+'-gcc',*flags,'-nostdlib','-nostartfiles','-Wl,--gc-sections',
                '-Wl,--defsym=board_clp_sim=1','-T','board/ddr.ld',str(out/'start.o'),
                'board/tests/coherent_qemu.c','board/tests/clp_qemu.c','-o',str(elf)],cwd=src,check=True)
qemu=os.environ.get('QEMU',str(root/'toolchain/qemu_linux/bin/qemu-system-riscv64'))
for vlen in (128,256,512):
    subprocess.run([qemu,'-machine','virt','-cpu',f'rv64,v=true,vlen={vlen}',
                    '-m','128M','-nographic','-monitor','none','-serial','none','-bios','none',
                    '-kernel',str(elf),'-semihosting-config','enable=on,target=native',
                    '-no-reboot'],cwd=out,check=True,timeout=30)
    print(f'PASS cache handoff choices vlen={vlen}',flush=True)
print('Physical address conversion + 3 wrapper cases passed; physical cache behaviour unverified.')
