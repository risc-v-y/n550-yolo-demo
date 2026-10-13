"""Native mailbox round trip + prepared-image and DDR reader regression."""
from pathlib import Path
import subprocess
import sys
def main():
    root=Path(__file__).resolve().parents[1]
    src=root/'yolo26_riscv'
    out=src/'build/host-protocol-tests'
    out.mkdir(parents=True,exist_ok=True)
    binary=out/'pcie-host'
    subprocess.run(['gcc','-std=c11','-O2','-Wall','-Wextra','-Werror','-mcmodel=large','-fno-pie','-no-pie',
                    '-DBOARD_SNAPSHOT_HOST_TEST=1','-Wl,--section-start=.pcie_snapshot=0x80a10800',
                    '-Wl,--section-start=.pcie_mailbox=0x81f00000','board/pcie.c','board/model_io.c',
                    'board/diagnostics.c','board/tests/pcie_host.c','-o',str(binary)],cwd=src,check=True)
    subprocess.run([sys.executable,'test_pcie_protocol.py',str(binary)],cwd=root/'yolo26_pc',check=True)
    subprocess.run([sys.executable,'-m','unittest','test_node_capture','test_node_report'],cwd=root/'yolo26_pc',check=True)
    subprocess.run([sys.executable,'-m','unittest','test_pcie_prepared'],cwd=root/'yolo26_pc',check=True)
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tools','-p','test_*.py'],cwd=root,check=True)

if __name__=='__main__':
    main()
