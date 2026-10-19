"""Verify the physical CLP-model package and frozen graph without hardware."""
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import collect_board_diag as diag

def main():
    root=Path(__file__).resolve().parents[1]
    subprocess.run(['sha256sum','-c','firmware/SHA256SUMS'],cwd=root,check=True)
    subprocess.run([sys.executable,'tools/check_assets.py'],cwd=root,check=True)
    subprocess.run([sys.executable,'tools/build_operator_audit.py','--check'],cwd=root,check=True)
    elf=(root/'firmware/demo-pcie.elf').read_bytes()
    entry,phoff=struct.unpack_from('<QQ',elf,24)
    entsize,count=struct.unpack_from('<HH',elf,54)
    assert entry==0x80000000 and entsize==56
    high=[]
    for index in range(count):
        kind,flags,offset,vaddr,paddr,filesz,memsz,align=struct.unpack_from('<IIQQQQQQ',elf,phoff+index*entsize)
        if kind==1 and paddr>=0xf0000000:
            assert paddr>=0xf0000000 and paddr+memsz<=0xf1000000 and filesz==0
            high.append((paddr,memsz))
    assert high and sum(size for address,size in high)==8143104
    symbols={}
    for line in (root/'firmware/pcie-symbols.txt').read_text().splitlines():
        address,kind,name=line.split()
        symbols[name]=int(address,16)
    assert symbols['__bss_start']<symbols['__bss_end']<=0x81f00000
    assert symbols['__bss_start']<=symbols['prepared_weights']<symbols['__bss_end']-5*1024*1024+1
    assert symbols['prepared_weights']%64==0 and symbols['board_profile']%64==0
    assert symbols['prepared_weights']+5*1024*1024<=symbols['__bss_end']
    assert symbols['__clp_start']==0xf0000000 and symbols['__clp_end']==0xf07c4100
    for name in ('arena','conv_scratch','left_tile','right_tile'):
        assert 0xf0000000<=symbols[name]<0xf1000000
    assert symbols['model_weights_end']-symbols['model_weights']==9678644
    blob=(root/'firmware/demo-pcie.bin').read_bytes()
    assert len(blob)<12*1024*1024
    at=symbols['model_weights']-0x80000000
    assert blob[at:at+9678644]==(root/'model/constants.bin').read_bytes()
    tc=root/'toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf'
    with tempfile.TemporaryDirectory() as temp:
        generated=Path(temp)/'demo.bin'
        subprocess.run([str(tc)+'-objcopy','-O','binary',str(root/'firmware/demo-pcie.elf'),str(generated)],check=True)
        assert generated.read_bytes()==blob
    info,addresses,prefix=diag.firmware_info(root,'demo-pcie')
    assert all(0x80000000<=address<0x82000000 for address in addresses.values())
    print(f'Physical CLP package verified: BIN={len(blob)} bytes, CLP NOLOAD=8143104 bytes.')
    print('Program, constants, diagnostic symbols and BIN/ELF match; physical inference remains to validate.')

if __name__=='__main__':
    main()
