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
    # The probe has its own BIN/ELF and preserves the physical model-data
    # layout. Never package a QEMU test image or a mismatched symbol file.
    probe_symbols={}
    for line in (root/'firmware/tile-probe-symbols.txt').read_text().splitlines():
        address,kind,name=line.split(); probe_symbols[name]=int(address,16)
    for name in ('arena','conv_scratch','left_tile','right_tile','__clp_start','__clp_end'):
        assert probe_symbols[name]==symbols[name],name
    probe_address=probe_symbols['board_tile_probe']
    assert probe_address%64==0 and 0x80000000<=probe_address<probe_address+25024<=probe_symbols['__bss_end']<=0x81f00000
    probe_elf=(root/'firmware/tile-probe.elf').read_bytes()
    assert struct.unpack_from('<Q',probe_elf,24)[0]==0x80000000
    probe_blob=(root/'firmware/tile-probe.bin').read_bytes()
    assert len(probe_blob)<12*1024*1024
    at=probe_symbols['model_weights']-0x80000000
    assert probe_blob[at:at+9678644]==(root/'model/constants.bin').read_bytes()
    with tempfile.TemporaryDirectory() as temp:
        generated=Path(temp)/'probe.bin'
        subprocess.run([str(tc)+'-objcopy','-O','binary',str(root/'firmware/tile-probe.elf'),str(generated)],check=True)
        assert generated.read_bytes()==probe_blob
    assert 'clp_qemu_exit' not in probe_symbols and '__wrap_board_pcie_loop' not in probe_symbols
    print(f'Tile diagnostic package verified: BIN={len(probe_blob)} bytes, snapshot=25024 bytes; physical model layout preserved.')
    print('Program, constants, diagnostic symbols and BIN/ELF match; physical inference remains to validate.')

if __name__=='__main__':
    main()
