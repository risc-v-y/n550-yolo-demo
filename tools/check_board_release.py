"""Verify the physical CLP-model package and frozen graph without hardware."""
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import collect_board_diag as diag

def check_image(root,image,symbol_file):
    elf=(root/'firmware'/(image+'.elf')).read_bytes()
    entry,phoff=struct.unpack_from('<QQ',elf,24)
    entsize,count=struct.unpack_from('<HH',elf,54)
    assert entry==0x80000000 and entsize==56
    high=[]
    for index in range(count):
        kind,flags,offset,vaddr,paddr,filesz,memsz,align=struct.unpack_from('<IIQQQQQQ',elf,phoff+index*entsize)
        if kind==1 and paddr>=0xf0000000:
            assert paddr>=0xf0000000 and paddr+memsz<=0xf1000000 and filesz==0
            high.append((paddr,memsz))
    assert high and 0<sum(size for address,size in high)<=16*1024*1024
    symbols={}
    for line in (root/'firmware'/symbol_file).read_text().splitlines():
        address,kind,name=line.split()
        symbols[name]=int(address,16)
    assert symbols['__bss_start']<symbols['__bss_end']<=0x81f00000
    assert symbols['__bss_start']<=symbols['prepared_weights']<symbols['__bss_end']-8*1024*1024+1
    assert symbols['prepared_weights']%64==0 and symbols['board_profile']%64==0
    assert symbols['prepared_weights']+8*1024*1024<=symbols['__bss_end']
    assert symbols['__clp_start']==0xf0000000 and symbols['__clp_end']<=0xf1000000
    assert sum(size for address,size in high)==symbols['__clp_end']-symbols['__clp_start']
    for name in ('arena','conv_scratch','converted_feature','input_slots','left_tile','right_tile'):
        assert 0xf0000000<=symbols[name]<0xf1000000
    for name in ('stream_info','stream_result'):
        assert symbols['__bss_start']<=symbols[name]<symbols['__bss_end'] and symbols[name]%64==0
    sizes={'arena':7614720,'conv_scratch':1048576,'converted_feature':1384448,'input_slots':1038464}
    # Diagnostic readers intentionally reject CLP addresses; inspect the ELF
    # symbol sizes here and apply the separately approved 16 MiB CLP bounds.
    tc=root/'toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf'
    entries=subprocess.check_output([str(tc)+'-nm','-S','--defined-only',str(root/'firmware'/(image+'.elf'))],text=True)
    actual={parts[3]:(int(parts[0],16),int(parts[1],16)) for line in entries.splitlines()
            if len(parts:=line.split())==4}
    for name,size in sizes.items():
        address,allocated=actual[name]
        assert allocated==size and address%64==0 and 0xf0000000<=address<=0xf1000000-size
    ranges=sorted((actual[n][0],actual[n][0]+size) for n,size in sizes.items())
    assert all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))
    assert symbols['model_weights_end']-symbols['model_weights']==9678644
    blob=(root/'firmware'/(image+'.bin')).read_bytes()
    assert len(blob)<12*1024*1024
    at=symbols['model_weights']-0x80000000
    assert blob[at:at+9678644]==(root/'model/constants.bin').read_bytes()
    tc=root/'toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf'
    with tempfile.TemporaryDirectory() as temp:
        generated=Path(temp)/'demo.bin'
        subprocess.run([str(tc)+'-objcopy','-O','binary',str(root/'firmware'/(image+'.elf')),str(generated)],check=True)
        assert generated.read_bytes()==blob
    info,addresses,prefix=diag.firmware_info(root,image)
    probe=diag.elf_symbols(elf,{'board_pack_probe':256},performance=image=='demo-fast')['board_pack_probe']
    assert probe%64==0 and 0x80000000<=probe<=0x81f00000-256
    assert all(0x80000000<=address<0x82000000 for address in addresses.values())
    if image=='demo-fast':
        assert 'board_clp_selftest' not in symbols and 'board_diag_phase' not in symbols
        assert 'board_diag_trap' in symbols and 'board_diag_error' in symbols
    print(f'{image} package verified: BIN={len(blob)} bytes, CLP NOLOAD={sum(size for address,size in high)} bytes.')
    print('Program, constants, diagnostic symbols and BIN/ELF match; physical inference remains to validate.')


def main():
    root=Path(__file__).resolve().parents[1]
    subprocess.run(['sha256sum','-c','firmware/SHA256SUMS'],cwd=root,check=True)
    subprocess.run([sys.executable,'tools/check_assets.py'],cwd=root,check=True)
    subprocess.run([sys.executable,'tools/build_operator_audit.py','--check'],cwd=root,check=True)
    check_image(root,'demo-pcie','pcie-symbols.txt')
    check_image(root,'demo-fast','demo-fast-symbols.txt')

if __name__=='__main__':
    main()
