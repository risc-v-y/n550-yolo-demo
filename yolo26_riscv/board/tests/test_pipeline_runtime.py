"""Native tests of actual CRC and alias-root interval allocator; no board."""
import ctypes
from pathlib import Path
import random
import subprocess
import tempfile
import zlib

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'yolo26_riscv'
with tempfile.TemporaryDirectory() as name:
    temp=Path(name)
    subprocess.run(['cc','-shared','-fPIC','-O2',str(SRC/'board/model_io.c'),'-o',str(temp/'crc.so')],check=True)
    lib=ctypes.CDLL(str(temp/'crc.so'))
    lib.board_crc32.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    lib.board_crc32.restype=ctypes.c_uint32
    rng=random.Random(416)
    for size in list(range(130))+[519168,2076672]:
        raw=bytes(rng.randrange(256) for _ in range(size+8))
        owner=ctypes.create_string_buffer(raw)
        for offset in range(8):
            assert lib.board_crc32(ctypes.addressof(owner)+offset,size)==zlib.crc32(raw[offset:offset+size])
    print('CRC: empty, tails, eight alignments, full UINT8/FP32 frame PASS',flush=True)
    test=temp/'plan.c'
    test.write_text(r'''
#include <assert.h>
#include <stdio.h>
#include "board/graph_plan.h"
int main(void) {
    for(unsigned flags=0;flags<16;++flags) {
        assert(!graph_plan(flags));
        if(!flags) continue;
        for(unsigned t=0;t<MODEL_TENSOR_COUNT;++t) {
            if(model_tensors[t].constant) continue;
            unsigned root=graph_root[t];
            assert(execution_tensors[t].offset+model_tensors[t].bytes<=MODEL_ARENA_BYTES);
            assert(execution_tensors[t].offset==execution_tensors[root].offset+graph_relative[t]);
            if(root!=t || graph_first[t]<-1) continue;
            assert(execution_tensors[t].offset%64==0);
            for(unsigned u=t+1;u<MODEL_TENSOR_COUNT;++u) {
                if(model_tensors[u].constant || graph_root[u]!=u || graph_first[u]<-1 ||
                   graph_last[t]<graph_first[u] || graph_last[u]<graph_first[t]) continue;
                unsigned a=execution_tensors[t].offset,b=execution_tensors[u].offset;
                assert(a+((model_tensors[t].bytes+63)&~63u)<=b ||
                       b+((model_tensors[u].bytes+63)&~63u)<=a);
            }
        }
        printf("Graph flags=%u peak=%zu PASS\n",flags,graph_peak);
    }
}
''')
    subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-I',str(SRC),
                    '-I',str(ROOT/'model'),str(test),str(ROOT/'model/model_generated.c'),
                    '-o',str(temp/'plan')],check=True)
    subprocess.run([str(temp/'plan')],check=True)
