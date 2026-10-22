#include "model_io.h"
uint32_t board_crc32(const void *data,size_t bytes) {
    /* Slicing-by-four IEEE CRC32. Initialize once during boot (one hart,
     * no interrupt handler calls CRC); every payload byte stays covered. */
    static uint32_t table[4][256];
    static int ready;
    if(!ready) {
        for(unsigned i=0;i<256;++i) {
            uint32_t c=i;
            for(unsigned j=0;j<8;++j) c=(c>>1)^((0u-(c&1u))&0xedb88320u);
            table[0][i]=c;
        }
        for(unsigned j=1;j<4;++j) for(unsigned i=0;i<256;++i) {
            uint32_t c=table[j-1][i];
            table[j][i]=(c>>8)^table[0][c&255];
        }
        ready=1;
    }
    const unsigned char *p=data;
    uint32_t crc=~0u;
    while(bytes>=4) {
        crc^=(uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
        crc=table[3][crc&255]^table[2][(crc>>8)&255]^table[1][(crc>>16)&255]^table[0][crc>>24];
        p+=4; bytes-=4;
    }
    while(bytes--) crc=(crc>>8)^table[0][(crc^*p++)&255];
    return ~crc;
}
