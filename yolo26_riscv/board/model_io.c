#include "model_io.h"
uint32_t board_crc32(const void *data,size_t bytes) {
    const unsigned char *p=data;
    uint32_t crc=~0u;
    while(bytes--) {
        crc^=*p++;
        for(unsigned i=0;i<8;++i) crc=(crc>>1)^(0xedb88320u & (0u-(crc&1)));
    }
    return ~crc;
}
