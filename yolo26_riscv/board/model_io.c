#include "model_io.h"
uint32_t board_crc32(const void *data,size_t bytes) {
    /* Same reflected IEEE CRC32, four bits per step. Fixed 64-byte table;
     * no initialization, allocation or change to the mailbox checksum. */
    static const uint32_t nibble[16]={
        0x00000000u,0x1db71064u,0x3b6e20c8u,0x26d930acu,
        0x76dc4190u,0x6b6b51f4u,0x4db26158u,0x5005713cu,
        0xedb88320u,0xf00f9344u,0xd6d6a3e8u,0xcb61b38cu,
        0x9b64c2b0u,0x86d3d2d4u,0xa00ae278u,0xbdbdf21cu};
    const unsigned char *p=data;
    uint32_t crc=~0u;
    while(bytes--) {
        crc^=*p++;
        crc=(crc>>4)^nibble[crc&15];
        crc=(crc>>4)^nibble[crc&15];
    }
    return ~crc;
}
