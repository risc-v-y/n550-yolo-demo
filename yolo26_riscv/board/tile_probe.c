#include "tile_probe.h"
#include "pcie.h"
#include "model_io.h"
#include "cache.h"
#include "uart.h"
TileProbe board_tile_probe __attribute__((aligned(64)));
_Static_assert(sizeof(TileRecord)==8320 && sizeof(TileProbe)==25024, "probe ABI");
static unsigned slot, active, captured;
static const unsigned starts[3]={0,15328,18848};
void tile_probe_begin(const void *input,size_t bytes,const void *weights,size_t weight_bytes) {
    unsigned char *p=(unsigned char *)&board_tile_probe;
    for(size_t i=0;i<sizeof(board_tile_probe);++i) p[i]=0;
    uint32_t *h=board_tile_probe.header;
    h[0]=0x31505459; h[1]=1; h[2]=sizeof(board_tile_probe); h[3]=3;
    h[4]=board_crc32(input,bytes); h[5]=board_crc32(weights,weight_bytes);
    slot=active=captured=0;
}
int tile_probe_select(size_t first) {
    for(unsigned i=0;i<3;++i) if(first==starts[i]) {
        slot=i; active=1; captured=0; board_tile_probe.records[i].meta[0]=(uint32_t)first;
        return 1;
    }
    active=0; return 0;
}
int tile_probe_amu(const uint16_t *left,const uint16_t *right,const float *output,
                   size_t m,size_t n,size_t k,size_t lda,size_t ldb,size_t ldc) {
    if(!active || captured || m!=16 || n!=32 || k!=27 || lda!=27 || ldb!=32 || ldc!=43264) {
        board_tile_probe.header[6]=1; return -1;
    }
    TileRecord *r=board_tile_probe.records+slot;
    r->meta[1]=m; r->meta[2]=n; r->meta[3]=k;
    r->meta[4]=lda; r->meta[5]=ldb; r->meta[6]=ldc;
    uintptr_t value;
    __asm__ volatile("csrr %0, frm":"=r"(value)); r->meta[7]=value;
    __asm__ volatile("csrr %0, xmcsr":"=r"(value)::"memory"); r->meta[8]=value;
    __asm__ volatile("csrr %0, xmfflags":"=r"(value)::"memory"); r->meta[9]=value;
    uintptr_t addresses[3]={(uintptr_t)left,(uintptr_t)right,(uintptr_t)output};
    for(unsigned i=0;i<3;++i) { r->meta[10+2*i]=(uint32_t)addresses[i]; r->meta[11+2*i]=addresses[i]>>32; }
    /* Observe after AMU completion/cache invalidation, before RVV bias.
     * Read every physical packed lane, including inactive K and M padding. */
    for(unsigned i=0;i<1024;++i) { r->left[i]=left[i]; r->right[i]=right[i]; }
    for(unsigned i=0;i<16;++i) for(unsigned j=0;j<32;++j) r->before[i*32+j]=output[i*ldc+j];
    captured=1; return 0;
}
void tile_probe_bias(const float *output,const float *bias,size_t stride) {
    if(!active || !captured || stride!=43264) { board_tile_probe.header[6]=2; return; }
    TileRecord *r=board_tile_probe.records+slot;
    for(unsigned i=0;i<16;++i) {
        r->bias[i]=bias[i];
        for(unsigned j=0;j<32;++j) r->after[i*32+j]=output[i*stride+j];
    }
    board_tile_probe.header[7]|=1u<<slot;
    board_uart_text("[YOLO] TILE PROBE first="); board_uart_u64(r->meta[0]); board_uart_text(" captured\n");
    active=0;
}
int tile_probe_finish(void) {
    return board_tile_probe.header[6] || board_tile_probe.header[7]!=7 ? -1 : 0;
}
