/* Real production packing/AMU code, with handoff call counts and bit checks.
 * QEMU does not model the physical cache; hardware acceptance is still needed. */
#include "../../amu.h"
#include "../memory.h"
#include "../diagnostics.h"
static float input[3*64] BOARD_CLP_DATA;
static uint16_t half[3*64] BOARD_CLP_DATA;
static uint16_t scratch[32*32] BOARD_CLP_DATA;
static uint16_t weights[32*32] BOARD_CLP_DATA;
static struct { float before[16],data[16*64],after[16]; } output BOARD_CLP_DATA;
static float ordinary[16*64] __attribute__((aligned(64)));
static float source_weights[16*3];
static uintptr_t watched;
static size_t watched_bytes;
static unsigned calls[3];
volatile BoardDiagnostics board_diag;
void cache_audit(unsigned kind,const void *p,size_t bytes) {
    uintptr_t at=(uintptr_t)p;
    if(bytes && at<watched+watched_bytes && watched<at+bytes) ++calls[kind];
}
static void watch(const void *p,size_t n) {
    watched=(uintptr_t)p; watched_bytes=n; calls[0]=calls[1]=calls[2]=0;
}
static uintptr_t host_call(uintptr_t op,uintptr_t arg) {
    register uintptr_t a0 __asm__("a0")=op,a1 __asm__("a1")=arg;
    __asm__ volatile(".option push\n.option norvc\n.balign 16\nslli zero,zero,31\n"
                     "ebreak\nsrai zero,zero,7\n.option pop" : "+r"(a0),"+r"(a1) :: "memory");
    return a0;
}
__attribute__((noreturn)) void platform_exit(uintptr_t code) {
    if(code) host_call(4,(uintptr_t)"OWNERSHIP FAIL\n");
    uintptr_t args[2]={0x20026,code}; host_call(0x20,(uintptr_t)args);
    for(;;) __asm__ volatile("wfi");
}
void platform_trap(uintptr_t cause,uintptr_t pc,uintptr_t value,const uintptr_t *r) {
    (void)cause; (void)pc; (void)value; (void)r; platform_exit(99);
}
#if !YOLO_FAST
void board_diag_phase(unsigned phase) { (void)phase; }
int board_pack_begin(size_t f,size_t c,size_t k,const void *i,const void *h,void *o,size_t b) {
    (void)f;(void)c;(void)k;(void)i;(void)h;(void)o;(void)b; return 0;
}
int board_pack_mark(unsigned s,size_t c,size_t i,int64_t r,size_t v,const size_t t[3],uint64_t n) {
    (void)s;(void)c;(void)i;(void)r;(void)v;(void)t;(void)n; return 0;
}
#endif
static void require(int valid,unsigned code) { if(!valid) platform_exit(code); }
static void run_output(float *dest,int acquire) {
    Conv2D p={1,3,1,64,16,1,1,1,1,0,0,1};
    AmuPackedWeights w;
    require(!amu_pack_weights(source_weights,16,3,3,weights,sizeof(weights),&w),10);
    if(acquire) require(!amu_output_acquire(dest,sizeof(output.data)),11);
    if(acquire && dest==output.data && !board_diag.options)
        require(amu_output_acquire(dest,sizeof(output.data))!=0,12);
    for(size_t first=0;first<64;first+=32) {
        AmuPackedInput block;
        require(!amu_pack_conv_input_rvv(&p,input,0,first,32,scratch,sizeof(scratch),&block),13);
        require(!amu_matmul_packed_fp16(&w,&block,dest+first,64),14);
    }
    amu_output_release();
    for(size_t c=0;c<16;++c) for(size_t pos=0;pos<64;++pos) {
        float expected=0;
        for(size_t k=0;k<3;++k) expected+=source_weights[c*3+k]*input[k*64+pos];
        union {float f;uint32_t u;} a={dest[c*64+pos]},b={expected};
        require(a.u==b.u,15);
    }
}
int main(void) {
    require(!amu_init(),1);
    for(size_t i=0;i<192;++i) input[i]=(float)(i%7);
    for(size_t i=0;i<48;++i) source_weights[i]=(float)(i%3+1);
    for(size_t i=0;i<16;++i) output.before[i]=output.after[i]=-123.0f;
    amu_input_flags=AMU_INPUT_PRECONVERT|AMU_INPUT_OWNED;
    Conv2D p={1,3,1,64,16,1,1,1,1,0,0,1};
    watch(half,sizeof(half));
    require(!amu_input_acquire(scratch,sizeof(scratch),half,sizeof(half)),2);
    require(!amu_input_prepare(&p,input,half,192),3);
    amu_input_release();
    require(!calls[0] && !calls[1] && !calls[2],4);
    p.kernel_h=p.kernel_w=3; p.pad_h=p.pad_w=1;
    require(!amu_input_acquire(scratch,sizeof(scratch),half,sizeof(half)),5);
    require(!amu_input_prepare(&p,input,half,192),6);
    require(calls[1]==1 && calls[2]==0,7);
    amu_input_release();
    require(calls[1]==1 && calls[2]==1,8);
    amu_input_flags=0;
    for(unsigned repeat=0;repeat<2;++repeat) {
        watch(output.data,sizeof(output.data));
        run_output(output.data,1);
        require(calls[1]==1 && calls[2]==1,16);
        watch(output.data,sizeof(output.data));
        run_output(output.data,0);
        require(calls[1]==32 && calls[2]==32,17);
        for(size_t i=0;i<192;++i) input[i]=-input[i];
    }
    /* Unaligned range and trace must not enter whole-buffer ownership. */
    watch(output.data,sizeof(output.data));
    require(!amu_output_acquire(output.data+1,sizeof(output.data)-4),18);
    amu_output_release(); require(!calls[1] && !calls[2],19);
    board_diag.options=DIAG_TRACE;
    watch(output.data,sizeof(output.data)); run_output(output.data,1);
    require(calls[1]==32 && calls[2]==32,20);
    board_diag.options=0;
    watch(ordinary,sizeof(ordinary)); run_output(ordinary,1);
    require(calls[1]==32 && calls[2]==32,21);
    /* Another destination is not covered by an outstanding ownership scope. */
    require(!amu_output_acquire(output.data,sizeof(output.data)),22);
    watch(ordinary,sizeof(ordinary)); run_output(ordinary,0);
    require(calls[1]==32 && calls[2]==32,23);
    amu_output_release();
    require(amu_output_acquire(0,64)!=0,24);
    require(!amu_output_acquire(output.data,sizeof(output.data)),25);
    amu_output_release();
    for(size_t i=0;i<16;++i) require(output.before[i]==-123.0f && output.after[i]==-123.0f,26);
    host_call(4,(uintptr_t)"OWNERSHIP PASS: unused half=0; used half=1/1; AMU output=1/1 vs 32/32; fallback/reuse/guards\n");
    platform_exit(0);
}
