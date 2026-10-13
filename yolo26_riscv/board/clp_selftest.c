/* Isolated CLP test. Code/stack/diagnostics stay in the normal DDR region.
 * Shared payloads are accessed ONLY by explicit RVV and AMU instructions.
 * Do not dereference the global DDR alias with scalar C loads/stores. */
#include "cache.h"
#include "diagnostics.h"
#include "uart.h"
#include "../amu.h"
#include "memory.h"
#include <xewmatrix_intrinsic.h>

#ifdef CLP_PROBE_EMBEDDED
#define CLP_RVV_BASE BOARD_CLP_RVV_BASE
#define CLP_DDR_BASE BOARD_CLP_DDR_BASE
#elif !defined(CLP_PROBE_SIM)
#define CLP_RVV_BASE ((uintptr_t)0x10000000u)
#define CLP_DDR_BASE ((uintptr_t)0xf0000000u)
#else
/* QEMU validates arithmetic/control only; it has no S2C CLP remapping. */
#define CLP_RVV_BASE ((uintptr_t)0x80100000u)
#define CLP_DDR_BASE ((uintptr_t)0x80100000u)
#endif
enum { REGION_BYTES=4096, BLOCK_BYTES=512, ROW_BYTES=64,
       A=0, B=512, C=1024, D=1536, ROUNDS=32 };
/* One guard row before/after the active rows; remaining rows are guards too.
 * A is M x K, B stores N x K (transposed GEMM right operand), C/D are M x N. */
static const unsigned shapes[][3]={{3,5,17},{1,1,1},{2,3,32},{3,5,31}};
_Static_assert(D+BLOCK_BYTES<=REGION_BYTES,"CLP buffers fit reserved DDR");
#define GUARD 0x35553555u
#define OUTPUT_GUARD 0xc49a4000u
volatile uint32_t board_test_stage;
#ifndef CLP_PROBE_EMBEDDED
volatile uintptr_t board_error,board_trap_cause,board_trap_pc,board_trap_value;
#endif
static void message(const char *text) { (void)board_uart_text(text); }
static void hex(uint64_t value) { (void)board_uart_hex(value); }

static void clp_fence(void) {
    /* Cover both memory and I/O PMA classifications for this platform probe.
     * This orders accesses; AMU completion additionally needs xmfflags. */
    __asm__ volatile("fence iorw,iorw" ::: "memory");
}
static void fill32(unsigned offset,unsigned words,uint32_t bits) {
    uintptr_t address=CLP_RVV_BASE+offset;
    while(words) {
        uintptr_t vl;
        __asm__ volatile("vsetvli %0,%3,e32,m1,ta,ma\n"
                         "vmv.v.x v8,%2\nvse32.v v8,(%1)"
                         : "=&r"(vl) : "r"(address),"r"(bits),"r"((uintptr_t)words)
                         : "v8","memory");
        address+=vl*4; words-=(unsigned)vl;
    }
}
static void write16(unsigned offset,uint16_t bits) {
    uintptr_t address=CLP_RVV_BASE+offset;
    __asm__ volatile("vsetivli zero,1,e16,m1,ta,ma\n"
                     "vmv.v.x v8,%1\nvse16.v v8,(%0)"
                     :: "r"(address),"r"((uintptr_t)bits) : "v8","memory");
}
static uint32_t read32(unsigned offset) {
    uintptr_t address=CLP_RVV_BASE+offset,bits;
    __asm__ volatile("vsetivli zero,1,e32,m1,ta,ma\n"
                     "vle32.v v8,(%1)\nvmv.x.s %0,v8"
                     : "=r"(bits) : "r"(address) : "v8","memory");
    return (uint32_t)bits;
}
static void add_tag(unsigned offset,unsigned count,float tag) {
    uintptr_t source=CLP_RVV_BASE+C+offset,destination=CLP_RVV_BASE+D+offset;
    while(count) {
        uintptr_t vl;
        __asm__ volatile("vsetvli %0,%3,e32,m1,ta,ma\n"
                         "vle32.v v8,(%1)\nvfadd.vf v8,v8,%4\nvse32.v v8,(%2)"
                         : "=&r"(vl) : "r"(source),"r"(destination),"r"((uintptr_t)count),"f"(tag)
                         : "v8","memory");
        source+=vl*4; destination+=vl*4; count-=(unsigned)vl;
    }
}
static uint32_t float_bits(float value) {
    union { float f; uint32_t u; } number={.f=value};
    return number.u;
}
static int compare(unsigned offset,uint32_t expected) {
    uint32_t actual=read32(offset);
    if(actual==expected) return 0;
    board_diag.tensor=offset/64;
    board_diag.element=(offset%64)/4;
    board_diag.observed=actual;
    board_diag.expected=expected;
    board_diag_error(4,-200-(int)(offset/64));
    return -1;
}
static int value(unsigned round,unsigned row,unsigned k,unsigned side) {
    return (int)((round*3+row*5+k*(side ? 3 : 2)+side)%7)-3;
}
static uint16_t half_integer(int number) {
    static const uint16_t magnitude[]={0,0x3c00,0x4000,0x4200};
    return magnitude[number<0 ? -number : number] | (number<0 ? 0x8000 : 0);
}
static uint16_t expected_half(unsigned round,unsigned row,unsigned col,unsigned rows,unsigned k,unsigned side) {
    return row>=1 && row<=rows && col<k ? half_integer(value(round,row-1,col,side)) : 0x3555;
}
static int expected_product(unsigned round,unsigned row,unsigned col,unsigned k) {
    int sum=0;
    for(unsigned at=0;at<k;++at) sum+=value(round,row,at,0)*value(round,col,at,1);
    return sum;
}
static int run_clp(void) {
    board_test_stage=1;
    board_clean((const void *)&board_test_stage,sizeof(board_test_stage));
    fill32(0,REGION_BYTES/4,GUARD);
    clp_fence();
    for(unsigned at=0;at<REGION_BYTES;at+=4)
        if(compare(at,GUARD)) return -1;
    board_test_stage=4;
    board_clean((const void *)&board_test_stage,sizeof(board_test_stage));
    for(unsigned round=0;round<ROUNDS;++round) {
        unsigned m=shapes[round%4][0],n=shapes[round%4][1],k=shapes[round%4][2];
        board_diag.frame=round;
        board_diag.phase=PHASE_AMU_SUBMIT;
        board_diag_commit();
        fill32(A,BLOCK_BYTES/4,GUARD);
        fill32(B,BLOCK_BYTES/4,GUARD);
        fill32(C,BLOCK_BYTES/4,OUTPUT_GUARD);
        fill32(D,BLOCK_BYTES/4,OUTPUT_GUARD);
        for(unsigned row=0;row<m;++row)
            for(unsigned col=0;col<k;++col)
                write16(A+(row+1)*ROW_BYTES+col*2,half_integer(value(round,row,col,0)));
        for(unsigned row=0;row<n;++row)
            for(unsigned col=0;col<k;++col)
                write16(B+(row+1)*ROW_BYTES+col*2,half_integer(value(round,row,col,1)));
        clp_fence(); /* RVV CLP store -> AMU DDR load, no cache operations. */
        uintptr_t old,flags;
        __riscv_msettilem(old,m,w);
        __riscv_msettilen(old,n,w);
        __riscv_msettilek(old,k,w);
        __riscv_mzero("acc0");
        __riscv_mlae16("tr0",(const void *)(CLP_DDR_BASE+A+ROW_BYTES),ROW_BYTES);
        __riscv_mlbe16("tr1",(const void *)(CLP_DDR_BASE+B+ROW_BYTES),ROW_BYTES);
        __riscv_mfmacc_s_h("acc0","tr1","tr0");
        __riscv_msce32("acc0",(void *)(CLP_DDR_BASE+C+ROW_BYTES),ROW_BYTES);
        __asm__ volatile("csrr %0,xmfflags" : "=r"(flags) : : "memory");
        clp_fence(); /* Completed AMU DDR store -> RVV CLP load. */
        for(unsigned row=1;row<=m;++row) add_tag(row*ROW_BYTES,n,(float)(round+1));
        clp_fence(); /* Finish RVV stores before validation and next reuse. */
        (void)old;
        board_diag.amu_flags=flags;
        board_diag.phase=PHASE_CHECK;
        board_diag_commit();
        /* Check inputs (including the odd-K high half), inactive rows,
         * output column padding, and guard rows after every round. */
        for(unsigned at=0;at<BLOCK_BYTES;at+=4) {
            unsigned row=at/ROW_BYTES,col=(at%ROW_BYTES)/2;
            uint32_t a=expected_half(round,row,col,m,k,0) |
                       ((uint32_t)expected_half(round,row,col+1,m,k,0)<<16);
            uint32_t b=expected_half(round,row,col,n,k,1) |
                       ((uint32_t)expected_half(round,row,col+1,n,k,1)<<16);
            if(compare(A+at,a) || compare(B+at,b)) return -1;
            col=(at%ROW_BYTES)/4;
            int active=row>=1 && row<=m && col<n;
            int product=active ? expected_product(round,row-1,col,k) : 0;
            uint32_t c=active ? float_bits((float)product) : OUTPUT_GUARD;
            uint32_t d=active ? float_bits((float)(product+(int)round+1)) : OUTPUT_GUARD;
            if(compare(C+at,c) || compare(D+at,d)) return -1;
        }
    }
    for(unsigned at=D+BLOCK_BYTES;at<REGION_BYTES;at+=4)
        if(compare(at,GUARD)) return -1;
    board_test_stage=3;
    board_clean((const void *)&board_test_stage,sizeof(board_test_stage));
    return 0;
}
/* Reused by model startup; caller initializes AMU and owns stage changes. */
int board_clp_selftest(void) {
    message("CLP SHAPES (M,N,K)=(3,5,17)/(1,1,1)/(2,3,32)/(3,5,31)\n");
    if(run_clp()) {
        message("CLP TEST FAIL stage="); hex(board_test_stage);
        message(" round="); hex(board_diag.frame);
        message(" buffer="); hex(board_diag.tensor);
        message(" element="); hex(board_diag.element);
        message(" got="); hex(board_diag.observed);
        message(" expected="); hex(board_diag.expected);
        message(" M="); (void)board_uart_u64(shapes[board_diag.frame%4][0]);
        message(" N="); (void)board_uart_u64(shapes[board_diag.frame%4][1]);
        message(" K="); (void)board_uart_u64(shapes[board_diag.frame%4][2]);
        message("\n");
        return (int)(int64_t)board_diag.detail;
    }
    message("CLP RVV->AMU->RVV REUSE 32 ROUNDS PASS\n");
    return 0;
}
#ifndef CLP_PROBE_EMBEDDED
/* One stable symbolic final stop, after PASS/FAIL metadata is committed.
 * No intermediate breakpoints or UART reaction timing are necessary. */
__attribute__((noinline,noreturn)) void clp_probe_halt(void) {
    for(;;) __asm__ volatile("wfi");
}
__attribute__((noreturn)) void platform_exit(uintptr_t code) {
    board_error=code;
    board_clean((const void *)&board_error,sizeof(board_error));
#ifdef CLP_PROBE_SIM
    extern void clp_qemu_exit(uintptr_t code) __attribute__((noreturn));
    clp_qemu_exit(code);
#else
    clp_probe_halt();
#endif
}
__attribute__((noreturn)) void platform_trap(uintptr_t cause,uintptr_t pc,uintptr_t value,const uintptr_t *registers) {
    board_trap_cause=cause; board_trap_pc=pc; board_trap_value=value;
    board_clean((const void *)&board_trap_cause,sizeof(board_trap_cause));
    board_clean((const void *)&board_trap_pc,sizeof(board_trap_pc));
    board_clean((const void *)&board_trap_value,sizeof(board_trap_value));
    board_diag_trap(cause,pc,value,registers);
    platform_exit(0x100);
}
int main(void) {
    board_diag_init();
    board_diag_stage(DIAG_UART);
    (void)board_uart_init(); /* Debug UART failure does not stop the test. */
#ifdef CLP_PROBE_SIM
    message("CLP QEMU SIM: unified RAM, no physical remapping/cache validation\n");
#else
    message("CLP TEST RVV=10000000 AMU=f0000000 bytes=4096 rounds=32\n");
#endif
    board_diag_stage(DIAG_AMU_INIT);
    int status=amu_init();
    if(status) { board_diag_error(3,status); message("CLP AMU INIT FAIL\n"); platform_exit(3); }
    board_diag_stage(DIAG_SELFTEST);
    if(board_clp_selftest()) platform_exit(4);
    board_diag_stage(DIAG_DONE);
    platform_exit(0);
}
#endif
