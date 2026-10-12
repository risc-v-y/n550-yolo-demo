/* Isolated CLP test. Code/stack/diagnostics stay in the normal DDR region.
 * Shared payloads are accessed ONLY by explicit RVV and AMU instructions.
 * Do not dereference the global DDR alias with scalar C loads/stores. */
#include "cache.h"
#include "diagnostics.h"
#include "uart.h"
#include "../amu.h"
#include <xewmatrix_intrinsic.h>

#ifndef CLP_PROBE_SIM
#define CLP_RVV_BASE ((uintptr_t)0x10000000u)
#define CLP_DDR_BASE ((uintptr_t)0xf0000000u)
#else
/* QEMU validates arithmetic/control only; it has no S2C CLP remapping. */
#define CLP_RVV_BASE ((uintptr_t)0x80100000u)
#define CLP_DDR_BASE ((uintptr_t)0x80100000u)
#endif
enum { REGION_BYTES=4096, A=0, B=64, C=128, D=192, ROUNDS=32 };
#define GUARD 0x35553555u
#define OUTPUT_GUARD 0xc49a4000u
volatile uint32_t board_test_stage;
volatile uintptr_t board_error,board_trap_cause,board_trap_pc,board_trap_value;

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
static void add_tag(float tag) {
    uintptr_t source=CLP_RVV_BASE+C,destination=CLP_RVV_BASE+D;
    __asm__ volatile("vsetivli zero,1,e32,m1,ta,ma\n"
                     "vle32.v v8,(%0)\nvfadd.vf v8,v8,%2\nvse32.v v8,(%1)"
                     :: "r"(source),"r"(destination),"f"(tag) : "v8","memory");
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
static int run_clp(void) {
    static const uint16_t half[4]={0x4000,0xbc00,0x3c00,0x4200};
    static const int integer[4]={2,-1,1,3};
    board_test_stage=1;
    board_clean((const void *)&board_test_stage,sizeof(board_test_stage));
    fill32(0,REGION_BYTES/4,GUARD);
    clp_fence();
    for(unsigned at=0;at<REGION_BYTES;at+=4)
        if(compare(at,GUARD)) return -1;
    board_test_stage=4;
    board_clean((const void *)&board_test_stage,sizeof(board_test_stage));
    for(unsigned round=0;round<ROUNDS;++round) {
        board_diag.frame=round;
        board_diag.phase=PHASE_AMU_SUBMIT;
        board_diag_commit();
        fill32(C,16,OUTPUT_GUARD);
        fill32(D,16,OUTPUT_GUARD);
        write16(A,half[round%4]);
        write16(B,0x4200); /* FP16 3 */
        clp_fence(); /* RVV CLP store -> AMU DDR load, no cache operations. */
        uintptr_t old,flags;
        __riscv_msettilem(old,1,w);
        __riscv_msettilen(old,1,w);
        __riscv_msettilek(old,1,w);
        __riscv_mzero("acc0");
        __riscv_mlae16("tr0",(const void *)(CLP_DDR_BASE+A),64);
        __riscv_mlbe16("tr1",(const void *)(CLP_DDR_BASE+B),64);
        __riscv_mfmacc_s_h("acc0","tr1","tr0");
        __riscv_msce32("acc0",(void *)(CLP_DDR_BASE+C),64);
        __asm__ volatile("csrr %0,xmfflags" : "=r"(flags) : : "memory");
        clp_fence(); /* Completed AMU DDR store -> RVV CLP load. */
        add_tag((float)(round+1));
        clp_fence(); /* Finish RVV stores before validation and next reuse. */
        (void)old;
        board_diag.amu_flags=flags;
        board_diag.phase=PHASE_CHECK;
        board_diag_commit();
        int product=integer[round%4]*3;
        /* Active data plus untouched neighbours, exact integer FP values. */
        for(unsigned at=0;at<256;at+=4) {
            uint32_t expected=GUARD;
            if(at==A) expected=(GUARD&0xffff0000u)|half[round%4];
            else if(at==B) expected=(GUARD&0xffff0000u)|0x4200;
            else if(at>=C && at<D+64) expected=OUTPUT_GUARD;
            if(at==C) expected=float_bits((float)product);
            else if(at==D) expected=float_bits((float)(product+(int)round+1));
            if(compare(at,expected)) return -1;
        }
    }
    for(unsigned at=256;at<REGION_BYTES;at+=4)
        if(compare(at,GUARD)) return -1;
    board_test_stage=3;
    board_clean((const void *)&board_test_stage,sizeof(board_test_stage));
    return 0;
}
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
static void message(const char *text) { (void)board_uart_text(text); }
static void hex(uint64_t value) { (void)board_uart_hex(value); }
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
    if(run_clp()) {
        message("CLP TEST FAIL stage="); hex(board_test_stage);
        message(" round="); hex(board_diag.frame);
        message(" buffer="); hex(board_diag.tensor);
        message(" element="); hex(board_diag.element);
        message(" got="); hex(board_diag.observed);
        message(" expected="); hex(board_diag.expected);
        message("\n"); platform_exit(4);
    }
    board_diag_stage(DIAG_DONE);
    message("CLP RVV->AMU->RVV REUSE 32 ROUNDS PASS\n");
    platform_exit(0);
}
