#include "cache.h"
#include "uart.h"
#include "../amu.h"
#include "diagnostics.h"
static float source[16] __attribute__((aligned(64)));
static float destination[16] __attribute__((aligned(64)));
static float left[16] __attribute__((aligned(64)));
static float right[16] __attribute__((aligned(64)));
volatile uint32_t board_test_stage;
int board_sync_selftest(void);
int board_selftest(void) {
    board_test_stage=1;
    board_clean((const void *)&board_test_stage,sizeof(board_test_stage));
    board_diag_phase(PHASE_COMPUTE);
    for(unsigned i=0;i<16;++i) { source[i]=(float)i+2; destination[i]=123; }
    /* Direct assembly intentionally bypasses the coherent intrinsic wrappers.
     * Exercise a partial-line writer and preserve untouched neighbours. */
    board_clean(source,sizeof(source));
    board_prepare_write(destination+1,sizeof(float));
    uintptr_t one=1;
    __asm__ volatile("vsetvli zero,%2,e32,m1,ta,ma\n"
                     "vle32.v v8,(%0)\nvse32.v v8,(%1)"
                     :: "r"(source),"r"(destination+1),"r"(one) : "v8","memory");
    board_finish_write(destination+1,sizeof(float));
    if(destination[1]!=2 || destination[0]!=123 || destination[2]!=123) {
        unsigned at=destination[0]!=123 ? 0 : destination[1]!=2 ? 1 : 2;
        union { float f; uint32_t u; } got={.f=destination[at]}, want={.f=at==1 ? 2 : 123};
        board_diag.element=at; board_diag.observed=got.u; board_diag.expected=want.u;
        board_diag_error(4,-1); return -1;
    }
    board_test_stage=2;
    board_clean((const void *)&board_test_stage,sizeof(board_test_stage));
    left[0]=2; right[0]=3;
    destination[0]=-1;
    if(amu_matmul_fp16(left,right,destination,1,1,1,1,1,1)) return -2;
    if(destination[0]!=6 || destination[1]!=2) {
        unsigned at=destination[0]!=6 ? 0 : 1;
        union { float f; uint32_t u; } got={.f=destination[at]}, want={.f=at==0 ? 6 : 2};
        board_diag.element=at; board_diag.observed=got.u; board_diag.expected=want.u;
        board_diag_error(4,-3); return -3;
    }
    board_test_stage=4;
    board_clean((const void *)&board_test_stage,sizeof(board_test_stage));
    int sync_status=board_sync_selftest();
    if(sync_status) return sync_status;
    board_test_stage=3;
    board_clean((const void *)&board_test_stage,sizeof(board_test_stage));
    return 0;
}
#ifdef BOARD_SELFTEST_STANDALONE
volatile uintptr_t board_error,board_trap_cause,board_trap_pc,board_trap_value;
__attribute__((noreturn)) void platform_exit(uintptr_t code) {
    board_error=code;
    if(code && !board_diag.error) board_diag_error((unsigned)code,0);
    board_clean((const void *)&board_error,sizeof(board_error));
    for(;;) __asm__ volatile("wfi");
}
__attribute__((noreturn)) void platform_trap(uintptr_t cause,uintptr_t pc,uintptr_t value,const uintptr_t *registers) {
    board_trap_cause=cause; board_trap_pc=pc; board_trap_value=value;
    board_diag_trap(cause,pc,value,registers);
    platform_exit(0x100);
}
static void message(const char *p) { (void)board_uart_text(p); }
static void hex_value(uint64_t value) { (void)board_uart_hex(value); }
int main(void) {
    board_diag_init();
    board_diag_stage(DIAG_UART);
    int status=board_uart_init();
    /* Debug output failure must not prevent cache/AMU validation. */
    if(!status) message("[YOLO] UART OK\n");
    board_diag_stage(DIAG_AMU_INIT);
    int result=amu_init();
    if(result) { board_diag_error(3,result); message("AMU INIT FAIL\n"); platform_exit(3); }
    board_diag_stage(DIAG_SELFTEST);
    result=board_selftest();
    if(result) {
        board_diag_error(4,result);
        message("CACHE/AMU TEST FAIL stage="); hex_value(board_test_stage);
        message(" round="); hex_value(board_diag.frame);
        message(" buffer="); hex_value(board_diag.tensor);
        message(" element="); hex_value(board_diag.element);
        message(" got="); hex_value(board_diag.observed);
        message(" expected="); hex_value(board_diag.expected);
        message("\n"); platform_exit(4);
    }
    board_diag_stage(DIAG_DONE);
    message("RVV->AMU->RVV REUSE 32 ROUNDS PASS\nCACHE/RVV/AMU TEST PASS\n");
    platform_exit(0);
}
#endif
