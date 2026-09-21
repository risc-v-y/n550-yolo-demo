#include "cache.h"
#include "uart.h"
#include "../amu.h"
static float source[16] __attribute__((aligned(64)));
static float destination[16] __attribute__((aligned(64)));
static float left[16] __attribute__((aligned(64)));
static float right[16] __attribute__((aligned(64)));
volatile uint32_t board_test_stage;
int board_selftest(void) {
    board_test_stage=1;
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
    if(destination[1]!=2 || destination[0]!=123 || destination[2]!=123) return -1;
    board_test_stage=2;
    left[0]=2; right[0]=3;
    destination[0]=-1;
    if(amu_matmul_fp16(left,right,destination,1,1,1,1,1,1)) return -2;
    if(destination[0]!=6 || destination[1]!=2) return -3;
    board_test_stage=3;
    return 0;
}
#ifdef BOARD_SELFTEST_STANDALONE
volatile uintptr_t board_error,board_trap_cause,board_trap_pc,board_trap_value;
__attribute__((noreturn)) void platform_exit(uintptr_t code) {
    board_error=code;
    for(;;) __asm__ volatile("wfi");
}
__attribute__((noreturn)) void platform_trap(uintptr_t cause,uintptr_t pc,uintptr_t value) {
    board_trap_cause=cause; board_trap_pc=pc; board_trap_value=value;
    platform_exit(0x100);
}
static void message(const char *p) { while(*p) if(board_uart_put((unsigned char)*p++)) break; }
int main(void) {
    int status=board_uart_init();
    if(status) platform_exit((uintptr_t)(0x20-status));
    message("UART OK\r\n");
    if(amu_init()) { message("AMU INIT FAIL\r\n"); platform_exit(3); }
    if(board_selftest()) { message("CACHE/AMU TEST FAIL\r\n"); platform_exit(4); }
    message("CACHE/RVV/AMU TEST PASS\r\n");
    platform_exit(0);
}
#endif
