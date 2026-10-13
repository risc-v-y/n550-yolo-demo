/* QEMU-only UART register/exit fixture. Never link into physical firmware. */
#include <stdint.h>
static uint32_t registers[64];
static uintptr_t semihost(uintptr_t operation,uintptr_t argument) {
    register uintptr_t a0 __asm__("a0")=operation;
    register uintptr_t a1 __asm__("a1")=argument;
    __asm__ volatile(".option push\n.option norvc\n.balign 16\n"
                     "slli zero,zero,31\nebreak\nsrai zero,zero,7\n.option pop"
                     : "+r"(a0) : "r"(a1) : "memory");
    return a0;
}
uint32_t board_uart_test_read(unsigned offset) {
    if(offset==0x14) return 0x60;
    if(offset==0x7c) return 0;
    return registers[offset/4];
}
void board_uart_test_write(unsigned offset,uint32_t value) {
    if(offset==0 && !(registers[0x0c/4]&0x80)) {
        unsigned char byte=(unsigned char)value;
        semihost(3,(uintptr_t)&byte);
    } else {
        registers[offset/4]=offset==0xc0 ? value&31u : value;
    }
}
__attribute__((noreturn)) void clp_qemu_exit(uintptr_t code) {
#ifndef CLP_MODEL_QEMU
    /* Capture actual RVV/AMU RAM so Python can check it independently.
     * Filename is relative to the isolated QEMU test output directory. */
    static const char name[]="clp-qemu-ddr.bin";
    const uintptr_t open_args[3]={(uintptr_t)name,5,sizeof(name)-1};
    uintptr_t handle=semihost(1,(uintptr_t)open_args);
    if(handle!=(uintptr_t)-1) {
        const uintptr_t write_args[3]={handle,0x80100000,4096};
        if(semihost(5,(uintptr_t)write_args)) code=0xfe;
        if(semihost(2,(uintptr_t)&handle)) code=0xfe;
    } else code=0xfe;
#endif
    const uintptr_t block[2]={0x20026,code};
    semihost(0x20,(uintptr_t)block);
    for(;;) __asm__ volatile("wfi");
}
