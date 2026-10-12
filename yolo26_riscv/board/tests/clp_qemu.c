/* QEMU-only UART register/exit fixture. Never link into physical firmware. */
#include <stdint.h>
static uint32_t registers[64];
static void semihost(uintptr_t operation,uintptr_t argument) {
    register uintptr_t a0 __asm__("a0")=operation;
    register uintptr_t a1 __asm__("a1")=argument;
    __asm__ volatile(".option push\n.option norvc\n"
                     "slli zero,zero,31\nebreak\nsrai zero,zero,7\n.option pop"
                     : "+r"(a0) : "r"(a1) : "memory");
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
    const uintptr_t block[2]={0x20026,code};
    semihost(0x20,(uintptr_t)block);
    for(;;) __asm__ volatile("wfi");
}
