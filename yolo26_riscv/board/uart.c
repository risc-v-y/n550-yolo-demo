#include "uart.h"
#include <stddef.h>
#define UART_BASE ((uintptr_t)0x20100000u)
#define UART_CLOCK 10000000u
#define UART_BAUD 115200u
#ifndef UART_POLL_LIMIT
#define UART_POLL_LIMIT 10000000u
#endif
enum { RBR=0x00, THR=0x00, DLL=0x00, IER=0x04, DLH=0x04,
       FCR=0x08, LCR=0x0c, MCR=0x10, LSR=0x14, USR=0x7c, DLF=0xc0 };
uint32_t board_uart_baud, board_uart_dlf_bits;
static uint32_t read_reg(unsigned offset) {
    uint32_t v = *(volatile uint32_t *)(UART_BASE + offset);
    __asm__ volatile("fence iorw,iorw" ::: "memory");
    return v;
}
static void write_reg(unsigned offset, uint32_t value) {
    __asm__ volatile("fence iorw,iorw" ::: "memory");
    *(volatile uint32_t *)(UART_BASE + offset) = value;
    __asm__ volatile("fence iorw,iorw" ::: "memory");
}
int board_uart_init(void) {
    /* No interrupt controller, DMA, RTS/CTS or software flow control required.
     * Clock/reset/pin routing is established by the FPGA platform. */
    write_reg(LCR, read_reg(LCR) & ~0x80u);
    write_reg(IER, 0);
    write_reg(MCR, 0);
    write_reg(FCR, 7);
    unsigned tries;
    for (tries=0; tries<UART_POLL_LIMIT && (read_reg(USR)&1); ++tries) { }
    if (tries == UART_POLL_LIMIT) return -1;
    uint32_t old = read_reg(DLF);
    write_reg(DLF, 0x3f);
    uint32_t mask = read_reg(DLF) & 0x3f;
    write_reg(DLF, old);
    if (mask != 15 && mask != 31 && mask != 63) return -2;
    board_uart_dlf_bits = mask == 15 ? 4 : mask == 31 ? 5 : 6;
    uint32_t scale = mask + 1;
    uint32_t divisor = (UART_CLOCK * scale + 8 * UART_BAUD) / (16 * UART_BAUD);
    write_reg(LCR, 0x83);
    if ((read_reg(LCR)&0xff) != 0x83) return -3;
    write_reg(DLL, (divisor / scale) & 255);
    write_reg(DLH, (divisor / scale) >> 8);
    write_reg(DLF, divisor % scale);
    if ((read_reg(DLL)&255) != ((divisor/scale)&255) ||
        (read_reg(DLH)&255) != ((divisor/scale)>>8) ||
        (read_reg(DLF)&mask) != divisor%scale) return -4;
    write_reg(LCR, 3); /* 8N1 */
    if ((read_reg(LCR)&255) != 3) return -5;
    /* More than eight 10 MHz APB cycles before the first transfer. */
    for (unsigned i=0; i<16; ++i) (void)read_reg(LCR);
    board_uart_baud = (UART_CLOCK * scale) / (16 * divisor);
    return 0;
}
int board_uart_get(unsigned char *value) {
    for (unsigned i=0; i<UART_POLL_LIMIT; ++i) {
        uint32_t status = read_reg(LSR);
        if (status & 0x1e) {
            if (status & 1) (void)read_reg(RBR);
            return -2;
        }
        if (status & 1) { *value = (unsigned char)read_reg(RBR); return 0; }
    }
    return -1;
}
int board_uart_put(unsigned char value) {
    for (unsigned i=0; i<UART_POLL_LIMIT; ++i)
        if (read_reg(LSR) & 0x20) { write_reg(THR, value); return 0; }
    return -1;
}
