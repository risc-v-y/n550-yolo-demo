#include "uart.h"
#include <stddef.h>
#include "diagnostics.h"
#ifdef YOLO_N550_BOARD
#include "cache.h"
#endif
#define UART_BASE ((uintptr_t)0x20100000u)
#define UART_CLOCK 10000000u
#define UART_BAUD 115200u
#ifndef UART_POLL_LIMIT
#define UART_POLL_LIMIT 100000u
#endif
enum { THR=0x00, DLL=0x00, IER=0x04, DLH=0x04,
       FCR=0x08, LCR=0x0c, MCR=0x10, LSR=0x14, USR=0x7c, DLF=0xc0 };
uint32_t board_uart_baud, board_uart_dlf_bits;
volatile int board_uart_status=-6;
#ifdef BOARD_UART_TEST
/* Native register fixture; production always uses 32-bit MMIO and fences. */
uint32_t board_uart_test_read(unsigned offset);
void board_uart_test_write(unsigned offset,uint32_t value);
static uint32_t read_reg(unsigned offset) { return board_uart_test_read(offset); }
static void write_reg(unsigned offset,uint32_t value) { board_uart_test_write(offset,value); }
#else
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
#endif
static int configure_uart(void) {
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
int board_uart_init(void) {
    board_uart_status=configure_uart();
#ifdef YOLO_N550_BOARD
    board_clean((const void *)&board_uart_status,sizeof(board_uart_status));
    board_clean(&board_uart_baud,sizeof(board_uart_baud));
    board_clean(&board_uart_dlf_bits,sizeof(board_uart_dlf_bits));
#endif
    if(board_uart_status) { ++board_diag.tx_errors; board_diag_commit(); }
    return board_uart_status;
}
int board_uart_put(unsigned char value) {
    if(board_uart_status) return board_uart_status;
    for (unsigned i=0; i<UART_POLL_LIMIT; ++i)
        if (read_reg(LSR) & 0x20) { write_reg(THR, value); return 0; }
    board_uart_status=-7;
#ifdef YOLO_N550_BOARD
    board_clean((const void *)&board_uart_status,sizeof(board_uart_status));
#endif
    ++board_diag.tx_errors; board_diag_commit();
    return board_uart_status;
}
int board_uart_text(const char *text) {
    while(*text) {
        if(*text=='\n' && board_uart_put('\r')) return board_uart_status;
        if(board_uart_put((unsigned char)*text++)) return board_uart_status;
    }
    return board_uart_status;
}
int board_uart_u64(uint64_t value) {
    unsigned char digits[20];
    unsigned count=0;
    do { digits[count++]=(unsigned char)('0'+value%10); value/=10; } while(value);
    while(count) if(board_uart_put(digits[--count])) return board_uart_status;
    return board_uart_status;
}
int board_uart_i64(int64_t value) {
    if(value<0 && board_uart_put('-')) return board_uart_status;
    return board_uart_u64(value<0 ? 0u-(uint64_t)value : (uint64_t)value);
}
int board_uart_hex(uint64_t value) {
    const char digits[]="0123456789abcdef";
    if(board_uart_text("0x")) return board_uart_status;
    for(int shift=60;shift>=0;shift-=4)
        if(board_uart_put((unsigned char)digits[(value>>shift)&15])) return board_uart_status;
    return board_uart_status;
}
