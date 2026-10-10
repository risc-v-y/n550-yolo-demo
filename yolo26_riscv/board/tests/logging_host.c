#include "../uart.h"
#include "../diagnostics.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
static uint32_t registers[64],dlf_mask=15;
static int tx_ready=1;
static unsigned polls;
static char text[32768];
static size_t used;
uint32_t board_uart_test_read(unsigned offset) {
    if(offset==0x14) { ++polls; return tx_ready ? 0x20 : 0; }
    return registers[offset/4];
}
void board_uart_test_write(unsigned offset,uint32_t value) {
    if(offset==0xc0) value &= dlf_mask;
    if(offset==0 && !(registers[3]&0x80)) {
        assert(used+1<sizeof(text)); text[used++]=(char)value; text[used]=0;
    } else registers[offset/4]=value;
}
int main(void) {
    board_diag_init();
    for(unsigned bits=4;bits<=6;++bits) {
        dlf_mask=(1u<<bits)-1;
        assert(board_uart_init()==0);
        assert(board_uart_dlf_bits==bits);
        assert(board_uart_baud==(bits==6 ? 115273 : 114942));
    }
    used=0;
    assert(!board_uart_text("hello\n"));
    assert(!board_uart_u64(UINT64_MAX)); board_uart_text(" ");
    assert(!board_uart_i64(INT64_MIN)); board_uart_text(" ");
    assert(!board_uart_hex(0x123));
    assert(!strcmp(text,"hello\r\n18446744073709551615 -9223372036854775808 0x0000000000000123"));
    used=0;
    board_diag_stage(DIAG_AMU_INIT); board_diag_stage(DIAG_SELFTEST); board_diag_stage(DIAG_READY);
    board_diag.frame=7; board_diag.options=DIAG_TRACE;
    board_diag_stage(DIAG_RUN);
    ModelTensor tensor={.dtype=1,.rank=4,.shape={1,3,416,416}};
    ModelNode node={.op=0,.inputs=1,.outputs=1,.in={0},.out={0}};
    board_diag_node(32,&node,&tensor);
    board_diag.phase=PHASE_NODE_END; board_diag.node_cycles=1234; board_diag_event();
    board_diag_stage(DIAG_DONE);
    board_diag_error(0x31,-77);
    uintptr_t regs[32]={0}; board_diag_trap(2,0x80001234,0xdead,regs);
    assert(strstr(text,"AMU_INIT")); assert(strstr(text,"SELFTEST")); assert(strstr(text,"READY"));
    assert(strstr(text,"RUN frame=7 node=32")); assert(strstr(text,"shape=[1,3,416,416]"));
    assert(strstr(text,"node_cycles=1234")); assert(strstr(text,"DONE frame=7"));
    assert(strstr(text,"error=0x0000000000000031 detail=-77"));
    assert(strstr(text,"mcause=0x0000000000000002 mepc=0x0000000080001234 mtval=0x000000000000dead"));
    tx_ready=0; polls=0; unsigned previous=used;
    assert(board_uart_text("unavailable")==-7);
    assert(polls==4 && used==previous && board_diag.tx_errors==1);
    for(unsigned i=0;i<100;++i) board_diag_event();
    assert(polls==4); /* Disabled UART cannot repeatedly stall inference. */
    tx_ready=1; dlf_mask=7;
    assert(board_uart_init()==-2);
    registers[0x7c/4]=1;
    assert(board_uart_init()==-1);
    registers[0x7c/4]=0; dlf_mask=15;
    assert(!board_uart_init());
    puts("PASS debug UART formatting, divider, progress, errors/traps and timeout disable");
    return 0;
}
