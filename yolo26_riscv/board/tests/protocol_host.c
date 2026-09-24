/* Native harness: runs the actual board packet parser; no hardware claims. */
#include <stdio.h>
#include <stdlib.h>
#include "../protocol.h"
#include "../diagnostics.h"
static unsigned char input[3*416*416*4];
static float output[300*6];
static unsigned runs;
uint32_t board_uart_baud=114943,board_uart_dlf_bits=4;
int board_uart_get(unsigned char *p) {
    int c=getchar();
    if(c==EOF) exit(0);
    *p=(unsigned char)c;
    return 0;
}
int board_uart_put(unsigned char c) { return putchar(c)==EOF ? -1 : 0; }
unsigned char *board_model_input(void) { return input; }
const unsigned char *board_model_output(void) { return (const unsigned char *)output; }
uint32_t board_model_input_size(void) { return sizeof(input); }
uint32_t board_model_output_size(void) { return sizeof(output); }
int board_model_run(void) {
    board_diag_stage(DIAG_RUN); board_diag_event();
    if(input[0]==255) { board_diag.node=17; board_diag_error(0x31,-7); return -7; }
    const ModelTensor tensors[2]={
        {.rank=4,.dtype=1,.shape={1,3,416,416}},
        {.rank=3,.dtype=1,.shape={1,300,6,0}}
    };
    const ModelNode node={.op=OP_CONV,.inputs=1,.outputs=1,.in={0},.out={1}};
    board_diag_node(0,&node,tensors);
    output[0]=(float)++runs; output[4]=0.9f;
    board_diag.phase=PHASE_NODE_END; board_diag.node_cycles=42;
    if(board_diag.options&DIAG_TRACE) board_diag_event();
    board_diag_stage(DIAG_DONE); board_diag_event();
    return 0;
}
int main(void) {
    board_diag_init(); board_diag_stage(DIAG_READY);
    setvbuf(stdout,NULL,_IONBF,0);
    if(board_crc32("123456789",9)!=0xcbf43926u) return 1;
    board_rpc_loop();
}
