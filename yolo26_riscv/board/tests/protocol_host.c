/* Native harness: runs the actual board packet parser; no hardware claims. */
#include <stdio.h>
#include <stdlib.h>
#include "../protocol.h"
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
int board_model_run(void) { output[0]=(float)++runs; output[4]=0.9f; return 0; }
int main(void) {
    setvbuf(stdout,NULL,_IONBF,0);
    if(board_crc32("123456789",9)!=0xcbf43926u) return 1;
    board_rpc_loop();
}
