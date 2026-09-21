#include "../accel_model.h"
#include "../amu.h"
#include "model_generated.h"
#include "cache.h"
#include "uart.h"
#include "protocol.h"
extern const unsigned char model_weights[], model_weights_end[];
static unsigned char arena[MODEL_ARENA_BYTES] __attribute__((aligned(64)));
/* Inspect these symbols with the team's debugger when UART cannot start. */
volatile uintptr_t board_error, board_trap_cause, board_trap_pc, board_trap_value;
volatile uint32_t board_current_node;
int board_selftest(void);
__attribute__((noreturn)) void platform_exit(uintptr_t code) {
    board_error=code;
    for(;;) __asm__ volatile("wfi");
}
__attribute__((noreturn)) void platform_trap(uintptr_t cause,uintptr_t pc,uintptr_t value) {
    board_trap_cause=cause; board_trap_pc=pc; board_trap_value=value;
    platform_exit(0x100);
}
unsigned char *board_model_input(void) { return arena+model_tensors[0].offset; }
const unsigned char *board_model_output(void) { return arena+model_tensors[MODEL_OUTPUT_ID].offset; }
uint32_t board_model_input_size(void) { return model_tensors[0].bytes; }
uint32_t board_model_output_size(void) { return model_tensors[MODEL_OUTPUT_ID].bytes; }
int board_model_run(void) {
    /* Reject corrupt/nonfinite inputs even if the UART CRC was valid. */
    const unsigned char *input=board_model_input();
    for(size_t i=0;i<board_model_input_size();i+=4) {
        uint32_t bits=(uint32_t)input[i] | (uint32_t)input[i+1]<<8 |
                      (uint32_t)input[i+2]<<16 | (uint32_t)input[i+3]<<24;
        if((bits&0x7f800000)==0x7f800000) return -1;
    }
    accel_rvv_nodes=accel_amu_nodes=accel_scalar_nodes=accel_alias_nodes=0;
    for(size_t step=0;step<MODEL_NODE_COUNT;++step) {
        board_current_node=(uint32_t)step;
        const ModelNode *node=model_nodes+step;
        for(size_t i=0;i<node->inputs;++i) {
            const ModelTensor *t=model_tensors+node->in[i];
            board_clean((t->constant ? model_weights : arena)+t->offset,t->bytes);
        }
        if(accel_model_node(node,model_tensors,arena,model_weights)) return -2;
    }
    return 0;
}
int main(void) {
    if((size_t)(model_weights_end-model_weights)!=MODEL_CONSTANT_BYTES) platform_exit(1);
    if(model_tensors[0].dtype!=1 || board_model_input_size()!=3*416*416*4 ||
       board_model_output_size()!=7200) platform_exit(2);
    int status=board_uart_init();
    if(status) platform_exit((uintptr_t)(0x20-status));
    if(amu_init()) platform_exit(3);
    if(board_selftest()) platform_exit(4);
    board_rpc_loop();
    return 0;
}
