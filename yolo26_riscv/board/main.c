#include "../accel_model.h"
#include "../amu.h"
#include "model_generated.h"
#include "cache.h"
#include "uart.h"
#include "protocol.h"
#include "diagnostics.h"
extern const unsigned char model_weights[], model_weights_end[];
static unsigned char arena[MODEL_ARENA_BYTES] __attribute__((aligned(64)));
/* Inspect these symbols with the team's debugger when UART cannot start. */
volatile uintptr_t board_error, board_trap_cause, board_trap_pc, board_trap_value;
volatile uint32_t board_current_node;
int board_selftest(void);
static int startup_failure;
__attribute__((noreturn)) void platform_exit(uintptr_t code) {
    board_error=code;
    if(board_diag.stage!=DIAG_TRAP) board_diag_error((unsigned)code,0);
    board_clean((const void *)&board_error,sizeof(board_error));
    for(;;) __asm__ volatile("wfi");
}
__attribute__((noreturn)) void platform_trap(uintptr_t cause,uintptr_t pc,uintptr_t value,const uintptr_t *registers) {
    board_trap_cause=cause; board_trap_pc=pc; board_trap_value=value;
    board_diag_trap(cause,pc,value,registers);
    platform_exit(0x100);
}
unsigned char *board_model_input(void) { return arena+model_tensors[0].offset; }
const unsigned char *board_model_output(void) { return arena+model_tensors[MODEL_OUTPUT_ID].offset; }
uint32_t board_model_input_size(void) { return model_tensors[0].bytes; }
uint32_t board_model_output_size(void) { return model_tensors[MODEL_OUTPUT_ID].bytes; }
int board_model_run(void) {
    if(startup_failure) return -3;
    board_diag.error=board_diag.detail=0;
    board_diag.node=board_diag.tensor=UINT64_MAX;
    board_diag.op=UINT64_MAX;
    board_diag.inputs=board_diag.outputs=board_diag.node_start=board_diag.node_cycles=0;
    board_diag.element=board_diag.observed=board_diag.expected=0;
    board_diag.phase=PHASE_INPUT;
    board_diag_stage(DIAG_RUN);
    board_diag_event();
    /* Reject corrupt/nonfinite inputs even if the UART CRC was valid. */
    const unsigned char *input=board_model_input();
    for(size_t i=0;i<board_model_input_size();i+=4) {
        uint32_t bits=(uint32_t)input[i] | (uint32_t)input[i+1]<<8 |
                      (uint32_t)input[i+2]<<16 | (uint32_t)input[i+3]<<24;
        if((bits&0x7f800000)==0x7f800000) {
            board_diag.tensor=0; board_diag.element=i/4; board_diag.observed=bits;
            board_diag_error(0x30,-1); return -1;
        }
    }
    accel_rvv_nodes=accel_amu_nodes=accel_scalar_nodes=accel_alias_nodes=0;
    for(size_t step=0;step<MODEL_NODE_COUNT;++step) {
        board_current_node=(uint32_t)step;
        const ModelNode *node=model_nodes+step;
        board_diag_node((uint32_t)step,node,model_tensors);
        board_diag_phase(PHASE_CACHE);
        for(size_t i=0;i<node->inputs;++i) {
            const ModelTensor *t=model_tensors+node->in[i];
            board_clean((t->constant ? model_weights : arena)+t->offset,t->bytes);
        }
        board_diag_phase(PHASE_COMPUTE);
        int status=accel_model_node(node,model_tensors,arena,model_weights);
        if(status) { board_diag_error(0x31,status); return -2; }
        if((board_diag.options&DIAG_FINITE) || step+1==MODEL_NODE_COUNT) {
            board_diag_phase(PHASE_CHECK);
            for(unsigned i=0;i<node->outputs;++i) {
                const ModelTensor *t=model_tensors+node->out[i];
                if(t->dtype!=1) continue;
                const uint32_t *data=(const uint32_t *)(arena+t->offset);
                for(size_t j=0;j<t->count;++j) if((data[j]&0x7f800000)==0x7f800000) {
                    board_diag.tensor=(uint64_t)node->out[i]; board_diag.element=j;
                    board_diag.observed=data[j]; board_diag_error(0x32,-1); return -2;
                }
            }
        }
        board_diag.node_cycles=board_diag_ticks()-board_diag.node_start;
        board_diag.phase=PHASE_NODE_END;
        board_diag_commit();
        if(board_diag.options&DIAG_TRACE) board_diag_event();
    }
    board_diag_stage(DIAG_DONE);
    board_diag_event();
    return 0;
}
int main(void) {
    board_diag_init();
    if((size_t)(model_weights_end-model_weights)!=MODEL_CONSTANT_BYTES) platform_exit(1);
    if(model_tensors[0].dtype!=1 || board_model_input_size()!=3*416*416*4 ||
       board_model_output_size()!=7200) platform_exit(2);
    board_diag_stage(DIAG_UART);
    int status=board_uart_init();
    if(status) platform_exit((uintptr_t)(0x20-status));
    board_diag_stage(DIAG_AMU_INIT);
    status=amu_init();
    if(status) { startup_failure=1; board_error=3; board_diag_error(3,status); }
    else {
        board_diag_stage(DIAG_SELFTEST);
        status=board_selftest();
        if(status) { startup_failure=1; board_error=4; board_diag_error(4,status); }
    }
    if(!startup_failure) board_diag_stage(DIAG_READY);
    board_rpc_loop();
    return 0;
}
