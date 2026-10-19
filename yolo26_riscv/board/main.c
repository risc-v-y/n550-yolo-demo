#include "../accel_model.h"
#include "../amu.h"
#include "model_generated.h"
#include "cache.h"
#include "memory.h"
#include "uart.h"
#include "model_io.h"
#include "diagnostics.h"
#include "pcie.h"
extern const unsigned char model_weights[], model_weights_end[];
static unsigned char arena[MODEL_ARENA_BYTES] BOARD_CLP_DATA;
/* Inspect these symbols with the team's debugger when UART cannot start. */
volatile uintptr_t board_error, board_trap_cause, board_trap_pc, board_trap_value;
volatile uint32_t board_current_node;
int board_clp_selftest(void);
static int startup_failure;
_Static_assert(MODEL_NODE_COUNT==BOARD_PROFILE_NODES,"profile node count");
__attribute__((noreturn)) void platform_exit(uintptr_t code);
static void initialize_clp(void) {
    size_t bytes=(size_t)(__clp_end-__clp_start);
    if((uintptr_t)__clp_start!=BOARD_CLP_DDR_BASE || bytes>BOARD_CLP_BYTES || bytes%64)
        platform_exit(2);
    board_uart_text("[YOLO] CLP data_ddr="); board_uart_hex((uintptr_t)__clp_start);
    board_uart_text(" rvv="); board_uart_hex(BOARD_CLP_RVV_BASE);
    board_uart_text(" bytes="); board_uart_u64(bytes); board_uart_text("\n");
    /* Relinquish global cache lines before uncached initialization, then
     * invalidate their aliases before any scalar/model/PCIe access. */
    board_prepare_write(__clp_start,bytes);
    uintptr_t at=(uintptr_t)board_rvv_address(__clp_start);
    size_t words=bytes/4;
    while(words) {
        uintptr_t vl;
        __asm__ volatile("vsetvli %0,%2,e32,m1,ta,ma\n"
                         "vmv.v.i v8,0\nvse32.v v8,(%1)"
                         : "=&r"(vl) : "r"(at),"r"(words) : "v8","memory");
        at+=vl*4; words-=vl;
    }
    board_finish_write(__clp_start,bytes);
}
__attribute__((noreturn)) void platform_exit(uintptr_t code) {
    board_error=code;
    if(board_diag.stage!=DIAG_TRAP) board_diag_error((unsigned)code,0);
    board_clean((const void *)&board_error,sizeof(board_error));
    board_pcie_fault((unsigned)code,0);
    for(;;) __asm__ volatile("wfi");
}
__attribute__((noreturn)) void platform_trap(uintptr_t cause,uintptr_t pc,uintptr_t value,const uintptr_t *registers) {
    board_trap_cause=cause; board_trap_pc=pc; board_trap_value=value;
    board_diag_trap(cause,pc,value,registers);
    platform_exit(0x100);
}
unsigned char *board_model_input(void) { return arena+model_tensors[0].offset; }
const unsigned char *board_model_output(void) {
    return arena+model_tensors[MODEL_OUTPUT_ID].offset;
}
uint32_t board_model_input_size(void) { return model_tensors[0].bytes; }
uint32_t board_model_output_size(void) {
    return model_tensors[MODEL_OUTPUT_ID].bytes;
}
int board_model_run(void) {
    if(startup_failure) return -3;
    board_diag.error=board_diag.detail=0;
    board_diag.node=board_diag.tensor=UINT64_MAX;
    board_diag.op=UINT64_MAX;
    board_diag.inputs=board_diag.outputs=board_diag.node_start=board_diag.node_cycles=0;
    board_diag.element=board_diag.observed=board_diag.expected=0;
    board_diag.phase=PHASE_INPUT;
    board_profile_begin(MODEL_NODE_COUNT);
    board_diag_stage(DIAG_RUN);
    uint64_t input_start=board_diag_ticks();
    /* Reject corrupt/nonfinite inputs even if the PCIe CRC was valid. */
    const unsigned char *input=board_model_input();
    for(size_t i=0;i<board_model_input_size();i+=4) {
        uint32_t bits=(uint32_t)input[i] | (uint32_t)input[i+1]<<8 |
                      (uint32_t)input[i+2]<<16 | (uint32_t)input[i+3]<<24;
        if((bits&0x7f800000)==0x7f800000) {
            board_diag.tensor=0; board_diag.element=i/4; board_diag.observed=bits;
            board_diag_error(0x30,-1); return -1;
        }
    }
    uint64_t graph_start=board_diag_ticks();
    uint64_t input_cycles=graph_start-input_start;
    accel_rvv_nodes=accel_amu_nodes=accel_scalar_nodes=accel_alias_nodes=0;
    for(size_t step=0;step<MODEL_NODE_COUNT;++step) {
        board_current_node=(uint32_t)step;
        const ModelNode *node=model_nodes+step;
        board_diag_node((uint32_t)step,node,model_tensors);
        uint64_t cache_start=board_diag_ticks();
        board_diag_phase(PHASE_CACHE);
        for(size_t i=0;i<node->inputs;++i) {
            const ModelTensor *t=model_tensors+node->in[i];
            board_clean((t->constant ? model_weights : arena)+t->offset,t->bytes);
        }
        uint64_t cache_cycles=board_diag_ticks()-cache_start;
        board_pcie_node_begin((uint32_t)step,node,model_tensors,arena,model_weights);
        board_diag_phase(PHASE_COMPUTE);
        uint64_t compute_start=board_diag_ticks();
        int status=accel_model_node(node,model_tensors,arena,model_weights);
        uint64_t compute_cycles=board_diag_ticks()-compute_start;
        if(status) { board_diag_error(0x31,status); return -2; }
        uint64_t check_start=board_diag_ticks();
        if((board_diag.options&DIAG_FINITE) || step+1==MODEL_NODE_COUNT) {
            board_diag_phase(PHASE_CHECK);
            for(unsigned i=0;i<node->outputs;++i) {
                if(node->out[i]<0) continue;
                const ModelTensor *t=model_tensors+node->out[i];
                if(t->dtype!=1) continue;
                /* Constant views retain the weights address; their offset
                 * must not be interpreted relative to the activation arena. */
                const unsigned char *base=t->constant ? model_weights : arena;
                const uint32_t *data=(const uint32_t *)(base+t->offset);
                for(size_t j=0;j<t->count;++j) if((data[j]&0x7f800000)==0x7f800000) {
                    board_diag.tensor=(uint64_t)node->out[i]; board_diag.element=j;
                    board_diag.observed=data[j]; board_diag_error(0x32,-1); return -2;
                }
            }
        }
        uint64_t check_cycles=board_diag_ticks()-check_start;
        board_profile.node[step]=(BoardNodeProfile){cache_cycles,compute_cycles,check_cycles};
        board_profile.node_cycles+=cache_cycles+compute_cycles+check_cycles;
        board_profile.completed=step+1;
        board_diag.node_cycles=board_diag_ticks()-board_diag.node_start;
        board_diag.phase=PHASE_NODE_END;
        board_diag_commit();
        if(board_diag.options&DIAG_TRACE) board_diag_event();
        if(board_pcie_node_end()) return -2;
    }
    board_profile_end(graph_start,input_cycles);
    board_diag_stage(DIAG_DONE);
    return 0;
}
int main(void) {
    board_diag_init();
    board_pcie_init();
    board_diag_stage(DIAG_UART);
    int status=board_uart_init();
    if(!status) {
        board_uart_text("[YOLO] BOOT transport=PCIe debug=UART baud=");
        board_uart_u64(board_uart_baud);
        board_uart_text(" dlf_bits="); board_uart_u64(board_uart_dlf_bits);
        board_uart_text(" nodes="); board_uart_u64(MODEL_NODE_COUNT);
        board_uart_text(" input_bytes="); board_uart_u64(board_model_input_size());
        board_uart_text(" output_bytes="); board_uart_u64(board_model_output_size());
        board_uart_text("\n");
    }
    if((size_t)(model_weights_end-model_weights)!=MODEL_CONSTANT_BYTES) platform_exit(1);
    if(model_tensors[0].dtype!=1 || board_model_input_size()!=3*416*416*4 ||
       board_model_output_size()!=7200) platform_exit(2);
    initialize_clp();
    board_diag_stage(DIAG_AMU_INIT);
    status=amu_init();
    if(status) { startup_failure=1; board_error=3; board_diag_error(3,status); }
    else {
        board_diag_stage(DIAG_SELFTEST);
        status=board_clp_selftest();
        if(status) { startup_failure=1; board_error=4; board_diag_error(4,status); }
    }
    if(!startup_failure) {
        board_diag.frame=0; board_diag.phase=PHASE_NONE;
        board_diag_stage(DIAG_READY);
    }
    board_pcie_loop(startup_failure);
    return 0;
}
