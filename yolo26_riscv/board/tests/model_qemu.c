/* Actual board main + mailbox + model, in QEMU unified RAM.
 * Link --wrap=board_pcie_loop; physical firmware never links this fixture. */
#include "../pcie.h"
#include "../model_io.h"
#include "../diagnostics.h"
#include "../cache.h"
#include "../memory.h"
#include "../uart.h"
#include "../../scalar_model.h"
#include "model_generated.h"
extern const unsigned char model_weights[];
extern void clp_qemu_exit(uintptr_t code) __attribute__((noreturn));
extern volatile uint32_t board_current_node,board_test_stage;
static uintptr_t host_call(uintptr_t op,uintptr_t argument) {
    register uintptr_t a0 __asm__("a0")=op;
    register uintptr_t a1 __asm__("a1")=argument;
    __asm__ volatile(".option push\n.option norvc\nslli zero,zero,31\n"
                     "ebreak\nsrai zero,zero,7\n.option pop"
                     : "+r"(a0),"+r"(a1) :: "memory");
    return a0;
}
static void file(const char *name,void *data,size_t bytes,int writing) {
    size_t length=0;
    while(name[length]) ++length;
    uintptr_t open_args[3]={(uintptr_t)name,writing ? 5 : 1,length};
    uintptr_t handle=host_call(1,(uintptr_t)open_args);
    if(handle==(uintptr_t)-1) clp_qemu_exit(10);
    uintptr_t transfer[3]={handle,(uintptr_t)data,bytes};
    if(host_call(writing ? 5 : 6,(uintptr_t)transfer)) clp_qemu_exit(11);
    if(host_call(2,(uintptr_t)&handle)) clp_qemu_exit(12);
}
static void submit(unsigned command,unsigned sequence,unsigned frame) {
    PcieRequest request={.session=0x12345678,.frame=frame,.command=command,
                        .sequence=sequence,.options=DIAG_FINITE};
    if(command==PCIE_RUN) {
        request.input_bytes=board_model_input_size();
        request.input_crc=board_crc32(board_model_input(),request.input_bytes);
    }
    request.crc=board_crc32(&request,60);
    const unsigned char *src=(const unsigned char *)&request;
    volatile unsigned char *dst=(volatile unsigned char *)&board_pcie_mailbox.request;
    for(unsigned i=0;i<sizeof(request);++i) dst[i]=src[i];
    board_clean((const void *)&board_pcie_mailbox.request,sizeof(request));
    board_pcie_mailbox.doorbell[0]=sequence;
    board_clean((const void *)board_pcie_mailbox.doorbell,64);
    board_pcie_poll();
}
static uintptr_t poison_constant_view_alias(void) {
    const ModelNode *node=model_nodes+305;
    const ModelTensor *input=model_tensors+515,*output=model_tensors+516;
    uintptr_t arena=(uintptr_t)board_model_input()-model_tensors[0].offset;
    uintptr_t wrong=arena+output->offset;
    if(node->op!=OP_COPY || node->in[0]!=515 || node->out[0]!=516 ||
       !input->constant || !output->constant || input->offset!=output->offset ||
       input->bytes!=output->bytes || output->dtype!=1 ||
       output->bytes!=output->count*4 || wrong<(uintptr_t)__clp_end ||
       wrong>BOARD_CLP_DDR_BASE+BOARD_CLP_BYTES-output->bytes ||
       *(const uint32_t *)(model_weights+output->offset)!=0x3f000000u) clp_qemu_exit(23);
    /* Emulate the board's uninitialized DDR at the WRONG arena-relative
     * address. The constant view lives in model_weights, not this region.
     * Leaving QEMU's RAM zero here allowed the old checker to pass. */
    volatile uint32_t *guard=(volatile uint32_t *)wrong;
    for(size_t i=0;i<output->count;++i) guard[i]=0xffffffffu;
    board_clean((const void *)guard,output->bytes);
    board_uart_text("QEMU CONSTANT VIEW GUARD tensor=516 wrong-region=NaN\n");
    return wrong;
}
void __wrap_board_pcie_loop(int failure) {
    board_pcie_start(failure);
    if(failure || board_test_stage!=3 || board_uart_status) clp_qemu_exit(20);
    if(!board_is_clp(board_model_input()) || !board_is_clp(board_model_output())) clp_qemu_exit(21);
    uintptr_t guard=poison_constant_view_alias();
    submit(PCIE_OPEN,1,0);
    static const char *const inputs[]={"frame-0-input.bin","frame-1-input.bin"};
    static const char *const outputs[]={"frame-0-candidates.bin","frame-1-candidates.bin"};
    for(unsigned sample=0;sample<2;++sample) {
        file(inputs[sample],board_model_input(),board_model_input_size(),0);
        board_clean(board_model_input(),board_model_input_size());
        submit(PCIE_RUN,sample+2,sample+1);
        PcieStatus status=board_pcie_mailbox.status;
        if(status.state!=PCIE_DONE || status.error || status.frame!=sample+1 ||
           status.sequence!=sample+2 || status.output_bytes!=7200 || board_current_node!=333 ||
           status.crc!=board_crc32(&status,60) ||
           status.output_crc!=board_crc32(board_model_output(),7200)) clp_qemu_exit(22);
        const volatile uint32_t *poison=(const volatile uint32_t *)guard;
        for(size_t i=0;i<model_tensors[516].count;++i)
            if(poison[i]!=0xffffffffu) clp_qemu_exit(24);
        file(outputs[sample],(void *)board_model_output(),7200,1);
        board_uart_text("QEMU MODEL FRAME PASS sample="); board_uart_u64(sample);
        board_uart_text(" nodes=334\n");
    }
    board_uart_text("QEMU MODEL TWO FRAMES PASS; unified RAM, physical cache behaviour unverified\n");
    clp_qemu_exit(0);
}
