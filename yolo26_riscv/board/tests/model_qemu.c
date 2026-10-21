/* Actual board main + mailbox + model, in QEMU unified RAM.
 * Link --wrap=board_pcie_loop; physical firmware never links this fixture. */
#include "../pcie.h"
#include "../model_io.h"
#include "../diagnostics.h"
#include "../cache.h"
#include "../memory.h"
#include "../uart.h"
#include "../../scalar_model.h"
#include "../../accel_model.h"
#include "model_generated.h"
extern const unsigned char model_weights[];
extern uint64_t amu_rvv_input_blocks,amu_rvv_cached_blocks,amu_rvv_row_values;
extern uint64_t rvv_depthwise3x3_calls;
extern void clp_qemu_exit(uintptr_t code) __attribute__((noreturn));
extern volatile uint32_t board_current_node,board_test_stage;
#ifdef MODEL_CAPTURE_TEST
static unsigned capture_sequence=1,capture_frame,capture_records;
static uintptr_t capture_file;
extern int __real_board_pcie_node_end(void);
#endif
static uintptr_t host_call(uintptr_t op,uintptr_t argument) {
    register uintptr_t a0 __asm__("a0")=op;
    register uintptr_t a1 __asm__("a1")=argument;
    __asm__ volatile(".option push\n.option norvc\n.balign 16\nslli zero,zero,31\n"
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
#ifdef MODEL_CAPTURE_TEST
static void capture_write(const void *data,size_t bytes) {
    uintptr_t transfer[3]={capture_file,(uintptr_t)data,bytes};
    if(host_call(5,(uintptr_t)transfer)) clp_qemu_exit(30);
}
static void capture_record(unsigned id,unsigned dtype,const void *data,unsigned bytes) {
    uint32_t header[3]={id,dtype,bytes};
    capture_write(header,sizeof(header)); capture_write(data,bytes); ++capture_records;
}
int __wrap_board_pcie_node_end(void) {
    if(!(board_diag.options&PCIE_CAPTURE)) return __real_board_pcie_node_end();
    /* Synthetic host ACK is precommitted in unified RAM. The separate native
     * host integration verifies real waiting, stable reads, ACK and ABORT. */
    PcieRequest request={.session=0x12345678,.frame=capture_frame,.command=PCIE_ADVANCE,
                        .sequence=++capture_sequence,.input_crc=board_node_snapshot.token,
                        .options=DIAG_FINITE|PCIE_CAPTURE};
#ifdef MODEL_PACK_PROBE_TEST
    request.options|=DIAG_TRACE;
#endif
    request.crc=board_crc32(&request,60);
    const unsigned char *src=(const unsigned char *)&request;
    volatile unsigned char *dst=(volatile unsigned char *)&board_pcie_mailbox.request;
    for(unsigned i=0;i<sizeof(request);++i) dst[i]=src[i];
    board_clean((const void *)&board_pcie_mailbox.request,64);
    board_pcie_mailbox.doorbell[0]=capture_sequence;
    board_clean((const void *)board_pcie_mailbox.doorbell,64);
    int result=__real_board_pcie_node_end();
    if(result || board_node_snapshot.node!=board_current_node ||
       board_node_snapshot.frame!=capture_frame ||
       board_node_snapshot.crc!=board_crc32((const void *)&board_node_snapshot,508) ||
       board_pcie_mailbox.status.state!=PCIE_BUSY ||
       board_pcie_mailbox.status.sequence!=capture_sequence) clp_qemu_exit(31);
    for(unsigned slot=4;slot<6;++slot) {
        const volatile PcieTensorSnapshot *d=board_node_snapshot.tensors+slot;
        if(d->id<0) continue;
        const void *data=(const void *)(uintptr_t)d->address;
        if(d->crc_after!=board_crc32(data,d->bytes)) clp_qemu_exit(32);
        capture_record((unsigned)d->id,d->dtype,data,d->bytes);
    }
    return 0;
}
#endif
static void submit(unsigned command,unsigned sequence,unsigned frame) {
    PcieRequest request={.session=0x12345678,.frame=frame,.command=command,
                        .sequence=sequence,.options=DIAG_FINITE};
#ifdef MODEL_CAPTURE_TEST
    request.options|=PCIE_CAPTURE;
#endif
#ifdef MODEL_PACK_PROBE_TEST
    request.options|=DIAG_TRACE;
#endif
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
    if(accel_weight_bytes!=5033984 || accel_weight_values!=2391664 || accel_weight_layers!=94)
        clp_qemu_exit(25);
    if(!board_is_clp(board_model_input()) || !board_is_clp(board_model_output())) clp_qemu_exit(21);
    uintptr_t guard=poison_constant_view_alias();
    submit(PCIE_OPEN,1,0);
    static const char *const inputs[]={"frame-0-input.bin","frame-1-input.bin"};
    static const char *const outputs[]={"frame-0-candidates.bin","frame-1-candidates.bin"};
    for(unsigned sample=0;sample<1;++sample) {
        file(inputs[sample],board_model_input(),board_model_input_size(),0);
        board_clean(board_model_input(),board_model_input_size());
#ifdef MODEL_CAPTURE_TEST
        static const char *const dumps[]={"frame-0-tensors.bin","frame-1-tensors.bin"};
        size_t length=0; while(dumps[sample][length]) ++length;
        uintptr_t open_args[3]={(uintptr_t)dumps[sample],5,length};
        capture_file=host_call(1,(uintptr_t)open_args);
        if(capture_file==(uintptr_t)-1) clp_qemu_exit(33);
        struct { char magic[8]; uint32_t version,count; } header={"Y26SCAL1",1,336};
        capture_write(&header,sizeof(header)); capture_records=0;
        capture_record(0,1,board_model_input(),board_model_input_size());
        capture_frame=sample+1;
        submit(PCIE_RUN,++capture_sequence,capture_frame);
        if(capture_records!=336 || host_call(2,(uintptr_t)&capture_file)) clp_qemu_exit(34);
        unsigned expected_sequence=capture_sequence;
#else
        submit(PCIE_RUN,sample+2,sample+1);
        unsigned expected_sequence=sample+2;
#endif
        PcieStatus status=board_pcie_mailbox.status;
        if(status.state!=PCIE_DONE || status.error || status.frame!=sample+1 ||
           status.sequence!=expected_sequence || status.output_bytes!=7200 || board_current_node!=333 ||
           status.crc!=board_crc32(&status,60) ||
           status.output_crc!=board_crc32(board_model_output(),7200)) clp_qemu_exit(22);
        if(board_profile.version!=7 || board_profile.bytes!=24320 ||
           board_profile.runtime_weight_conversions ||
           board_profile.phases[PROFILE_WEIGHT_PREPARE].ticks ||
           board_profile.phases[PROFILE_WEIGHT_PREPARE].calls ||
           board_profile.crc!=board_crc32((const void *)&board_profile,offsetof(BoardProfile,crc)))
            clp_qemu_exit(26);
        uint64_t input_blocks=0,depthwise_nodes=0,cached_blocks=0,row_values=0;
        for(unsigned i=0;i<MODEL_NODE_COUNT;++i) {
            const ModelNode *node=model_nodes+i;
            if(node->op!=OP_CONV) continue;
            if(node->p[4]!=1) {
                ++depthwise_nodes;
                if(board_profile.node[i].phase_cycles[PROFILE_INPUT_IM2COL] ||
                   board_profile.node[i].phase_cycles[PROFILE_BIAS_DEQUANT]) clp_qemu_exit(35);
                continue;
            }
            const ModelTensor *out=model_tensors+node->out[0];
            size_t area=out->shape[2]*out->shape[3];
            input_blocks+=out->shape[0]*(area/32+(area%32!=0));
            const ModelTensor *weight=model_tensors+node->in[1];
            if(weight->shape[2]==3 && weight->shape[3]==3) {
                cached_blocks+=out->shape[0]*(area/32+(area%32!=0));
                row_values+=model_tensors[node->in[0]].count;
            }
            if(board_profile.node[i].phase_cycles[PROFILE_INPUT_IM2COL] ||
               !board_profile.node[i].phase_cycles[PROFILE_INPUT_PACK]) clp_qemu_exit(28);
        }
        if(board_profile.phases[PROFILE_INPUT_PACK].calls!=input_blocks ||
           amu_rvv_input_blocks!=input_blocks) clp_qemu_exit(29);
        if(cached_blocks!=3543 || amu_rvv_cached_blocks!=cached_blocks ||
           row_values!=3853200 || amu_rvv_row_values!=row_values ||
           board_profile.phases[PROFILE_BIAS_DEQUANT].calls!=94) clp_qemu_exit(37);
        if(depthwise_nodes!=8 || rvv_depthwise3x3_calls!=depthwise_nodes ||
           board_profile.phases[PROFILE_INPUT_IM2COL].calls ||
           board_profile.phases[PROFILE_INPUT_IM2COL].ticks) clp_qemu_exit(36);
        for(unsigned phase=0;phase<PROFILE_PHASE_COUNT;++phase) {
            uint64_t sum=0;
            for(unsigned i=0;i<334;++i) sum+=board_profile.node[i].phase_cycles[phase];
            if(sum!=board_profile.phases[phase].ticks) clp_qemu_exit(27);
        }
        const volatile uint32_t *poison=(const volatile uint32_t *)guard;
        for(size_t i=0;i<model_tensors[516].count;++i)
            if(poison[i]!=0xffffffffu) clp_qemu_exit(24);
        file(outputs[sample],(void *)board_model_output(),7200,1);
        file("frame-0-profile.bin",(void *)&board_profile,sizeof(board_profile),1);
#ifdef MODEL_PACK_PROBE_TEST
        if(board_pack_probe.stage!=PACK_BLOCK_DONE || board_pack_probe.error ||
           board_pack_probe.column!=32 || board_pack_probe.next_column!=32 ||
           board_pack_probe.converted!=2496 || !(board_profile.options&DIAG_TRACE)) clp_qemu_exit(38);
        file("pack-probe.bin",(void *)&board_pack_probe,sizeof(board_pack_probe),1);
#endif
        board_uart_text("QEMU MODEL FRAME PASS sample="); board_uart_u64(sample);
        board_uart_text(" nodes=334\n");
    }
    board_uart_text("QEMU MODEL BUS PASS; cached rows/whole-layer bias, exact counters; physical cache unverified\n");
    clp_qemu_exit(0);
}
