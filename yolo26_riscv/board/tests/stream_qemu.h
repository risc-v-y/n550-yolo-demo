/* Included by the model fixture; never linked into board firmware. */
#include "../stream.h"
void *memset(void *pointer,int value,size_t bytes) {
    volatile unsigned char *p=pointer;
    while(bytes--) *p++=(unsigned char)value;
    return pointer;
}
static void stream_commit(const PcieRequest *request,unsigned slot) {
    const PcieStreamInfo *info=(const void *)(uintptr_t)board_pcie_mailbox.info.reserved2;
    PcieInputSlot *inputs=(void *)(uintptr_t)info->slots;
    volatile void *target=slot ? (void *)&inputs[slot-1].request : (void *)&board_pcie_mailbox.request;
    volatile unsigned char *dst=target;
    const unsigned char *src=(const void *)request;
    for(unsigned i=0;i<64;++i) dst[i]=src[i];
    board_clean((const void *)target,64);
    board_pcie_mailbox.doorbell[0]=request->sequence;
    board_pcie_mailbox.doorbell[1]=slot;
    board_clean((const void *)board_pcie_mailbox.doorbell,64);
    board_pcie_poll();
}
static void stream_test(void) {
    const PcieStreamInfo *info=(const void *)(uintptr_t)board_pcie_mailbox.info.reserved2;
    if(info->version!=2 || info->crc!=board_crc32(info,120) || info->max_rows<128 ||
       info->accumulator_bytes<128*128*4 || (info->reserved&31)!=31) clp_qemu_exit(40);
    PcieInputSlot *slots=(void *)(uintptr_t)info->slots;
    const PcieStreamResult *result=(const void *)(uintptr_t)info->result;
    file("frame-0-input.bin",board_model_input(),board_model_input_size(),0);
    const float *input=(const float *)board_model_input();
    for(unsigned i=0;i<STREAM_INPUT_BYTES;++i) {
        unsigned value=(unsigned)(input[i]*255.f+.5f);
        if(value>255 || (float)value/255.f!=input[i]) clp_qemu_exit(41);
        slots[0].input[i]=slots[1].input[i]=(uint8_t)value;
    }
    /* Prepare both independent slots before consuming either one. */
    board_clean(slots,2*sizeof(*slots));
    PcieRequest configure={.session=0x12345678,.command=PCIE_CONFIGURE,.sequence=2,
                           .reserved={128,128,29,0,11}};
    configure.crc=board_crc32(&configure,60);
    stream_commit(&configure,0);
    if(board_pcie_mailbox.status.state!=PCIE_READY) clp_qemu_exit(42);
    static uint8_t first[7200];
    for(unsigned pass=0;pass<2;++pass) {
        PcieRequest request={.session=0x12345678,.command=PCIE_RUN_U8,.sequence=3+pass,
            .frame=pass+1,.options=pass ? DIAG_LITE : 0,.input_bytes=STREAM_INPUT_BYTES};
        request.input_crc=board_crc32(slots[pass].input,STREAM_INPUT_BYTES);
        request.crc=board_crc32(&request,60);
        stream_commit(&request,pass+1);
        if(result->status.state!=PCIE_DONE || result->status.frame!=pass+1 ||
           result->status.sequence!=3+pass || result->status.output_bytes!=7200 ||
           result->crc!=board_crc32(result,offsetof(PcieStreamResult,crc)) ||
           result->status.output_crc!=board_crc32(result->candidates,7200) ||
           result->profiled!=(pass==0) || !result->model || !result->service ||
           board_current_node!=333) clp_qemu_exit(43);
        if(!pass) {
            for(unsigned i=0;i<7200;++i) first[i]=result->candidates[i];
            file("frame-0-candidates.bin",first,7200,1);
            file("frame-0-profile.bin",(void *)&board_profile,sizeof(board_profile),1);
        } else for(unsigned i=0;i<7200;++i) if(first[i]!=result->candidates[i]) clp_qemu_exit(44);
        board_uart_text("QEMU STREAM FRAME PASS\n");
    }
    board_uart_text("QEMU MODEL BUS PASS; UINT8/two slots/MN128/pack29/compute0/graph11/full+lite\n");
    clp_qemu_exit(0);
}
