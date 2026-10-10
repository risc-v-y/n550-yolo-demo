#include "pcie.h"
#include "model_io.h"
#include "diagnostics.h"
#ifdef YOLO_N550_BOARD
#include "cache.h"
#include "uart.h"
#else
/* Native protocol regression only; physical cache behaviour is not emulated. */
#define board_clean(p,n) ((void)(p),(void)(n))
#define board_prepare_write(p,n) ((void)(p),(void)(n))
#define board_finish_write(p,n) ((void)(p),(void)(n))
#endif
volatile PcieMailbox board_pcie_mailbox __attribute__((section(".pcie_mailbox"),aligned(64)));
static PcieStatus status;
static uint32_t last_sequence;
static int startup_failed;
static void copy_to(volatile void *dst,const void *src,unsigned bytes) {
    volatile unsigned char *d=dst;
    const unsigned char *s=src;
    while(bytes--) *d++=*s++;
}
static void publish(unsigned state,unsigned error,int detail) {
    status.state=state; status.error=error; status.detail=detail;
    ++status.generation;
    status.crc=board_crc32(&status,60);
    copy_to(&board_pcie_mailbox.status,&status,sizeof(status));
    board_clean((const void *)&board_pcie_mailbox.status,sizeof(status));
#ifdef YOLO_N550_BOARD
    if(state==PCIE_READY || state==PCIE_DONE || state==PCIE_ERROR) {
        board_uart_text("[YOLO] PCIE state="); board_uart_u64(state);
        board_uart_text(" session="); board_uart_u64(status.session);
        board_uart_text(" sequence="); board_uart_u64(status.sequence);
        board_uart_text(" frame="); board_uart_u64(status.frame);
        board_uart_text(" output_bytes="); board_uart_u64(status.output_bytes);
        board_uart_text(" error="); board_uart_hex(error);
        board_uart_text(" detail="); board_uart_i64(detail);
        board_uart_text("\n");
    }
#endif
}
void board_pcie_init(void) {
    volatile unsigned char *p=(volatile unsigned char *)&board_pcie_mailbox;
    for(unsigned i=0;i<sizeof(board_pcie_mailbox);++i) p[i]=0;
    board_prepare_write((const void *)&board_pcie_mailbox,sizeof(board_pcie_mailbox));
    PcieInfo info={.magic=PCIE_MAGIC,.version=1,.bytes=256,
        .input_address=(uintptr_t)board_model_input(),.output_address=(uintptr_t)board_model_output(),
        .diagnostic_address=(uintptr_t)&board_diag,.input_bytes=board_model_input_size(),
        .output_bytes=board_model_output_size(),.diagnostic_bytes=sizeof(board_diag),.nodes=334};
    info.crc=board_crc32(&info,60);
    copy_to(&board_pcie_mailbox.info,&info,sizeof(info));
    board_clean((const void *)&board_pcie_mailbox.info,sizeof(info));
    status=(PcieStatus){.magic=PCIE_MAGIC,.version=1};
    last_sequence=0; startup_failed=0;
    publish(PCIE_BOOT,0,0);
}
void board_pcie_fault(unsigned error,int detail) {
    publish(PCIE_ERROR,error,detail);
}
void board_pcie_start(int failure) {
    startup_failed=failure;
    /* Hand host ownership of the entire input range, including dirty BSS. */
    board_prepare_write(board_model_input(),board_model_input_size());
    publish(failure ? PCIE_ERROR : PCIE_READY,failure ? (unsigned)board_diag.error : 0,
            failure ? (int)board_diag.detail : 0);
}
void board_pcie_poll(void) {
    board_finish_write((const void *)board_pcie_mailbox.doorbell,64);
    uint32_t sequence=board_pcie_mailbox.doorbell[0];
    uint32_t distance=sequence-last_sequence;
    if(!sequence || !distance || distance>=0x80000000u || startup_failed) return;
    board_finish_write((const void *)&board_pcie_mailbox.request,64);
    PcieRequest request;
    unsigned char *dst=(unsigned char *)&request;
    const volatile unsigned char *src=(const volatile unsigned char *)&board_pcie_mailbox.request;
    for(unsigned i=0;i<sizeof(request);++i) dst[i]=src[i];
    last_sequence=sequence; /* Never repeat a command, including a failed one. */
    status.sequence=sequence; status.output_bytes=status.output_crc=0;
    if(request.sequence!=sequence || request.crc!=board_crc32(&request,60) ||
       !request.session || (request.options&~(DIAG_TRACE|DIAG_FINITE))) {
        publish(PCIE_ERROR,PCIE_BAD_REQUEST,0); return;
    }
    if(request.command==PCIE_OPEN) {
        status.session=request.session; status.frame=request.frame;
        board_diag.options=request.options;
        board_diag.frame=request.frame;
        board_diag.error=board_diag.detail=0;
        board_diag_stage(DIAG_READY);
        board_prepare_write(board_model_input(),board_model_input_size());
        publish(PCIE_READY,0,0); return;
    }
    if(request.session!=status.session) { publish(PCIE_ERROR,PCIE_BAD_SESSION,0); return; }
    if(request.command!=PCIE_RUN || request.input_bytes!=board_model_input_size() || request.frame<=status.frame) {
        publish(PCIE_ERROR,PCIE_BAD_REQUEST,0); return;
    }
    status.frame=request.frame;
    publish(PCIE_BUSY,0,0);
    board_finish_write(board_model_input(),board_model_input_size());
    uint32_t crc=board_crc32(board_model_input(),board_model_input_size());
    if(crc!=request.input_crc) {
        board_diag.frame=request.frame; board_diag.observed=crc; board_diag.expected=request.input_crc;
        board_diag_error(PCIE_BAD_INPUT,0);
        board_prepare_write(board_model_input(),board_model_input_size());
        publish(PCIE_ERROR,PCIE_BAD_INPUT,0); return;
    }
    board_diag.frame=request.frame; board_diag.options=request.options;
    int result=board_model_run();
    if(!result) {
        status.output_bytes=board_model_output_size();
        status.output_crc=board_crc32(board_model_output(),status.output_bytes);
        board_clean(board_model_output(),status.output_bytes);
    }
    /* Model arena aliases its tensors. Relinquish input lines again before
     * DONE; host may reuse them only AFTER it has downloaded the old output. */
    board_prepare_write(board_model_input(),board_model_input_size());
    board_diag_commit();
    publish(result ? PCIE_ERROR : PCIE_DONE,result ? PCIE_MODEL_ERROR : 0,result);
}
void board_pcie_loop(int failure) {
    board_pcie_start(failure);
    for(;;) board_pcie_poll();
}
