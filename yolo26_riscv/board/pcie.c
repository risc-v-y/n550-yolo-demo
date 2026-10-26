#include "pcie.h"
#include "model_io.h"
#include "diagnostics.h"
#ifdef YOLO_N550_BOARD
#include "cache.h"
#include "uart.h"
#include "stream.h"
#include "memory.h"
#include "../amu.h"
static PcieInputSlot input_slots[2] BOARD_CLP_DATA;
static PcieStreamInfo stream_info __attribute__((aligned(64)));
static PcieStreamResult stream_result __attribute__((aligned(64)));
static int stream_active;
static uint64_t stream_configuration;
static void stream_capabilities(void) {
    uintptr_t acc,vector;
    __asm__ volatile("csrr %0,0xcc3" : "=r"(acc));
    __asm__ volatile("csrr %0,vlenb" : "=r"(vector));
    stream_info=(PcieStreamInfo){.magic=PCIE_MAGIC,.version=2,.bytes=sizeof(stream_info),
        .slots=(uintptr_t)input_slots,.result=(uintptr_t)&stream_result,.profile=(uintptr_t)&board_profile,
        .snapshot=(uintptr_t)&board_node_snapshot,.input_bytes=STREAM_INPUT_BYTES,
        .slot_bytes=sizeof(PcieInputSlot),.result_bytes=sizeof(stream_result),.max_rows=amu_capacity_rows,
        .accumulator_bytes=acc,.vector_bytes=vector,.option_mask=YOLO_FAST ? DIAG_LITE : 30,.reserved=AMU_INPUT_MASK};
    stream_info.crc=board_crc32(&stream_info,offsetof(PcieStreamInfo,crc));
    board_clean(&stream_info,sizeof(stream_info));
}
#else
/* Native protocol regression only; physical cache behaviour is not emulated. */
#define board_clean(p,n) ((void)(p),(void)(n))
#define board_prepare_write(p,n) ((void)(p),(void)(n))
#define board_finish_write(p,n) ((void)(p),(void)(n))
#endif
volatile PcieMailbox board_pcie_mailbox __attribute__((section(".pcie_mailbox"),aligned(64)));
#ifdef BOARD_SNAPSHOT_HOST_TEST
volatile PcieNodeSnapshot board_node_snapshot __attribute__((section(".pcie_snapshot"),aligned(64)));
#else
volatile PcieNodeSnapshot board_node_snapshot __attribute__((aligned(64)));
#endif
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
#ifdef YOLO_N550_BOARD
    {
        /* Payload is visible before the final CRC-protected status line.
         * Fast firmware omits data CRC, never control/header CRC. */
        PcieStatus previous=stream_result.status;
        stream_result.status=status;
        stream_result.crc=YOLO_FAST ? 0 : board_crc32(&stream_result,offsetof(PcieStreamResult,crc));
        stream_result.status=previous;
        board_clean((char *)&stream_result+64,sizeof(stream_result)-64);
        copy_to(&stream_result.status,&status,64);
        board_clean(&stream_result.status,64);
    }
#endif
    copy_to(&board_pcie_mailbox.status,&status,sizeof(status));
    board_clean((const void *)&board_pcie_mailbox.status,sizeof(status));
#ifdef YOLO_N550_BOARD
    if(state==PCIE_ERROR || (!YOLO_FAST && (state==PCIE_READY || state==PCIE_DONE))) {
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
    PcieInfo info={.magic=PCIE_MAGIC,.version=1,.bytes=256,.reserved=PCIE_HAS_CAPTURE,
        .input_address=(uintptr_t)board_model_input(),.output_address=(uintptr_t)board_model_output(),
        .diagnostic_address=(uintptr_t)&board_diag,.input_bytes=board_model_input_size(),
        .output_bytes=board_model_output_size(),.diagnostic_bytes=sizeof(board_diag),.nodes=334,
        .reserved2=(uint32_t)(uintptr_t)&board_node_snapshot};
    info.crc=board_crc32(&info,60);
#ifdef YOLO_N550_BOARD
    info.version=2;
    info.reserved=PCIE_HAS_STREAM|PCIE_RESULT_POLL|(YOLO_FAST ? PCIE_FAST_IMAGE : PCIE_HAS_CAPTURE);
    info.reserved2=(uint32_t)(uintptr_t)&stream_info;
    info.crc=board_crc32(&info,60);
#endif
    copy_to(&board_pcie_mailbox.info,&info,sizeof(info));
    board_clean((const void *)&board_pcie_mailbox.info,sizeof(info));
    status=(PcieStatus){.magic=PCIE_MAGIC,.version=1};
#ifdef YOLO_N550_BOARD
    status.version=2;
#endif
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
#ifdef YOLO_N550_BOARD
    if(!failure) stream_capabilities();
    board_prepare_write(input_slots,sizeof(input_slots));
#endif
    publish(failure ? PCIE_ERROR : PCIE_READY,failure ? (unsigned)board_diag.error : 0,
            failure ? (int)board_diag.detail : 0);
}
static void read_request(PcieRequest *request) {
    board_finish_write((const void *)&board_pcie_mailbox.request,64);
    unsigned char *dst=(unsigned char *)request;
    const volatile unsigned char *src=(const volatile unsigned char *)&board_pcie_mailbox.request;
    for(unsigned i=0;i<sizeof(*request);++i) dst[i]=src[i];
}
static int valid_request(const PcieRequest *request,uint32_t sequence) {
    return request->sequence==sequence && request->crc==board_crc32(request,60) &&
           request->session && !(request->options&~(DIAG_TRACE|DIAG_FINITE|PCIE_CAPTURE|DIAG_LITE)) &&
           (!YOLO_FAST || request->options==DIAG_LITE);
}
void board_pcie_node_begin(uint32_t id,const ModelNode *node,const ModelTensor *tensors,
                           unsigned char *arena,const unsigned char *constants) {
    if(!(board_diag.options&PCIE_CAPTURE)) return;
    volatile unsigned char *p=(volatile unsigned char *)&board_node_snapshot;
    for(unsigned i=0;i<sizeof(board_node_snapshot);++i) p[i]=0;
    board_node_snapshot.magic=PCIE_MAGIC; board_node_snapshot.version=1;
    board_node_snapshot.bytes=sizeof(board_node_snapshot); board_node_snapshot.node=id;
    board_node_snapshot.session=status.session; board_node_snapshot.frame=status.frame;
    board_node_snapshot.op=node->op; board_node_snapshot.inputs=node->inputs;
    board_node_snapshot.outputs=node->outputs; board_node_snapshot.token=id+1;
    board_node_snapshot.options=(uint32_t)board_diag.options;
    for(unsigned slot=0;slot<6;++slot) {
        volatile PcieTensorSnapshot *d=board_node_snapshot.tensors+slot;
        d->id=-1;
        if(slot<4 ? slot>=node->inputs : slot-4>=node->outputs) continue;
        int index=slot<4 ? node->in[slot] : node->out[slot-4];
        if(index<0) continue;
        const ModelTensor *t=tensors+index;
        d->id=index; d->dtype=t->dtype; d->rank=t->rank; d->bytes=t->bytes;
        d->address=(uintptr_t)((t->constant ? constants : arena)+t->offset);
        d->constant=t->constant; d->count=t->count;
        for(unsigned i=0;i<4;++i) d->shape[i]=t->shape[i];
        if(slot<4) {
            board_clean((const void *)(uintptr_t)d->address,d->bytes);
            d->crc_before=board_crc32((const void *)(uintptr_t)d->address,d->bytes);
        }
    }
}
int board_pcie_node_end(void) {
    if(!(board_diag.options&PCIE_CAPTURE)) return 0;
    for(unsigned slot=0;slot<6;++slot) {
        volatile PcieTensorSnapshot *d=board_node_snapshot.tensors+slot;
        if(d->id<0) continue;
        board_clean((const void *)(uintptr_t)d->address,d->bytes);
        d->crc_after=board_crc32((const void *)(uintptr_t)d->address,d->bytes);
        if(slot>=4) d->crc_before=d->crc_after;
    }
    board_node_snapshot.sequence=last_sequence;
    board_node_snapshot.tail[0]=(uint32_t)board_diag.amu_flags;
    board_node_snapshot.tail[1]=(uint32_t)(board_diag.amu_flags>>32);
    board_node_snapshot.tail[2]=(uint32_t)board_diag.node_cycles;
    board_node_snapshot.tail[3]=(uint32_t)(board_diag.node_cycles>>32);
    board_node_snapshot.tail[4]=(uint32_t)board_diag.phase;
    board_node_snapshot.crc=board_crc32((const void *)&board_node_snapshot,508);
    board_clean((const void *)&board_node_snapshot,sizeof(board_node_snapshot));
    status.output_bytes=sizeof(board_node_snapshot);
    status.output_crc=board_crc32((const void *)&board_node_snapshot,sizeof(board_node_snapshot));
    publish(PCIE_NODE,0,(int)board_node_snapshot.node);
    /* The model arena remains frozen until an authenticated ACK/ABORT. No
     * nested model call, input upload or new session is allowed here. */
    for(;;) {
        board_finish_write((const void *)board_pcie_mailbox.doorbell,64);
        uint32_t sequence=board_pcie_mailbox.doorbell[0],distance=sequence-last_sequence;
        if(!sequence || !distance || distance>=0x80000000u) continue;
        PcieRequest request; read_request(&request);
        last_sequence=sequence; status.sequence=sequence;
        status.output_bytes=status.output_crc=0;
        if(!valid_request(&request,sequence) || request.input_bytes ||
           request.options!=board_diag.options || request.input_crc!=board_node_snapshot.token) {
            board_diag_error(PCIE_BAD_REQUEST,(int)board_node_snapshot.node); return -2;
        }
        if(request.session!=status.session || request.frame!=status.frame) {
            board_diag_error(PCIE_BAD_SESSION,(int)board_node_snapshot.node); return -2;
        }
        if(request.command==PCIE_ABORT) {
            board_diag_error(PCIE_ABORTED,(int)board_node_snapshot.node); return -2;
        }
        if(request.command!=PCIE_ADVANCE) {
            board_diag_error(PCIE_BAD_REQUEST,(int)board_node_snapshot.node); return -2;
        }
        publish(PCIE_BUSY,0,0); return 0;
    }
}
void board_pcie_poll(void) {
    board_finish_write((const void *)board_pcie_mailbox.doorbell,64);
    uint32_t sequence=board_pcie_mailbox.doorbell[0];
    uint32_t distance=sequence-last_sequence;
    if(!sequence || !distance || distance>=0x80000000u || startup_failed) return;
    PcieRequest request;
    read_request(&request);
#ifdef YOLO_N550_BOARD
    unsigned slot=board_pcie_mailbox.doorbell[1];
    stream_active=0;
    if(slot) {
        if(slot>2) {last_sequence=sequence; status.sequence=sequence; publish(PCIE_ERROR,PCIE_BAD_REQUEST,0); return;}
        board_finish_write(input_slots+slot-1,sizeof(PcieInputSlot));
        const unsigned char *from=(const unsigned char *)&input_slots[slot-1].request;
        unsigned char *to=(unsigned char *)&request;
        for(unsigned i=0;i<64;++i) to[i]=from[i];
        stream_active=1;
    }
#endif
    last_sequence=sequence; /* Never repeat a command, including a failed one. */
    status.sequence=sequence; status.output_bytes=status.output_crc=0;
    if(!valid_request(&request,sequence)) {
        publish(PCIE_ERROR,PCIE_BAD_REQUEST,0); return;
    }
    if(request.command==PCIE_OPEN) {
#ifdef YOLO_N550_BOARD
        if(slot || board_model_configure(32,32,0,0,0)) {publish(PCIE_ERROR,PCIE_MODEL_ERROR,-1); return;}
        board_model_set_u8(0); stream_configuration=0;
#endif
        status.session=request.session; status.frame=request.frame;
        board_diag.options=request.options;
        board_diag.frame=request.frame;
        board_diag.error=board_diag.detail=0;
        board_diag_stage(DIAG_READY);
        board_prepare_write(board_model_input(),board_model_input_size());
        publish(PCIE_READY,0,0); return;
    }
    if(request.session!=status.session) { publish(PCIE_ERROR,PCIE_BAD_SESSION,0); return; }
#ifdef YOLO_N550_BOARD
    if(request.command==PCIE_CONFIGURE && !slot && !request.input_bytes) {
        if(board_model_configure(request.reserved[0],request.reserved[1],request.reserved[2],
                                  request.reserved[3],request.reserved[4])) {
            startup_failed=1; publish(PCIE_ERROR,PCIE_MODEL_ERROR,-6); return;
        }
        stream_configuration=request.reserved[0] | (uint64_t)request.reserved[1]<<8 |
            (uint64_t)request.reserved[2]<<16 | (uint64_t)request.reserved[3]<<24 |
            (uint64_t)request.reserved[4]<<32;
        publish(PCIE_READY,0,0); return;
    }
    int compact=request.command==PCIE_RUN_U8;
    if(YOLO_FAST && !compact) {publish(PCIE_ERROR,PCIE_BAD_REQUEST,0); return;}
    if(compact && (!slot || request.input_bytes!=STREAM_INPUT_BYTES || (request.options&PCIE_CAPTURE))) {
        publish(PCIE_ERROR,PCIE_BAD_REQUEST,0); return;
    }
    if(!compact && slot) {publish(PCIE_ERROR,PCIE_BAD_REQUEST,0); return;}
#else
    int compact=0;
#endif
    if((!compact && (request.command!=PCIE_RUN || request.input_bytes!=board_model_input_size())) || request.frame<=status.frame) {
        publish(PCIE_ERROR,PCIE_BAD_REQUEST,0); return;
    }
    status.frame=request.frame;
    uint64_t service_start=YOLO_FAST ? 0 : board_diag_ticks();
    publish(PCIE_BUSY,0,0);
    uint64_t crc_start=YOLO_FAST ? 0 : board_diag_ticks();
    const void *input_data=board_model_input();
    size_t input_bytes=board_model_input_size();
#ifdef YOLO_N550_BOARD
    if(compact) {input_data=input_slots[slot-1].input; input_bytes=STREAM_INPUT_BYTES;}
    board_model_set_u8(compact ? input_data : 0);
#endif
    board_finish_write(input_data,input_bytes);
    uint32_t crc=YOLO_FAST ? 0 : board_crc32(input_data,input_bytes);
    uint64_t crc_cycles=YOLO_FAST ? 0 : board_diag_ticks()-crc_start;
    if(!YOLO_FAST && crc!=request.input_crc) {
        board_diag.frame=request.frame; board_diag.observed=crc; board_diag.expected=request.input_crc;
        board_diag_error(PCIE_BAD_INPUT,0);
        board_prepare_write(board_model_input(),board_model_input_size());
        publish(PCIE_ERROR,PCIE_BAD_INPUT,0); return;
    }
    board_diag.frame=request.frame; board_diag.options=request.options;
    int result=board_model_run();
    uint64_t output_start=YOLO_FAST ? 0 : board_diag_ticks();
    if(!result) {
        status.output_bytes=board_model_output_size();
        status.output_crc=YOLO_FAST ? 0 : board_crc32(board_model_output(),status.output_bytes);
        board_clean(board_model_output(),status.output_bytes);
    }
    /* Model arena aliases its tensors. Relinquish input lines again before
     * DONE; host may reuse them only AFTER it has downloaded the old output. */
    board_prepare_write(board_model_input(),board_model_input_size());
    if(!YOLO_FAST || result) board_diag_commit();
    if(!YOLO_FAST && !result && board_profile.magic==BOARD_PROFILE_MAGIC) {
        board_profile.reserved=status.session;
        board_profile.input_crc_cycles=crc_cycles;
        board_profile.output_cycles=board_diag_ticks()-output_start;
        board_profile.service_cycles=board_diag_ticks()-service_start;
        board_profile.crc=board_crc32((const void *)&board_profile,offsetof(BoardProfile,crc));
        board_clean((const void *)&board_profile,sizeof(board_profile));
    }
#ifdef YOLO_N550_BOARD
    if(compact && !result) {
        copy_to(stream_result.candidates,board_model_output(),7200);
        if(!YOLO_FAST) {
        stream_result.model=board_profile.model_cycles;
        stream_result.service=board_diag_ticks()-service_start;
        stream_result.input_crc=crc_cycles;
        stream_result.output=board_diag_ticks()-output_start;
        stream_result.pack=board_profile.phases[PROFILE_INPUT_PACK].ticks;
        stream_result.bias=board_profile.phases[PROFILE_BIAS_DEQUANT].ticks;
        stream_result.profiled=!(request.options&DIAG_LITE);
        }
        stream_result.configuration=stream_configuration;
        /* Relinquish only the consumed slot. The other slot may currently be
         * written by host; no cache operation or CPU store touches it. */
        board_prepare_write(input_slots+slot-1,sizeof(PcieInputSlot));
    }
#endif
    unsigned error=board_diag.error==PCIE_ABORTED ? PCIE_ABORTED : PCIE_MODEL_ERROR;
    publish(result ? PCIE_ERROR : PCIE_DONE,result ? error : 0,result);
}
void board_pcie_loop(int failure) {
    board_pcie_start(failure);
    for(;;) board_pcie_poll();
}
