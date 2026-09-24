#include "protocol.h"
#include "uart.h"
#include "diagnostics.h"
#define MAGIC 0x42363259u /* Y26B, little endian */
static unsigned char payload[RPC_LIMIT], answer[RPC_LIMIT];
static RpcHeader last_request, last_answer;
static int have_last;
static RpcHeader event_request;
static int event_active;

uint32_t board_crc32(const void *data, size_t bytes) {
    const unsigned char *p=data;
    uint32_t crc=~0u;
    while(bytes--) {
        crc ^= *p++;
        for(unsigned i=0;i<8;++i) crc=(crc>>1) ^ (0xedb88320u & (0u-(crc&1)));
    }
    return ~crc;
}
static int read_bytes(void *data, size_t n) {
    unsigned char *p=data;
    while(n--) if(board_uart_get(p++)) return -1;
    return 0;
}
static void send_packet(RpcHeader h, const void *data) {
    h.magic=MAGIC; h.version=1;
    h.payload_crc=board_crc32(data,h.length);
    h.header_crc=board_crc32(&h,28);
    const unsigned char *p=(const unsigned char *)&h;
    for(size_t i=0;i<32;++i) if(board_uart_put(p[i])) return;
    p=data;
    for(size_t i=0;i<h.length;++i) if(board_uart_put(p[i])) return;
}
void board_diag_event(void) {
    if(!event_active || !(board_diag.options&DIAG_EVENTS)) return;
    RpcHeader h=event_request;
    h.kind=RPC_DIAG; h.length=sizeof(board_diag);
    board_diag_commit();
    send_packet(h,(const void *)&board_diag);
}
static int request(RpcHeader *h) {
    const unsigned char magic[4]={'Y','2','6','B'};
    unsigned match=0;
    while(match<4) {
        unsigned char c;
        if(board_uart_get(&c)) { match=0; continue; }
        match = c==magic[match] ? match+1 : (c=='Y' ? 1 : 0);
    }
    h->magic=MAGIC;
    if(read_bytes((unsigned char *)h+4,28)) return -1;
    if(h->header_crc!=board_crc32(h,28)) { ++board_diag.crc_errors; board_diag_commit(); return -1; }
    if(h->version!=1 || h->length>RPC_LIMIT) { ++board_diag.protocol_errors; board_diag_commit(); return -1; }
    if(read_bytes(payload,h->length)) return -1;
    if(h->payload_crc!=board_crc32(payload,h->length)) { ++board_diag.crc_errors; board_diag_commit(); return -1; }
    return 0;
}
static int equal_request(const RpcHeader *a,const RpcHeader *b) {
    return a->kind==b->kind && a->seq==b->seq && a->frame==b->frame &&
           a->offset==b->offset && a->length==b->length && a->payload_crc==b->payload_crc;
}
void board_rpc_loop(void) {
    uint32_t active_frame=0, received=0;
    int active=0, ready=0;
    for(;;) {
        RpcHeader h;
        if(request(&h)) continue; /* Host retries a bounded number of times. */
        if(have_last && equal_request(&h,&last_request)) {
            ++board_diag.retries; board_diag_commit();
            event_request=h; event_active=1; board_diag_event(); event_active=0;
            send_packet(last_answer,answer); continue;
        }
        event_request=h; event_active=1;
        RpcHeader response=h;
        response.kind=RPC_ACK; response.length=0;
        uint32_t error=0;
        if(h.kind==RPC_DIAG_GET && !h.length) {
            response.kind=RPC_DIAG; response.length=sizeof(board_diag);
            board_diag_commit();
            for(size_t i=0;i<sizeof(board_diag);++i) answer[i]=((const volatile unsigned char *)&board_diag)[i];
        } else if(h.kind==RPC_DIAG_CONFIG && h.length==4) {
            uint32_t flags=(uint32_t)payload[0] | (uint32_t)payload[1]<<8 |
                           (uint32_t)payload[2]<<16 | (uint32_t)payload[3]<<24;
            if(flags&~7u) error=2;
            else { board_diag.options=flags; board_diag_commit(); }
        } else if(h.kind==RPC_HELLO && !h.length) {
            active=ready=0; received=0;
            board_diag.options=0; board_diag_commit();
            uint32_t info[]={board_model_input_size(),board_model_output_size(),334,
                             board_uart_baud,board_uart_dlf_bits,64};
            response.kind=RPC_INFO; response.length=sizeof(info);
            for(size_t i=0;i<sizeof(info);++i) answer[i]=((unsigned char *)info)[i];
        } else if(h.kind==RPC_PING) {
            response.kind=RPC_PONG; response.length=h.length;
            for(size_t i=0;i<h.length;++i) answer[i]=payload[i];
        } else if(h.kind==RPC_BEGIN && !h.length) {
            if(board_diag.error && board_diag.error<0x30) error=3;
            else {
                active=1; ready=0; received=0; active_frame=h.frame;
                board_diag.frame=h.frame; board_diag.received=0;
                board_diag.error=board_diag.detail=0;
                board_diag.node=board_diag.tensor=UINT64_MAX;
                board_diag.phase=PHASE_INPUT;
                board_diag_stage(DIAG_RECEIVE); board_diag_event();
            }
        } else if(h.kind==RPC_STOP && !h.length) {
            active=ready=0;
        } else if(!active || h.frame!=active_frame) error=1;
        else if(h.kind==RPC_DATA && !ready && h.length && h.offset==received &&
                received <= board_model_input_size() && h.length <= board_model_input_size()-received) {
            unsigned char *input=board_model_input();
            for(size_t i=0;i<h.length;++i) input[received+i]=payload[i];
            received+=h.length;
            board_diag.received=received; board_diag_commit();
        } else if(h.kind==RPC_RUN && !h.length && !ready && received==board_model_input_size()) {
            if(board_model_run()) { error=3; active=0; }
            else ready=1;
        } else if(h.kind==RPC_GET && !h.length && ready && h.offset<board_model_output_size()) {
            response.kind=RPC_RESULT;
            response.length=board_model_output_size()-h.offset;
            if(response.length>RPC_LIMIT) response.length=RPC_LIMIT;
            const unsigned char *out=board_model_output()+h.offset;
            for(size_t i=0;i<response.length;++i) answer[i]=out[i];
        } else error=2;
        if(error) {
            ++board_diag.protocol_errors; board_diag_commit(); board_diag_event();
            response.kind=RPC_ERROR; response.length=4;
            for(unsigned i=0;i<4;++i) answer[i]=(unsigned char)(error>>(8*i));
        }
        last_request=h; last_answer=response; have_last=1;
        send_packet(response,answer);
        event_active=0;
    }
}
