#ifndef YOLO_BOARD_PROTOCOL_H
#define YOLO_BOARD_PROTOCOL_H
#include <stdint.h>
#include <stddef.h>
#define RPC_LIMIT 1024u
enum { RPC_HELLO=1, RPC_INFO, RPC_BEGIN, RPC_DATA, RPC_RUN, RPC_ACK,
       RPC_GET, RPC_RESULT, RPC_ERROR, RPC_PING, RPC_PONG, RPC_STOP };
typedef struct {
    uint32_t magic;
    uint16_t version, kind;
    uint32_t seq, frame, offset, length, payload_crc, header_crc;
} RpcHeader;
_Static_assert(sizeof(RpcHeader)==32,"wire header size");
uint32_t board_crc32(const void *data, size_t bytes);
unsigned char *board_model_input(void);
const unsigned char *board_model_output(void);
uint32_t board_model_input_size(void);
uint32_t board_model_output_size(void);
int board_model_run(void);
void board_rpc_loop(void);
#endif
