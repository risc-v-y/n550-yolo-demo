#ifndef YOLO_BOARD_STREAM_H
#define YOLO_BOARD_STREAM_H
#include "pcie.h"
enum { PCIE_HAS_STREAM=2, PCIE_RUN_U8=5, PCIE_CONFIGURE=6,
       STREAM_INPUT_BYTES=3*416*416 };
enum { PCIE_FAST_IMAGE=4, PCIE_RESULT_POLL=8 };
typedef struct {
    PcieRequest request;
    uint8_t input[STREAM_INPUT_BYTES];
} PcieInputSlot;
typedef struct {
    uint64_t magic,version,bytes,slots,result,profile,snapshot;
    uint64_t input_bytes,slot_bytes,result_bytes,max_rows,accumulator_bytes;
    uint64_t vector_bytes,option_mask,reserved,crc;
} PcieStreamInfo;
typedef struct {
    PcieStatus status;
    uint64_t model,service,input_crc,output,pack,bias,profiled,configuration;
    uint8_t candidates[7200];
    uint64_t crc;
} PcieStreamResult;
_Static_assert(sizeof(PcieInputSlot)==519232,"stream input slot");
_Static_assert(sizeof(PcieStreamInfo)==128,"stream capabilities");
_Static_assert(sizeof(PcieStreamResult)==7336,"stream result");
int board_model_configure(unsigned m,unsigned n,unsigned pack,unsigned compute,unsigned graph);
void board_model_set_u8(const uint8_t *data);
#endif
