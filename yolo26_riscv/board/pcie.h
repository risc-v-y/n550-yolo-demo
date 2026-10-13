#ifndef YOLO_BOARD_PCIE_H
#define YOLO_BOARD_PCIE_H
#include <stdint.h>
#include "../scalar_model.h"
#define PCIE_MAILBOX_ADDRESS UINT64_C(0x81f00000)
#define PCIE_MAGIC 0x50363259u
enum { PCIE_BOOT, PCIE_READY, PCIE_BUSY, PCIE_DONE, PCIE_ERROR, PCIE_NODE };
enum { PCIE_OPEN=1, PCIE_RUN=2, PCIE_ADVANCE=3, PCIE_ABORT=4 };
enum { PCIE_CAPTURE=8, PCIE_HAS_CAPTURE=1 };
enum { PCIE_BAD_REQUEST=0x40, PCIE_BAD_SESSION, PCIE_BAD_INPUT, PCIE_MODEL_ERROR, PCIE_ABORTED };
typedef struct {
    uint32_t magic,version,bytes,reserved;
    uint64_t input_address,output_address,diagnostic_address;
    uint32_t input_bytes,output_bytes,diagnostic_bytes,nodes,reserved2,crc;
} PcieInfo;
typedef struct {
    uint64_t session,frame;
    uint32_t command,sequence,input_bytes,input_crc,options,reserved[6],crc;
} PcieRequest;
typedef struct {
    uint32_t magic,version,state,error;
    uint64_t session,frame;
    uint32_t sequence,output_bytes,output_crc;
    int32_t detail;
    uint32_t generation,reserved[2],crc;
} PcieStatus;
typedef struct {
    PcieInfo info;             /* Board owns this cache line. */
    PcieRequest request;       /* Host writes body before doorbell. */
    uint32_t doorbell[16];     /* Host owns a separate cache line. */
    PcieStatus status;         /* Board owns this cache line. */
} PcieMailbox;
_Static_assert(sizeof(PcieInfo)==64,"PCIe info");
_Static_assert(sizeof(PcieRequest)==64,"PCIe request");
_Static_assert(sizeof(PcieStatus)==64,"PCIe status");
_Static_assert(sizeof(PcieMailbox)==256,"PCIe mailbox");
extern volatile PcieMailbox board_pcie_mailbox;
/* Node capture is opt-in. CRCs cover the values before/after the operation;
 * input storage may be reused by an output. Descriptors use global addresses,
 * never the RVV alias. The host owns no tensor until PCIE_NODE is published. */
typedef struct {
    int32_t id;
    uint32_t dtype,rank,bytes;
    uint64_t address;
    uint32_t constant,count,shape[4],crc_before,crc_after,reserved[2];
} PcieTensorSnapshot;
typedef struct {
    uint32_t magic,version,bytes,node;
    uint64_t session,frame;
    uint32_t sequence,op,inputs,outputs,token,options,reserved[2];
    PcieTensorSnapshot tensors[6];
    uint32_t tail[15],crc;
} PcieNodeSnapshot;
_Static_assert(sizeof(PcieTensorSnapshot)==64,"PCIe tensor snapshot");
_Static_assert(sizeof(PcieNodeSnapshot)==512,"PCIe node snapshot");
extern volatile PcieNodeSnapshot board_node_snapshot;
void board_pcie_node_begin(uint32_t id,const ModelNode *node,const ModelTensor *tensors,
                           unsigned char *arena,const unsigned char *constants);
int board_pcie_node_end(void);
void board_pcie_init(void);
void board_pcie_start(int failure);
void board_pcie_poll(void);
void board_pcie_fault(unsigned error,int detail);
void board_pcie_loop(int failure);
#endif
