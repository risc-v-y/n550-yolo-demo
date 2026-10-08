#ifndef YOLO_BOARD_PCIE_H
#define YOLO_BOARD_PCIE_H
#include <stdint.h>
#define PCIE_MAILBOX_ADDRESS UINT64_C(0x81f00000)
#define PCIE_MAGIC 0x50363259u
enum { PCIE_BOOT, PCIE_READY, PCIE_BUSY, PCIE_DONE, PCIE_ERROR };
enum { PCIE_OPEN=1, PCIE_RUN=2 };
enum { PCIE_BAD_REQUEST=0x40, PCIE_BAD_SESSION, PCIE_BAD_INPUT, PCIE_MODEL_ERROR };
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
void board_pcie_init(void);
void board_pcie_start(int failure);
void board_pcie_poll(void);
void board_pcie_fault(unsigned error,int detail);
void board_pcie_loop(int failure);
#endif
