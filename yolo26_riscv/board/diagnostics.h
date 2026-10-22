#ifndef YOLO_BOARD_DIAGNOSTICS_H
#define YOLO_BOARD_DIAGNOSTICS_H
#include <stdint.h>
#include <stddef.h>
#include "../scalar_model.h"
#include "../model_profile.h"

#define BOARD_PROFILE_NODES 334u
#define BOARD_PROFILE_MAGIC UINT64_C(0x3152465036325955)
/* Little-endian uint64 fields. CRC covers all bytes before crc; read only
 * after PCIe DONE. Timings are CPU cycles, never QEMU performance estimates. */
typedef struct {
    uint64_t cache, compute, check, phase_cycles[PROFILE_PHASE_COUNT];
} BoardNodeProfile;
typedef struct {
    uint64_t magic,version,bytes,frame,nodes,completed,options,state;
    uint64_t model_cycles,graph_cycles,input_check_cycles,service_cycles;
    uint64_t input_crc_cycles,output_cycles,node_cycles,reserved;
    BoardNodeProfile node[BOARD_PROFILE_NODES];
    ModelProfilePhase phases[PROFILE_PHASE_COUNT];
    uint64_t weight_pack_bytes,weight_pack_values,weight_pack_cycles,prepared_layers;
    uint64_t runtime_weight_conversions;
    uint64_t crc;
} BoardProfile;
_Static_assert(sizeof(BoardProfile)==24320,"profile wire layout v2/v3");
extern volatile BoardProfile board_profile;
void board_profile_begin(unsigned nodes);
void board_profile_end(uint64_t graph_start,uint64_t input_cycles);
enum { DIAG_BOOT=1, DIAG_UART, DIAG_AMU_INIT, DIAG_SELFTEST, DIAG_READY,
       DIAG_RECEIVE, DIAG_RUN, DIAG_DONE, DIAG_FAILED, DIAG_TRAP, DIAG_WEIGHTS };
enum { PHASE_NONE, PHASE_INPUT, PHASE_NODE_BEGIN, PHASE_CACHE, PHASE_COMPUTE,
       PHASE_AMU_PACK, PHASE_AMU_SUBMIT, PHASE_AMU_WAIT, PHASE_AMU_STORE,
       PHASE_CHECK, PHASE_NODE_END };
enum { DIAG_EVENTS=1, DIAG_TRACE=2, DIAG_FINITE=4, DIAG_LITE=16 };
/* All fields are little-endian uint64, including signed error detail (2's complement).
 * Fixed ELF symbol, 64-byte aligned. Legacy RX/RPC counters remain reserved
 * to preserve the snapshot ABI; the physical UART only transmits debug text. */
typedef struct {
    uint64_t version, stage, frame, node, op, phase, error, detail;
    uint64_t received, rx_errors, rx_timeouts, tx_errors, crc_errors, protocol_errors, retries;
    uint64_t amu_flags, tensor, element, observed, expected;
    uint64_t trap_cause, trap_pc, trap_value, node_start, node_cycles, inputs, outputs, options;
    uint64_t tensors[6][7]; /* id,dtype,rank,shape[4]; four inputs then two outputs */
    uint64_t registers[32]; /* integer x0..x31, meaningful after a CPU trap */
    /* Optional extension; the original 816-byte version-1 prefix is stable. */
    uint64_t profile_address,profile_bytes;
} BoardDiagnostics;
_Static_assert(offsetof(BoardDiagnostics,profile_address)==816,"diagnostic prefix");
_Static_assert(sizeof(BoardDiagnostics)==832,"diagnostic wire layout");
extern volatile BoardDiagnostics board_diag;
void board_diag_init(void);
void board_diag_commit(void);
void board_diag_stage(unsigned stage);
void board_diag_phase(unsigned phase);
void board_diag_error(unsigned code, int detail);
void board_diag_node(uint32_t id, const ModelNode *node, const ModelTensor *tensors);
void board_diag_trap(uintptr_t cause, uintptr_t pc, uintptr_t value, const uintptr_t *registers);
uint64_t board_diag_ticks(void);
/* Best-effort debug text. Native PCIe fixtures keep this silent. */
void board_diag_event(void);

/* First input-block probe, enabled only by DIAG_TRACE on node 0. Separate
 * from the stable PCIe diagnostics/profile wire layouts. Ordinary frames
 * never enable it. All fields are u64; signed row uses two's complement. */
#define BOARD_PACK_MAGIC UINT64_C(0x314b435036325959)
enum {
    PACK_BEGIN=1, PACK_PREPARE_BEGIN, PACK_PREPARE_DONE, PACK_ROW_CHECK,
    PACK_ROW_FILL_BEGIN, PACK_ROW_FILL_DONE, PACK_ROW_HIT, PACK_ROWS_DONE,
    PACK_INDEX_BEGIN, PACK_INDEX_DONE, PACK_GATHER_BEGIN, PACK_GATHER_DONE,
    PACK_STORE_BEGIN, PACK_STORE_DONE, PACK_COLUMN_DONE, PACK_BLOCK_DONE,
    PACK_GUARD_FAIL
};
typedef struct {
    uint64_t magic,version,bytes,generation,session,frame,node,stage;
    uint64_t first,columns,column,inner,row,vl,events,error;
    uint64_t input,cache,output,cache_bytes,row0,row1,row2,converted;
    uint64_t next_column,next_inner,row_visits,elapsed,started,updated,k,checksum;
} BoardPackProbe;
_Static_assert(sizeof(BoardPackProbe)==256,"pack probe layout");
extern volatile BoardPackProbe board_pack_probe;
int board_pack_begin(size_t first,size_t columns,size_t k,const void *input,
                     const void *cache,void *output,size_t cache_bytes);
int board_pack_mark(unsigned stage,size_t column,size_t inner,int64_t row,
                    size_t vl,const size_t tags[3],uint64_t converted);
#endif
