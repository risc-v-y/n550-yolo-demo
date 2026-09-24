#ifndef YOLO_BOARD_DIAGNOSTICS_H
#define YOLO_BOARD_DIAGNOSTICS_H
#include <stdint.h>
#include "../scalar_model.h"
enum { DIAG_BOOT=1, DIAG_UART, DIAG_AMU_INIT, DIAG_SELFTEST, DIAG_READY,
       DIAG_RECEIVE, DIAG_RUN, DIAG_DONE, DIAG_FAILED, DIAG_TRAP };
enum { PHASE_NONE, PHASE_INPUT, PHASE_NODE_BEGIN, PHASE_CACHE, PHASE_COMPUTE,
       PHASE_AMU_PACK, PHASE_AMU_SUBMIT, PHASE_AMU_WAIT, PHASE_AMU_STORE,
       PHASE_CHECK, PHASE_NODE_END };
enum { DIAG_EVENTS=1, DIAG_TRACE=2, DIAG_FINITE=4 };
/* All fields are little-endian uint64, including signed error detail (2's complement).
 * Fixed ELF symbol, 64-byte aligned; no heap or strings needed to inspect it. */
typedef struct {
    uint64_t version, stage, frame, node, op, phase, error, detail;
    uint64_t received, rx_errors, rx_timeouts, tx_errors, crc_errors, protocol_errors, retries;
    uint64_t amu_flags, tensor, element, observed, expected;
    uint64_t trap_cause, trap_pc, trap_value, node_start, node_cycles, inputs, outputs, options;
    uint64_t tensors[6][7]; /* id,dtype,rank,shape[4]; four inputs then two outputs */
    uint64_t registers[32]; /* integer x0..x31, meaningful after a CPU trap */
} BoardDiagnostics;
_Static_assert(sizeof(BoardDiagnostics)==816,"diagnostic wire layout");
extern volatile BoardDiagnostics board_diag;
void board_diag_init(void);
void board_diag_commit(void);
void board_diag_stage(unsigned stage);
void board_diag_phase(unsigned phase);
void board_diag_error(unsigned code, int detail);
void board_diag_node(uint32_t id, const ModelNode *node, const ModelTensor *tensors);
void board_diag_trap(uintptr_t cause, uintptr_t pc, uintptr_t value, const uintptr_t *registers);
uint64_t board_diag_ticks(void);
/* Protocol supplies this only while a request is active; no raw UART printf. */
void board_diag_event(void);
#endif
