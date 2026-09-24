#include "diagnostics.h"
#ifdef YOLO_N550_BOARD
#include "cache.h"
#endif
volatile BoardDiagnostics board_diag __attribute__((aligned(64)));
__attribute__((weak)) void board_diag_event(void) { }
uint64_t board_diag_ticks(void) {
#ifdef YOLO_N550_BOARD
    uint64_t ticks;
    __asm__ volatile("rdcycle %0" : "=r"(ticks));
    return ticks;
#else
    return 0;
#endif
}
void board_diag_commit(void) {
#ifdef YOLO_N550_BOARD
    board_clean((const void *)&board_diag,sizeof(board_diag));
#endif
}
void board_diag_init(void) {
    volatile uint64_t *p=(volatile uint64_t *)&board_diag;
    for(unsigned i=0;i<sizeof(board_diag)/8;++i) p[i]=0;
    board_diag.version=1;
    board_diag.node=board_diag.tensor=UINT64_MAX;
    board_diag.op=UINT64_MAX;
    board_diag.stage=DIAG_BOOT;
    board_diag_commit();
}
void board_diag_stage(unsigned stage) {
    board_diag.stage=stage;
    board_diag_commit();
}
void board_diag_phase(unsigned phase) {
    board_diag.phase=phase;
#ifdef YOLO_N550_BOARD
    board_clean((const void *)&board_diag.phase,sizeof(board_diag.phase));
#endif
}
void board_diag_error(unsigned code,int detail) {
    board_diag.error=code;
    board_diag.detail=(uint64_t)(int64_t)detail;
    board_diag_stage(DIAG_FAILED);
}
void board_diag_node(uint32_t id,const ModelNode *node,const ModelTensor *tensors) {
    board_diag.node=id; board_diag.op=node->op;
    board_diag.inputs=node->inputs; board_diag.outputs=node->outputs;
    for(unsigned slot=0;slot<6;++slot) {
        int valid=slot<4 ? slot<node->inputs : slot-4<node->outputs;
        for(unsigned j=0;j<7;++j) board_diag.tensors[slot][j]=0;
        if(!valid) { board_diag.tensors[slot][0]=UINT64_MAX; continue; }
        int index=slot<4 ? node->in[slot] : node->out[slot-4];
        const ModelTensor *t=tensors+index;
        board_diag.tensors[slot][0]=(uint64_t)index;
        board_diag.tensors[slot][1]=t->dtype;
        board_diag.tensors[slot][2]=t->rank;
        for(unsigned j=0;j<4;++j) board_diag.tensors[slot][3+j]=t->shape[j];
    }
    board_diag.phase=PHASE_NODE_BEGIN;
    board_diag.node_cycles=0;
    board_diag_commit();
    if(board_diag.options&DIAG_TRACE) board_diag_event();
    board_diag.node_start=board_diag_ticks();
}
void board_diag_trap(uintptr_t cause,uintptr_t pc,uintptr_t value,const uintptr_t *registers) {
    board_diag.trap_cause=cause; board_diag.trap_pc=pc; board_diag.trap_value=value;
    for(unsigned i=0;i<32;++i) board_diag.registers[i]=registers[i];
    board_diag.error=0x100;
    board_diag.stage=DIAG_TRAP;
    board_diag_commit(); /* Best effort: a nested cache/bus fault parks in the assembly fallback. */
}
