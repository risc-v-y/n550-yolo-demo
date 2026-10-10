#include "diagnostics.h"
#ifdef YOLO_N550_BOARD
#include "cache.h"
#endif
#if defined(YOLO_N550_BOARD) || defined(BOARD_LOG_TEST)
#include "uart.h"
#endif
volatile BoardDiagnostics board_diag __attribute__((aligned(64)));
static uint64_t frame_start;
void board_diag_event(void) {
#if defined(YOLO_N550_BOARD) || defined(BOARD_LOG_TEST)
    if(board_uart_status) return;
    static const char *const stages[]={"UNKNOWN","BOOT","UART","AMU_INIT","SELFTEST",
        "READY","RECEIVE","RUN","DONE","FAILED","TRAP"};
    board_uart_text("[YOLO] ");
    board_uart_text(board_diag.stage<=DIAG_TRAP ? stages[board_diag.stage] : stages[0]);
    board_uart_text(" frame="); board_uart_u64(board_diag.frame);
    if(board_diag.stage==DIAG_RUN || board_diag.stage==DIAG_DONE ||
       board_diag.stage==DIAG_FAILED || board_diag.stage==DIAG_TRAP) {
        board_uart_text(" node=");
        if(board_diag.node==UINT64_MAX) board_uart_text("none");
        else board_uart_u64(board_diag.node);
        board_uart_text(" op="); board_uart_u64(board_diag.op);
        board_uart_text(" phase="); board_uart_u64(board_diag.phase);
    }
    if(board_diag.stage==DIAG_RUN && board_diag.phase==PHASE_NODE_END) {
        board_uart_text(" node_cycles="); board_uart_u64(board_diag.node_cycles);
    }
    if(board_diag.stage==DIAG_DONE) {
        board_uart_text(" frame_cycles="); board_uart_u64(board_diag_ticks()-frame_start);
    }
    if(board_diag.stage==DIAG_FAILED || board_diag.stage==DIAG_TRAP) {
        board_uart_text(" error="); board_uart_hex(board_diag.error);
        board_uart_text(" detail="); board_uart_i64((int64_t)board_diag.detail);
        board_uart_text(" tensor="); board_uart_u64(board_diag.tensor);
        board_uart_text(" element="); board_uart_u64(board_diag.element);
        board_uart_text(" got="); board_uart_hex(board_diag.observed);
        board_uart_text(" expected="); board_uart_hex(board_diag.expected);
        board_uart_text(" amu_flags="); board_uart_hex(board_diag.amu_flags);
    }
    if(board_diag.stage==DIAG_TRAP) {
        board_uart_text(" mcause="); board_uart_hex(board_diag.trap_cause);
        board_uart_text(" mepc="); board_uart_hex(board_diag.trap_pc);
        board_uart_text(" mtval="); board_uart_hex(board_diag.trap_value);
    }
    board_uart_text("\n");
    if(board_diag.stage==DIAG_RUN && board_diag.phase==PHASE_NODE_BEGIN &&
       (board_diag.options&DIAG_TRACE)) {
        for(unsigned slot=0;slot<6;++slot) {
            if(board_diag.tensors[slot][0]==UINT64_MAX) continue;
            board_uart_text("[YOLO]   "); board_uart_text(slot<4 ? "input" : "output");
            board_uart_u64(slot<4 ? slot : slot-4);
            board_uart_text(" tensor="); board_uart_u64(board_diag.tensors[slot][0]);
            board_uart_text(" dtype="); board_uart_u64(board_diag.tensors[slot][1]);
            board_uart_text(" shape=[");
            for(unsigned j=0;j<board_diag.tensors[slot][2] && j<4;++j) {
                if(j) board_uart_text(",");
                board_uart_u64(board_diag.tensors[slot][3+j]);
            }
            board_uart_text("]\n");
        }
    }
#endif
}
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
    if(stage==DIAG_RUN) frame_start=board_diag_ticks();
    board_diag.stage=stage;
    board_diag_commit();
    board_diag_event();
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
    if((board_diag.options&DIAG_TRACE) || id%32==0) board_diag_event();
    board_diag.node_start=board_diag_ticks();
}
void board_diag_trap(uintptr_t cause,uintptr_t pc,uintptr_t value,const uintptr_t *registers) {
    board_diag.trap_cause=cause; board_diag.trap_pc=pc; board_diag.trap_value=value;
    for(unsigned i=0;i<32;++i) board_diag.registers[i]=registers[i];
    board_diag.error=0x100;
    board_diag.stage=DIAG_TRAP;
    board_diag_commit(); /* Best effort: a nested cache/bus fault parks in the assembly fallback. */
    board_diag_event();
}
