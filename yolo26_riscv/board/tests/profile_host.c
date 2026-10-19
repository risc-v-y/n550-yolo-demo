/* Deterministic timer fixture: actual accounting/reset helpers, no hardware. */
#include "../diagnostics.h"
#include <assert.h>
#include <stdio.h>
uint64_t board_profile_test_ticks;
int main(void) {
    board_diag_init();
    assert(offsetof(BoardDiagnostics,profile_address)==816);
    board_diag.frame=7; board_diag.options=DIAG_TRACE;
    board_profile_test_ticks=100;
    board_profile_begin(334);
    assert(board_profile.frame==7 && board_profile.options==DIAG_TRACE);
    assert(board_profile.bytes==8152 && board_profile.state==0);
    board_profile.node[333]=(BoardNodeProfile){2,7,1};
    board_profile_test_ticks=1000;
    board_profile_end(450,100);
    assert(board_profile.model_cycles==900 && board_profile.graph_cycles==550);
    assert(board_profile.input_check_cycles==100 && board_profile.state==1);
    board_diag.frame=8; board_diag.options=0;
    board_profile_test_ticks=2000;
    board_profile_begin(334);
    assert(board_profile.frame==8 && board_profile.completed==0);
    assert(board_profile.node[333].compute==0 && board_profile.state==0);
    puts("PASS profile reset, ABI prefix and frozen cycle accounting");
    return 0;
}
