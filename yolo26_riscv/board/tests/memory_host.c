#include "../memory.h"
#include <assert.h>
int main(void) {
    assert(board_is_clp((void *)(uintptr_t)0xf0000000));
    assert(board_is_clp((void *)(uintptr_t)0xf0ffffff));
    assert(!board_is_clp((void *)(uintptr_t)0xefffffff));
    assert(!board_is_clp((void *)(uintptr_t)0xf1000000));
    assert((uintptr_t)board_rvv_address((void *)(uintptr_t)0xf0000000)==0x10000000);
    assert((uintptr_t)board_rvv_address((void *)(uintptr_t)0xf0ffffff)==0x10ffffff);
    assert((uintptr_t)board_rvv_address((void *)(uintptr_t)0x80001234)==0x80001234);
    assert((uintptr_t)board_rvv_address((void *)(uintptr_t)0xf1000000)==0xf1000000);
    return 0;
}
