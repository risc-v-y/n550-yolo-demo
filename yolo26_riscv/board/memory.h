#ifndef YOLO_BOARD_MEMORY_H
#define YOLO_BOARD_MEMORY_H
#include <stddef.h>
#include <stdint.h>
/* Platform-approved model data region; code, weights, stacks and mailbox
 * remain in 0x80000000..0x81ffffff. QEMU has unified RAM, no CLP remapping. */
#ifdef BOARD_CLP_SIM
#define BOARD_CLP_DDR_BASE ((uintptr_t)0x84000000u)
#define BOARD_CLP_RVV_BASE BOARD_CLP_DDR_BASE
#else
#define BOARD_CLP_DDR_BASE ((uintptr_t)0xf0000000u)
#define BOARD_CLP_RVV_BASE ((uintptr_t)0x10000000u)
#endif
#define BOARD_CLP_BYTES ((size_t)0x1000000u)
#define BOARD_CLP_DATA __attribute__((section(".clp_data"),aligned(64)))
static inline int board_is_clp(const void *pointer) {
    uintptr_t address=(uintptr_t)pointer;
    return address>=BOARD_CLP_DDR_BASE && address-BOARD_CLP_DDR_BASE<BOARD_CLP_BYTES;
}
static inline void *board_rvv_address(const void *pointer) {
    if(!board_is_clp(pointer)) return (void *)pointer;
    return (void *)(BOARD_CLP_RVV_BASE+((uintptr_t)pointer-BOARD_CLP_DDR_BASE));
}
extern unsigned char __clp_start[],__clp_end[];
#endif
