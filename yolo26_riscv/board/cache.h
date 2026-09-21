#ifndef YOLO_BOARD_CACHE_H
#define YOLO_BOARD_CACHE_H
#include <stddef.h>
#include <stdint.h>
#define BOARD_CACHE_LINE 64u
#define BOARD_INLINE static inline __attribute__((always_inline))

/* Single hart, M-mode, interrupts disabled. RVV/AMU bypass DCache.
 * Ownership is transferred over complete cache lines. No scalar writes to
 * those lines are permitted between prepare_write and finish_write. */
BOARD_INLINE void board_fence(void) { __asm__ volatile("fence rw,rw" ::: "memory"); }
BOARD_INLINE void board_clean(const void *address, size_t bytes) {
    if (!bytes) return;
    uintptr_t end = (uintptr_t)address + bytes;
    board_fence();
    for (uintptr_t p = (uintptr_t)address & ~(uintptr_t)63; p < end; p += 64)
        __asm__ volatile("cbo.clean (%0)" :: "r"(p) : "memory");
    board_fence();
}
BOARD_INLINE void board_prepare_write(const void *address, size_t bytes) {
    if (!bytes) return;
    uintptr_t end = (uintptr_t)address + bytes;
    board_fence();
    for (uintptr_t p = (uintptr_t)address & ~(uintptr_t)63; p < end; p += 64)
        __asm__ volatile("cbo.flush (%0)" :: "r"(p) : "memory");
    board_fence();
}
BOARD_INLINE void board_finish_write(const void *address, size_t bytes) {
    if (!bytes) return;
    uintptr_t end = (uintptr_t)address + bytes;
    board_fence();
    for (uintptr_t p = (uintptr_t)address & ~(uintptr_t)63; p < end; p += 64)
        __asm__ volatile("cbo.inval (%0)" :: "r"(p) : "memory");
    board_fence();
}
#endif
