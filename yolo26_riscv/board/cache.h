#ifndef YOLO_BOARD_CACHE_H
#define YOLO_BOARD_CACHE_H
#include <stddef.h>
#include <stdint.h>
#define BOARD_CACHE_LINE 64u
#define BOARD_INLINE static inline __attribute__((always_inline))

/* Single hart, M-mode, interrupts disabled. Memory-port RVV uses DCache;
 * only RVV CLP-window accesses bypass it. AMU/PCIe are external writers.
 * prepare/finish apply to external writes, NOT cached RVV stores: invalidating
 * a dirty RVV destination can discard its data. No scalar writes to a shared
 * line are allowed while an external writer owns it. */
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
