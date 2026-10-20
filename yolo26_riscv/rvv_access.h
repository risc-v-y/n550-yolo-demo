#ifndef YOLO26_RVV_ACCESS_H_
#define YOLO26_RVV_ACCESS_H_
#include <stddef.h>
#include <stdint.h>
#ifdef YOLO_N550_BOARD
#include "board/cache.h"
#include "board/memory.h"
#endif

/* Synchronous kernels own complete destination cache lines. Transfer each
 * contiguous buffer once; raw m2/m4 intrinsics bypass m1 per-vector wrappers.
 * Normal cached DDR stores are fenced, NEVER invalidated. No concurrent
 * scalar or external writer may touch the destination while owned. */
static inline const void *rvv_read_buffer(const void *p, size_t bytes) {
#ifdef YOLO_N550_BOARD
    if (board_is_clp(p)) board_clean(p, bytes);
    return board_rvv_address(p);
#else
    (void)bytes;
    return p;
#endif
}
static inline void *rvv_write_buffer(void *p, size_t bytes) {
#ifdef YOLO_N550_BOARD
    if (board_is_clp(p)) board_prepare_write(p, bytes);
    return (void *)board_rvv_address(p);
#else
    (void)bytes;
    return p;
#endif
}
static inline void rvv_publish_buffer(void *p, size_t bytes) {
#ifdef YOLO_N550_BOARD
    if (board_is_clp(p)) board_finish_write(p, bytes);
    else board_fence();
#else
    (void)p; (void)bytes;
#endif
}
#endif
