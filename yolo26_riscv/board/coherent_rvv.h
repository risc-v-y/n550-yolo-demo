#ifndef YOLO_BOARD_COHERENT_RVV_H
#define YOLO_BOARD_COHERENT_RVV_H
#include <riscv_vector.h>
#include "cache.h"
#include "memory.h"
/* Scalar pointers use global DDR addresses. Remap model-data RVV accesses
 * to the uncached CLP alias, with cache ownership transfer for scalar users.
 * Normal RAM (weights, stack indices) uses cached RVV and MUST NOT be
 * invalidated after a vector store. Indexed source tensors are cleaned at
 * the graph boundary; their full extent is not available in the intrinsic. */
BOARD_INLINE vfloat32m1_t board_vle32(const float *p, size_t n) {
    if(board_is_clp(p)) board_clean(p,n*4);
    return __riscv_vle32_v_f32m1((const float *)board_rvv_address(p),n);
}
BOARD_INLINE vuint32m1_t board_vleu32(const uint32_t *p, size_t n) {
    if(board_is_clp(p)) board_clean(p,n*4);
    return __riscv_vle32_v_u32m1((const uint32_t *)board_rvv_address(p),n);
}
BOARD_INLINE vuint8m1_t board_vle8(const uint8_t *p, size_t n) {
    if(board_is_clp(p)) board_clean(p,n);
    return __riscv_vle8_v_u8m1((const uint8_t *)board_rvv_address(p),n);
}
BOARD_INLINE void board_vse32(float *p, vfloat32m1_t v, size_t n) {
    int clp=board_is_clp(p);
    if(clp) board_prepare_write(p,n*4);
    __riscv_vse32_v_f32m1((float *)board_rvv_address(p),v,n);
    if(clp) board_finish_write(p,n*4); else board_fence();
}
BOARD_INLINE void board_vse8(uint8_t *p, vuint8m1_t v, size_t n) {
    int clp=board_is_clp(p);
    if(clp) board_prepare_write(p,n);
    __riscv_vse8_v_u8m1((uint8_t *)board_rvv_address(p),v,n);
    if(clp) board_finish_write(p,n); else board_fence();
}
BOARD_INLINE vfloat32m1_t board_vlse32(const float *p, ptrdiff_t stride, size_t n) {
    if(board_is_clp(p)) for(size_t i=0;i<n;++i) board_clean((const char *)p+i*stride,4);
    return __riscv_vlse32_v_f32m1((const float *)board_rvv_address(p),stride,n);
}
BOARD_INLINE void board_vsse32(float *p, ptrdiff_t stride, vfloat32m1_t v, size_t n) {
    int clp=board_is_clp(p);
    if(clp) for(size_t i=0;i<n;++i) board_prepare_write((char *)p+i*stride,4);
    __riscv_vsse32_v_f32m1((float *)board_rvv_address(p),stride,v,n);
    if(clp) for(size_t i=0;i<n;++i) board_finish_write((char *)p+i*stride,4);
    else board_fence();
}
BOARD_INLINE vfloat32m1_t board_vluxei32(const float *p,vuint32m1_t indices,size_t n) {
    board_fence();
    return __riscv_vluxei32_v_f32m1((const float *)board_rvv_address(p),indices,n);
}
#define __riscv_vle32_v_f32m1 board_vle32
#define __riscv_vle32_v_u32m1 board_vleu32
#define __riscv_vle8_v_u8m1 board_vle8
#define __riscv_vse32_v_f32m1 board_vse32
#define __riscv_vse8_v_u8m1 board_vse8
#define __riscv_vlse32_v_f32m1 board_vlse32
#define __riscv_vsse32_v_f32m1 board_vsse32
#define __riscv_vluxei32_v_f32m1 board_vluxei32
#endif
