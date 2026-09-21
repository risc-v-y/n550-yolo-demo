#ifndef YOLO_BOARD_COHERENT_RVV_H
#define YOLO_BOARD_COHERENT_RVV_H
#include <riscv_vector.h>
#include "cache.h"
/* Included ONLY by the board build, after the intrinsic declarations.
 * Indexed source tensors are cleaned by the graph boundary; index arrays
 * are scalar-produced and use the u32 contiguous-load wrapper below. */
BOARD_INLINE vfloat32m1_t board_vle32(const float *p, size_t n) {
    board_clean(p,n*4); return __riscv_vle32_v_f32m1(p,n);
}
BOARD_INLINE vuint32m1_t board_vleu32(const uint32_t *p, size_t n) {
    board_clean(p,n*4); return __riscv_vle32_v_u32m1(p,n);
}
BOARD_INLINE vuint8m1_t board_vle8(const uint8_t *p, size_t n) {
    board_clean(p,n); return __riscv_vle8_v_u8m1(p,n);
}
BOARD_INLINE void board_vse32(float *p, vfloat32m1_t v, size_t n) {
    board_prepare_write(p,n*4);
    __riscv_vse32_v_f32m1(p,v,n);
    board_finish_write(p,n*4);
}
BOARD_INLINE void board_vse8(uint8_t *p, vuint8m1_t v, size_t n) {
    board_prepare_write(p,n);
    __riscv_vse8_v_u8m1(p,v,n);
    board_finish_write(p,n);
}
BOARD_INLINE vfloat32m1_t board_vlse32(const float *p, ptrdiff_t stride, size_t n) {
    for(size_t i=0;i<n;++i) board_clean((const char *)p+i*stride,4);
    return __riscv_vlse32_v_f32m1(p,stride,n);
}
BOARD_INLINE void board_vsse32(float *p, ptrdiff_t stride, vfloat32m1_t v, size_t n) {
    for(size_t i=0;i<n;++i) board_prepare_write((char *)p+i*stride,4);
    __riscv_vsse32_v_f32m1(p,stride,v,n);
    for(size_t i=0;i<n;++i) board_finish_write((char *)p+i*stride,4);
}
#define __riscv_vle32_v_f32m1 board_vle32
#define __riscv_vle32_v_u32m1 board_vleu32
#define __riscv_vle8_v_u8m1 board_vle8
#define __riscv_vse32_v_f32m1 board_vse32
#define __riscv_vse8_v_u8m1 board_vse8
#define __riscv_vlse32_v_f32m1 board_vlse32
#define __riscv_vsse32_v_f32m1 board_vsse32
#endif
