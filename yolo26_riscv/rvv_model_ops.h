#ifndef YOLO26_RVV_MODEL_OPS_H_
#define YOLO26_RVV_MODEL_OPS_H_
#include <stddef.h>
#include "conv2d.h"
/* Depth multiplier=1, 3x3/stride1/pad1, finite FP32 operands. Global
 * nonoverlapping buffers; the caller owns complete output cache lines.
 * Ky/kx multiply/add order and final separate bias match ordered im2col.
 * Returns -1 before writing for unsupported shapes or overlapping buffers. */
int rvv_depthwise3x3(const Conv2D *p, const float *input, const float *weight,
                     const float *bias, float *output);
void rvv_matmul_ordered(const float *a, const float *b, float *c,
                        size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
void rvv_copy_bytes(void *output, const void *input, size_t bytes);
void rvv_nonlinear(const float *input, float *output, size_t count, int silu);
void rvv_softmax(const float *input, float *output, size_t rows, size_t width);
#endif
