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
/* Contiguous left/output, periodic right; op: 0 add, 1 subtract, 2 multiply.
 * Supports exact in-place elementwise output; no partial overlapping ranges. */
int rvv_binary_periodic(const float *left, const float *right, float *output,
                        size_t count, size_t period, unsigned op);
/* In-place [channels,columns] plus one scalar bias per channel. Owns the
 * complete output cache lines for the synchronous call; bias must not overlap.
 * Each element receives one independent FP32 add after its dot product. */
int rvv_bias_channels(float *output, const float *bias, size_t channels, size_t columns);
void rvv_transpose(const float *input, float *output,
                   size_t batches, size_t rows, size_t columns);
void rvv_max_rows(const float *input, float *values, int64_t *indices,
                  size_t rows, size_t width);
int rvv_gather_rows(const void *input, const int64_t *indices, void *output,
                     size_t outer, size_t axis, size_t inner,
                     size_t out_axis, size_t element_bytes);
void rvv_repeat_i64(const int64_t *input, int64_t *output,
                     size_t count, size_t repeats);
#endif
