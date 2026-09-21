#ifndef YOLO26_BASIC_H_
#define YOLO26_BASIC_H_

#include <stddef.h>
#include <stdint.h>

typedef struct { uint32_t n, c, h, w; } Shape4;
typedef struct {
    uint32_t n, c, h, w, kh, kw, sh, sw, ph, pw;
} Pool2D;

/* Finite FP32; contiguous NCHW or flattened [outer, axis, inner].
 * Caller provides sufficient non-overlapping buffers. Add permits output==a/b.
 * Kernels return 0 on success, -1 on invalid dimensions/pointers without writes.
 * Pool: dilation=1, floor output shape, implicit negative infinity padding.
 * Nearest: positive integer height/width scale factors.
 */
int basic_pool_shape(const Pool2D *p, size_t *oh, size_t *ow);
int basic_nearest_shape(const Shape4 *p, size_t sh, size_t sw, size_t *oh, size_t *ow);
size_t basic_join_shape(size_t outer, size_t inner, size_t parts, const size_t *widths, size_t *axis);
int basic_slice_shape(size_t outer, size_t axis, size_t inner, size_t start, size_t width);

int maxpool_fp32_scalar(const Pool2D *p, const float *input, float *output);
int maxpool_fp32_rvv(const Pool2D *p, const float *input, float *output);
int nearest_fp32_scalar(const Shape4 *p, size_t sh, size_t sw, const float *input, float *output);
int nearest_fp32_rvv(const Shape4 *p, size_t sh, size_t sw, const float *input, float *output);
int add_fp32_scalar(const float *a, const float *b, float *output, size_t count);
int add_fp32_rvv(const float *a, const float *b, float *output, size_t count);
int concat_fp32_scalar(const float *const *inputs, float *output, size_t outer, size_t inner,
                       size_t parts, const size_t *widths);
int concat_fp32_rvv(const float *const *inputs, float *output, size_t outer, size_t inner,
                    size_t parts, const size_t *widths);
int slice_fp32_scalar(const float *input, float *output, size_t outer, size_t axis,
                      size_t inner, size_t start, size_t width);
int slice_fp32_rvv(const float *input, float *output, size_t outer, size_t axis,
                   size_t inner, size_t start, size_t width);

#endif
