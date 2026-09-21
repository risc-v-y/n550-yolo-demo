#ifndef YOLO26_CONV2D_H_
#define YOLO26_CONV2D_H_

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t batch, in_channels, in_h, in_w, out_channels;
    uint32_t kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, groups;
} Conv2D;

/* NCHW input/output; OIHW/group weights; dilation=1, symmetric zero padding.
 * Buffers must not overlap and must cover the shapes. Bias may be NULL.
 * Convolution entry points return -1 on error without output/scratch writes.
 * Shape helper returns 0/-1; scratch helper returns 0 for an invalid request.
 */
int conv2d_output_shape(const Conv2D *p, size_t *out_h, size_t *out_w);
size_t conv2d_scratch_floats(const Conv2D *p, size_t tile_columns);
int conv2d_fp32_scalar(const Conv2D *p, const float *input, const float *weight,
                       const float *bias, float *output);
int conv2d_fp32_rvv(const Conv2D *p, const float *input, const float *weight,
                    const float *bias, float *output, float *scratch,
                    size_t scratch_floats, size_t tile_columns);

/* In-place SiLU, using our scalar range-reduced exponential approximation.
 * Saturates exp(-abs(x)) beyond |x|=80. No external activation kernel/libm.
 */
float silu_fp32_value(float value);
void silu_fp32(float *values, size_t count);

#endif
