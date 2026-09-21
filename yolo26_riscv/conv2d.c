#include "conv2d.h"
#include "matmul.h"

#include <riscv_vector.h>

static int multiply(size_t a, size_t b, size_t *result) {
    if (b != 0 && a > SIZE_MAX / b) return -1;
    *result = a * b;
    return 0;
}

int conv2d_output_shape(const Conv2D *p, size_t *oh, size_t *ow) {
    if (!p || !oh || !ow || !p->batch || !p->in_channels || !p->in_h || !p->in_w ||
        !p->out_channels || !p->kernel_h || !p->kernel_w || !p->stride_h ||
        !p->stride_w || !p->groups || p->in_channels % p->groups ||
        p->out_channels % p->groups) return -1;
    uint64_t padded_h = (uint64_t)p->in_h + 2ull * p->pad_h;
    uint64_t padded_w = (uint64_t)p->in_w + 2ull * p->pad_w;
    if (padded_h < p->kernel_h || padded_w < p->kernel_w) return -1;
    *oh = (size_t)((padded_h - p->kernel_h) / p->stride_h + 1);
    *ow = (size_t)((padded_w - p->kernel_w) / p->stride_w + 1);
    size_t size;
    if (multiply(p->batch, p->in_channels, &size) || multiply(size, p->in_h, &size) ||
        multiply(size, p->in_w, &size) || size > SIZE_MAX / sizeof(float)) return -1;
    if (multiply(p->out_channels, p->in_channels / p->groups, &size) ||
        multiply(size, p->kernel_h, &size) || multiply(size, p->kernel_w, &size) ||
        size > SIZE_MAX / sizeof(float)) return -1;
    if (multiply(p->batch, p->out_channels, &size) || multiply(size, *oh, &size) ||
        multiply(size, *ow, &size) || size > SIZE_MAX / sizeof(float)) return -1;
    return 0;
}

size_t conv2d_scratch_floats(const Conv2D *p, size_t columns) {
    size_t oh, ow, size;
    if (!columns || conv2d_output_shape(p, &oh, &ow)) return 0;
    if (multiply(p->in_channels / p->groups, p->kernel_h, &size) ||
        multiply(size, p->kernel_w, &size) || multiply(size, columns, &size) ||
        size > SIZE_MAX / sizeof(float)) return 0;
    return size;
}

int conv2d_fp32_scalar(const Conv2D *p, const float *input, const float *weight,
                       const float *bias, float *output) {
    size_t oh, ow;
    if (!input || !weight || !output || conv2d_output_shape(p, &oh, &ow)) return -1;
    size_t icg = p->in_channels / p->groups, ocg = p->out_channels / p->groups;
    for (size_t batch = 0; batch < p->batch; ++batch) {
        for (size_t oc = 0; oc < p->out_channels; ++oc) {
            size_t group = oc / ocg;
            for (size_t y = 0; y < oh; ++y) {
                for (size_t x = 0; x < ow; ++x) {
                    float sum = 0.0f;
                    for (size_t ic = 0; ic < icg; ++ic) {
                        for (size_t ky = 0; ky < p->kernel_h; ++ky) {
                            int64_t iy = (int64_t)(y * p->stride_h + ky) - p->pad_h;
                            for (size_t kx = 0; kx < p->kernel_w; ++kx) {
                                int64_t ix = (int64_t)(x * p->stride_w + kx) - p->pad_w;
                                if (iy < 0 || ix < 0 || iy >= p->in_h || ix >= p->in_w) continue;
                                size_t at = ((batch * p->in_channels + group * icg + ic) * p->in_h + (size_t)iy) * p->in_w + (size_t)ix;
                                size_t wt = ((oc * icg + ic) * p->kernel_h + ky) * p->kernel_w + kx;
                                sum += input[at] * weight[wt];
                            }
                        }
                    }
                    output[(batch * p->out_channels + oc) * oh * ow + y * ow + x] = sum + (bias ? bias[oc] : 0.0f);
                }
            }
        }
    }
    return 0;
}

int conv2d_fp32_rvv(const Conv2D *p, const float *input, const float *weight,
                    const float *bias, float *output, float *scratch,
                    size_t scratch_floats, size_t columns) {
    size_t oh, ow;
    if (!input || !weight || !output || !scratch || conv2d_output_shape(p, &oh, &ow)) return -1;
    size_t required = conv2d_scratch_floats(p, columns);
    if (!required || scratch_floats < required) return -1;
    size_t icg = p->in_channels / p->groups, ocg = p->out_channels / p->groups;
    size_t k = icg * p->kernel_h * p->kernel_w, area = oh * ow;
    for (size_t batch = 0; batch < p->batch; ++batch) {
        for (size_t group = 0; group < p->groups; ++group) {
            for (size_t first = 0; first < area;) {
                size_t count = area - first < columns ? area - first : columns;
                size_t inner = 0;
                /* Pack only K x columns, preserving the weight's IC/KH/KW order. */
                for (size_t ic = 0; ic < icg; ++ic) {
                    for (size_t ky = 0; ky < p->kernel_h; ++ky) {
                        for (size_t kx = 0; kx < p->kernel_w; ++kx, ++inner) {
                            for (size_t col = 0; col < count; ++col) {
                                size_t position = first + col;
                                int64_t iy = (int64_t)(position / ow * p->stride_h + ky) - p->pad_h;
                                int64_t ix = (int64_t)(position % ow * p->stride_w + kx) - p->pad_w;
                                float value = 0.0f;
                                if (iy >= 0 && ix >= 0 && iy < p->in_h && ix < p->in_w) {
                                    size_t at = ((batch * p->in_channels + group * icg + ic) * p->in_h + (size_t)iy) * p->in_w + (size_t)ix;
                                    value = input[at];
                                }
                                scratch[inner * columns + col] = value;
                            }
                        }
                    }
                }
                float *dest = output + (batch * p->out_channels + group * ocg) * area + first;
                matmul_fp32_rvv(weight + group * ocg * k, scratch, dest,
                                ocg, count, k, k, columns, area);
                if (bias) {
                    for (size_t oc = 0; oc < ocg; ++oc) {
                        for (size_t col = 0; col < count;) {
                            size_t vl = __riscv_vsetvl_e32m1(count - col);
                            vfloat32m1_t data = __riscv_vle32_v_f32m1(dest + oc * area + col, vl);
                            data = __riscv_vfadd_vf_f32m1(data, bias[group * ocg + oc], vl);
                            __riscv_vse32_v_f32m1(dest + oc * area + col, data, vl);
                            col += vl;
                        }
                    }
                }
                first += count;
            }
        }
    }
    return 0;
}
