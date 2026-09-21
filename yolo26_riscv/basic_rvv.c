#include "basic.h"
#include <riscv_vector.h>

static void copy(float *output, const float *input, size_t count) {
    for (size_t i = 0; i < count;) {
        size_t vl = __riscv_vsetvl_e32m1(count - i);
        vfloat32m1_t x = __riscv_vle32_v_f32m1(input + i, vl);
        __riscv_vse32_v_f32m1(output + i, x, vl);
        i += vl;
    }
}

int maxpool_fp32_rvv(const Pool2D *p, const float *input, float *output) {
    size_t oh, ow;
    if (!input || !output || basic_pool_shape(p, &oh, &ow)) return -1;
    size_t in_area = (size_t)p->h * p->w, out_area = oh * ow;
    const union { uint32_t bits; float value; } negative_inf = {.bits = 0xff800000u};
    /* Lanes cover channels. All lanes share the same spatial boundary decision. */
    for (size_t batch = 0; batch < p->n; ++batch) {
        for (size_t y = 0; y < oh; ++y) {
            for (size_t x = 0; x < ow; ++x) {
                for (size_t channel = 0; channel < p->c;) {
                    size_t vl = __riscv_vsetvl_e32m1(p->c - channel);
                    size_t plane = batch * p->c + channel;
                    vfloat32m1_t best = __riscv_vfmv_v_f_f32m1(negative_inf.value, vl);
                    for (size_t ky = 0; ky < p->kh; ++ky) {
                        int64_t iy = (int64_t)(y * p->sh + ky) - p->ph;
                        if (iy < 0 || iy >= p->h) continue;
                        for (size_t kx = 0; kx < p->kw; ++kx) {
                            int64_t ix = (int64_t)(x * p->sw + kx) - p->pw;
                            if (ix < 0 || ix >= p->w) continue;
                            const float *at = input + plane * in_area + (size_t)iy * p->w + (size_t)ix;
                            vfloat32m1_t value = __riscv_vlse32_v_f32m1(at, (ptrdiff_t)(in_area * sizeof(float)), vl);
                            best = __riscv_vfmax_vv_f32m1(best, value, vl);
                        }
                    }
                    __riscv_vsse32_v_f32m1(output + plane * out_area + y * ow + x,
                                         (ptrdiff_t)(out_area * sizeof(float)), best, vl);
                    channel += vl;
                }
            }
        }
    }
    return 0;
}

int nearest_fp32_rvv(const Shape4 *p, size_t sh, size_t sw, const float *input, float *output) {
    size_t oh, ow;
    if (!input || !output || basic_nearest_shape(p, sh, sw, &oh, &ow)) return -1;
    for (size_t plane = 0; plane < (size_t)p->n * p->c; ++plane) {
        for (size_t y = 0; y < p->h; ++y) {
            for (size_t x = 0; x < p->w;) {
                size_t vl = __riscv_vsetvl_e32m1(p->w - x);
                vfloat32m1_t value = __riscv_vle32_v_f32m1(input + plane * p->h * p->w + y * p->w + x, vl);
                for (size_t dy = 0; dy < sh; ++dy) {
                    float *row = output + plane * oh * ow + (y * sh + dy) * ow + x * sw;
                    for (size_t dx = 0; dx < sw; ++dx)
                        __riscv_vsse32_v_f32m1(row + dx, (ptrdiff_t)(sw * sizeof(float)), value, vl);
                }
                x += vl;
            }
        }
    }
    return 0;
}

int add_fp32_rvv(const float *a, const float *b, float *output, size_t count) {
    if (count > (size_t)PTRDIFF_MAX / sizeof(float) || (count && (!a || !b || !output))) return -1;
    for (size_t i = 0; i < count;) {
        size_t vl = __riscv_vsetvl_e32m1(count - i);
        vfloat32m1_t x = __riscv_vle32_v_f32m1(a + i, vl), y = __riscv_vle32_v_f32m1(b + i, vl);
        __riscv_vse32_v_f32m1(output + i, __riscv_vfadd_vv_f32m1(x, y, vl), vl);
        i += vl;
    }
    return 0;
}

int concat_fp32_rvv(const float *const *inputs, float *output, size_t outer, size_t inner,
                    size_t parts, const size_t *widths) {
    size_t axis;
    if (!inputs || !output || !basic_join_shape(outer, inner, parts, widths, &axis)) return -1;
    for (size_t part = 0; part < parts; ++part) if (widths[part] && !inputs[part]) return -1;
    for (size_t row = 0; row < outer; ++row) {
        size_t start = 0;
        for (size_t part = 0; part < parts; ++part) {
            size_t count = widths[part] * inner;
            if (count) copy(output + row * axis * inner + start, inputs[part] + row * count, count);
            start += count;
        }
    }
    return 0;
}

int slice_fp32_rvv(const float *input, float *output, size_t outer, size_t axis,
                   size_t inner, size_t start, size_t width) {
    if (basic_slice_shape(outer, axis, inner, start, width) || (width && (!input || !output))) return -1;
    if (width)
        for (size_t row = 0; row < outer; ++row)
            copy(output + row * width * inner, input + (row * axis + start) * inner, width * inner);
    return 0;
}
