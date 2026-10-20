#include "basic.h"
#include "rvv_access.h"
#include <riscv_vector.h>

static void copy(float *output, const float *input, size_t count) {
    for (size_t i = 0; i < count;) {
        size_t vl = __riscv_vsetvl_e32m2(count - i);
        vfloat32m2_t x = __riscv_vle32_v_f32m2(input + i, vl);
        __riscv_vse32_v_f32m2(output + i, x, vl);
        i += vl;
    }
}

int maxpool_fp32_rvv(const Pool2D *p, const float *input, float *output) {
    size_t oh, ow;
    if (!input || !output || basic_pool_shape(p, &oh, &ow)) return -1;
    size_t in_area = (size_t)p->h * p->w, out_area = oh * ow;
    const float *source = rvv_read_buffer(input, (size_t)p->n*p->c*in_area*4);
    float *dest = rvv_write_buffer(output, (size_t)p->n*p->c*out_area*4);
    const union { uint32_t bits; float value; } negative_inf = {.bits = 0xff800000u};
    if (in_area <= UINT32_MAX/4 && out_area <= UINT32_MAX &&
        (uint64_t)p->h+p->ph+p->kh < UINT32_MAX &&
        (uint64_t)p->w+p->pw+p->kw < UINT32_MAX) {
        /* NCHW: lanes follow spatial pixels, not distant channel planes.
         * Build one spatial plan, reuse across all batch/channel planes. */
        for (size_t first = 0; first < out_area;) {
            size_t vl = __riscv_vsetvl_e32m2(out_area-first);
            vuint32m2_t position = __riscv_vadd_vx_u32m2(__riscv_vid_v_u32m2(vl), first, vl);
            vuint32m2_t yy = __riscv_vsub_vx_u32m2(__riscv_vmul_vx_u32m2(
                __riscv_vdivu_vx_u32m2(position, ow, vl), p->sh, vl), p->ph, vl);
            vuint32m2_t xx = __riscv_vsub_vx_u32m2(__riscv_vmul_vx_u32m2(
                __riscv_vremu_vx_u32m2(position, ow, vl), p->sw, vl), p->pw, vl);
            vuint32m2_t base = __riscv_vsll_vx_u32m2(__riscv_vadd_vv_u32m2(
                __riscv_vmul_vx_u32m2(yy, p->w, vl), xx, vl), 2, vl);
            for (size_t plane = 0; plane < (size_t)p->n*p->c; ++plane) {
                vfloat32m2_t best = __riscv_vfmv_v_f_f32m2(negative_inf.value, vl);
                for (size_t ky = 0; ky < p->kh; ++ky) {
                    vbool16_t valid_y = __riscv_vmsltu_vx_u32m2_b16(
                        __riscv_vadd_vx_u32m2(yy, ky, vl), p->h, vl);
                    for (size_t kx = 0; kx < p->kw; ++kx) {
                        vbool16_t valid = __riscv_vmand_mm_b16(valid_y,
                            __riscv_vmsltu_vx_u32m2_b16(__riscv_vadd_vx_u32m2(xx, kx, vl), p->w, vl), vl);
                        vuint32m2_t offsets = __riscv_vadd_vx_u32m2(base, (ky*p->w+kx)*4, vl);
                        vfloat32m2_t value = __riscv_vluxei32_v_f32m2_m(valid,
                            source+plane*in_area, offsets, vl);
                        value = __riscv_vmerge_vvm_f32m2(
                            __riscv_vfmv_v_f_f32m2(negative_inf.value, vl), value, valid, vl);
                        best = __riscv_vfmax_vv_f32m2(best, value, vl);
                    }
                }
                __riscv_vse32_v_f32m2(dest+plane*out_area+first, best, vl);
            }
            first += vl;
        }
        rvv_publish_buffer(output, (size_t)p->n*p->c*out_area*4);
        return 0;
    }
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
                            const float *at = source + plane * in_area + (size_t)iy * p->w + (size_t)ix;
                            vfloat32m1_t value = __riscv_vlse32_v_f32m1(at, (ptrdiff_t)(in_area * sizeof(float)), vl);
                            best = __riscv_vfmax_vv_f32m1(best, value, vl);
                        }
                    }
                    __riscv_vsse32_v_f32m1(dest + plane * out_area + y * ow + x,
                                         (ptrdiff_t)(out_area * sizeof(float)), best, vl);
                    channel += vl;
                }
            }
        }
    }
    rvv_publish_buffer(output, (size_t)p->n*p->c*out_area*4);
    return 0;
}

int nearest_fp32_rvv(const Shape4 *p, size_t sh, size_t sw, const float *input, float *output) {
    size_t oh, ow;
    if (!input || !output || basic_nearest_shape(p, sh, sw, &oh, &ow)) return -1;
    const float *source = rvv_read_buffer(input, (size_t)p->n*p->c*p->h*p->w*4);
    float *dest = rvv_write_buffer(output, (size_t)p->n*p->c*oh*ow*4);
    for (size_t plane = 0; plane < (size_t)p->n * p->c; ++plane) {
        for (size_t y = 0; y < p->h; ++y) {
            for (size_t x = 0; x < p->w;) {
                if (sw == 2) {
                    size_t n = __riscv_vsetvl_e32m1(p->w-x);
                    vfloat32m2_t original = __riscv_vle32_v_f32m2(
                        source+plane*p->h*p->w+y*p->w+x, n);
                    size_t vl = __riscv_vsetvl_e32m2(n*2);
                    vuint32m2_t indices = __riscv_vsrl_vx_u32m2(__riscv_vid_v_u32m2(vl), 1, vl);
                    vfloat32m2_t repeated = __riscv_vrgather_vv_f32m2(original, indices, vl);
                    for (size_t dy = 0; dy < sh; ++dy)
                        __riscv_vse32_v_f32m2(dest+plane*oh*ow+(y*sh+dy)*ow+x*2, repeated, vl);
                    x += n;
                    continue;
                }
                size_t vl = __riscv_vsetvl_e32m1(p->w - x);
                vfloat32m1_t value = __riscv_vle32_v_f32m1(source + plane * p->h * p->w + y * p->w + x, vl);
                for (size_t dy = 0; dy < sh; ++dy) {
                    float *row = dest + plane * oh * ow + (y * sh + dy) * ow + x * sw;
                    for (size_t dx = 0; dx < sw; ++dx)
                        __riscv_vsse32_v_f32m1(row + dx, (ptrdiff_t)(sw * sizeof(float)), value, vl);
                }
                x += vl;
            }
        }
    }
    rvv_publish_buffer(output, (size_t)p->n*p->c*oh*ow*4);
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
    const float *sources[parts];
    for (size_t part = 0; part < parts; ++part)
        sources[part] = rvv_read_buffer(inputs[part], outer*widths[part]*inner*4);
    float *dest = rvv_write_buffer(output, outer*axis*inner*4);
    for (size_t row = 0; row < outer; ++row) {
        size_t start = 0;
        for (size_t part = 0; part < parts; ++part) {
            size_t count = widths[part] * inner;
            if (count) copy(dest + row * axis * inner + start, sources[part] + row * count, count);
            start += count;
        }
    }
    rvv_publish_buffer(output, outer*axis*inner*4);
    return 0;
}

int slice_fp32_rvv(const float *input, float *output, size_t outer, size_t axis,
                   size_t inner, size_t start, size_t width) {
    if (basic_slice_shape(outer, axis, inner, start, width) || (width && (!input || !output))) return -1;
    if (width) {
        const float *source = rvv_read_buffer(input, outer*axis*inner*4);
        float *dest = rvv_write_buffer(output, outer*width*inner*4);
        for (size_t row = 0; row < outer; ++row)
            copy(dest + row * width * inner, source + (row * axis + start) * inner, width * inner);
        rvv_publish_buffer(output, outer*width*inner*4);
    }
    return 0;
}
