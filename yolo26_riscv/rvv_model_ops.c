#include "rvv_model_ops.h"
#include "rvv_access.h"
#include <stdint.h>
#include <riscv_vector.h>
#ifdef YOLO_N550_BOARD
#include "board/cache.h"
#include "board/memory.h"
#endif

/* Private coverage counter, not a new mailbox/profile ABI field. */
uint64_t rvv_depthwise3x3_calls;

static int ranges_overlap(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t aa = (uintptr_t)a, bb = (uintptr_t)b;
    if (na > UINTPTR_MAX - aa || nb > UINTPTR_MAX - bb) return 1;
    return aa < bb + nb && bb < aa + na;
}

int rvv_binary_periodic(const float *left, const float *right, float *output,
                        size_t count, size_t period, unsigned op) {
    if (op > 2 || !period || count > SIZE_MAX / 4 ||
        (count && (!left || !right || !output)) || period > count || count % period)
        return -1;
    if ((output != left && ranges_overlap(output, count*4, left, count*4)) ||
        (output != right && ranges_overlap(output, count*4, right, period*4)) ||
        (output == right && period != count)) return -1;
    const float *a = rvv_read_buffer(left, count*4);
    const float *b = rvv_read_buffer(right, period*4);
    float scalar = period == 1 ? right[0] : 0.0f;
    float *dest = rvv_write_buffer(output, count*4);
    if (period == 1) {
        for (size_t i = 0; i < count;) {
            size_t vl = __riscv_vsetvl_e32m2(count-i);
            vfloat32m2_t value = __riscv_vle32_v_f32m2(a+i, vl);
            if (op == 0) value = __riscv_vfadd_vf_f32m2(value, scalar, vl);
            else if (op == 1) value = __riscv_vfsub_vf_f32m2(value, scalar, vl);
            else value = __riscv_vfmul_vf_f32m2(value, scalar, vl);
            __riscv_vse32_v_f32m2(dest+i, value, vl);
            i += vl;
        }
        rvv_publish_buffer(output, count*4);
        return 0;
    }
    for (size_t base = 0; base < count; base += period) {
        for (size_t i = 0; i < period;) {
            size_t vl = __riscv_vsetvl_e32m2(period-i);
            vfloat32m2_t x = __riscv_vle32_v_f32m2(a+base+i, vl);
            vfloat32m2_t y = __riscv_vle32_v_f32m2(b+i, vl);
            vfloat32m2_t value;
            if (op == 0) value = __riscv_vfadd_vv_f32m2(x, y, vl);
            else if (op == 1) value = __riscv_vfsub_vv_f32m2(x, y, vl);
            else value = __riscv_vfmul_vv_f32m2(x, y, vl);
            __riscv_vse32_v_f32m2(dest+base+i, value, vl);
            i += vl;
        }
    }
    rvv_publish_buffer(output, count*4);
    return 0;
}

int rvv_bias_channels(float *output, const float *bias, size_t channels, size_t columns) {
    if (!output || !bias || !channels || !columns || channels > SIZE_MAX/columns ||
        channels*columns > SIZE_MAX/sizeof(float) ||
        ranges_overlap(output, channels*columns*sizeof(float), bias, channels*sizeof(float)))
        return -1;
    size_t bytes = channels*columns*sizeof(float);
    const float *source = rvv_read_buffer(output, bytes);
    (void)rvv_read_buffer(bias, channels*sizeof(float));
    float *dest = rvv_write_buffer(output, bytes);
    for (size_t channel = 0; channel < channels; ++channel) {
        float value = bias[channel];
        for (size_t column = 0; column < columns;) {
            size_t vl = __riscv_vsetvl_e32m2(columns-column);
            size_t at = channel*columns+column;
            vfloat32m2_t x = __riscv_vle32_v_f32m2(source+at, vl);
            __riscv_vse32_v_f32m2(dest+at, __riscv_vfadd_vf_f32m2(x, value, vl), vl);
            column += vl;
        }
    }
    rvv_publish_buffer(output, bytes);
    return 0;
}

void rvv_transpose(const float *input, float *output,
                   size_t batches, size_t rows, size_t columns) {
    size_t count = batches*rows*columns;
    const float *source = rvv_read_buffer(input, count*4);
    float *dest = rvv_write_buffer(output, count*4);
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t column = 0; column < columns; ++column) {
            for (size_t row = 0; row < rows;) {
                size_t vl = __riscv_vsetvl_e32m2(rows-row);
                vfloat32m2_t value = __riscv_vlse32_v_f32m2(
                    source+batch*rows*columns+row*columns+column,
                    (ptrdiff_t)(columns*4), vl);
                __riscv_vse32_v_f32m2(dest+batch*rows*columns+column*rows+row, value, vl);
                row += vl;
            }
        }
    }
    rvv_publish_buffer(output, count*4);
}

void rvv_max_rows(const float *input, float *values, int64_t *indices,
                  size_t rows, size_t width) {
    const float *source = rvv_read_buffer(input, rows*width*4);
    float *dest = rvv_write_buffer(values, rows*4);
    uint64_t *index_dest = indices ? rvv_write_buffer(indices, rows*8) : 0;
    for (size_t row = 0; row < rows;) {
        size_t vl = __riscv_vsetvl_e32m2(rows-row);
        vfloat32m2_t best = __riscv_vlse32_v_f32m2(source+row*width, (ptrdiff_t)(width*4), vl);
        vuint32m2_t selected = __riscv_vmv_v_x_u32m2(0, vl);
        for (size_t column = 1; column < width; ++column) {
            vfloat32m2_t value = __riscv_vlse32_v_f32m2(source+row*width+column, (ptrdiff_t)(width*4), vl);
            vbool16_t greater = __riscv_vmfgt_vv_f32m2_b16(value, best, vl);
            best = __riscv_vmerge_vvm_f32m2(best, value, greater, vl);
            selected = __riscv_vmerge_vxm_u32m2(selected, (uint32_t)column, greater, vl);
        }
        __riscv_vse32_v_f32m2(dest+row, best, vl);
        if (index_dest) __riscv_vse64_v_u64m4(index_dest+row, __riscv_vwaddu_vx_u64m4(selected, 0, vl), vl);
        row += vl;
    }
    rvv_publish_buffer(values, rows*4);
    if (indices) rvv_publish_buffer(indices, rows*8);
}

int rvv_gather_rows(const void *input, const int64_t *indices, void *output,
                     size_t outer, size_t axis, size_t inner,
                     size_t out_axis, size_t element_bytes) {
    if (!input || !indices || !output || !outer || !axis || !inner || !out_axis ||
        (element_bytes != 4 && element_bytes != 8)) return -1;
    size_t count = outer*out_axis*inner;
    /* Validate all selections before touching output; scalar index reads use
     * global DDR after the producer's synchronous publication. */
    for (size_t i = 0; i < count; ++i)
        if (indices[i] < 0 || (uint64_t)indices[i] >= axis) return -1;
    const unsigned char *source = rvv_read_buffer(input, outer*axis*inner*element_bytes);
    unsigned char *dest = rvv_write_buffer(output, count*element_bytes);
    for (size_t batch = 0; batch < outer; ++batch) {
        for (size_t row = 0; row < out_axis; ++row) {
            size_t first = (batch*out_axis+row)*inner;
            size_t selected = (size_t)indices[first], run = 1;
            while (run < inner && (size_t)indices[first+run] == selected) ++run;
            if (run == inner) {
                size_t bytes = inner*element_bytes;
                const unsigned char *from = source+(batch*axis+selected)*bytes;
                unsigned char *to = dest+first*element_bytes;
                for (size_t i = 0; i < bytes;) {
                    size_t vl = __riscv_vsetvl_e8m2(bytes-i);
                    __riscv_vse8_v_u8m2(to+i, __riscv_vle8_v_u8m2(from+i, vl), vl);
                    i += vl;
                }
            } else {
                for (size_t i = 0; i < inner; ++i) {
                    size_t at = (batch*axis+(size_t)indices[first+i])*inner+i;
                    if (element_bytes == 4) {
                        size_t vl = __riscv_vsetvl_e32m2(1);
                        __riscv_vse32_v_u32m2((uint32_t *)dest+first+i,
                            __riscv_vle32_v_u32m2((const uint32_t *)source+at, vl), vl);
                    } else {
                        size_t vl = __riscv_vsetvl_e64m2(1);
                        __riscv_vse64_v_u64m2((uint64_t *)dest+first+i,
                            __riscv_vle64_v_u64m2((const uint64_t *)source+at, vl), vl);
                    }
                }
            }
        }
    }
    rvv_publish_buffer(output, count*element_bytes);
    return 0;
}

void rvv_repeat_i64(const int64_t *input, int64_t *output,
                     size_t count, size_t repeats) {
    int64_t *dest = rvv_write_buffer(output, count*repeats*8);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < repeats;) {
            size_t vl = __riscv_vsetvl_e64m2(repeats-j);
            __riscv_vse64_v_i64m2(dest+i*repeats+j, __riscv_vmv_v_x_i64m2(input[i], vl), vl);
            j += vl;
        }
    }
    rvv_publish_buffer(output, count*repeats*8);
}

int rvv_depthwise3x3(const Conv2D *p, const float *input, const float *weight,
                     const float *bias, float *output) {
    size_t oh, ow;
    if (!input || !weight || !bias || !output || conv2d_output_shape(p, &oh, &ow) ||
        p->groups != p->in_channels || p->out_channels != p->in_channels ||
        p->kernel_h != 3 || p->kernel_w != 3 || p->stride_h != 1 || p->stride_w != 1 ||
        p->pad_h != 1 || p->pad_w != 1 || oh != p->in_h || ow != p->in_w) return -1;
    size_t area = oh * ow, bytes = (size_t)p->batch * p->in_channels * area * sizeof(float);
    if (ranges_overlap(input, bytes, output, bytes) ||
        ranges_overlap(weight, (size_t)p->in_channels * 9 * sizeof(float), output, bytes) ||
        ranges_overlap(bias, p->in_channels * sizeof(float), output, bytes)) return -1;
    const float *source = input;
    float *dest = output;
#ifdef YOLO_N550_BOARD
    int clp_output = board_is_clp(output);
    if ((board_is_clp(input) && bytes > BOARD_CLP_BYTES - ((uintptr_t)input - BOARD_CLP_DDR_BASE)) ||
        (clp_output && bytes > BOARD_CLP_BYTES - ((uintptr_t)output - BOARD_CLP_DDR_BASE))) return -1;
    if (board_is_clp(input)) board_clean(input, bytes);
    if (clp_output) board_prepare_write(output, bytes);
    source = (const float *)board_rvv_address(input);
    dest = (float *)board_rvv_address(output);
#endif
    /* m2 loads/stores intentionally bypass the per-vector m1 coherence
     * wrappers: output ownership covers this entire synchronous call.
     * Cached normal-DDR vector stores must never be invalidated afterward. */
    for (size_t batch = 0; batch < p->batch; ++batch) {
        for (size_t channel = 0; channel < p->in_channels; ++channel) {
            const float *plane = source + (batch * p->in_channels + channel) * area;
            const float *w = weight + channel * 9;
            float *out = dest + (batch * p->in_channels + channel) * area;
            for (size_t y = 0; y < oh; ++y) {
                for (size_t x = 0; x < ow;) {
                    /* Only edge pixels need horizontal bounds checks. Interior
                     * vectors stop before the right edge, never cross a row. */
                    int edge = x == 0 || x + 1 == ow;
                    size_t vl = __riscv_vsetvl_e32m2(edge ? 1 : ow - x - 1);
                    vfloat32m2_t sum = __riscv_vfmv_v_f_f32m2(0.0f, vl);
                    for (size_t ky = 0; ky < 3; ++ky) {
                        size_t iy = y + ky - 1;
                        for (size_t kx = 0; kx < 3; ++kx) {
                            size_t ix = x + kx - 1;
                            vfloat32m2_t value = __riscv_vfmv_v_f_f32m2(0.0f, vl);
                            if (iy < oh && (!edge || ix < ow))
                                value = __riscv_vle32_v_f32m2(plane + iy * ow + ix, vl);
                            /* Do not skip padded zeros: signed-zero and directed
                             * rounding must match the former nine-term path. */
                            vfloat32m2_t term = __riscv_vfmul_vf_f32m2(value, w[ky * 3 + kx], vl);
                            sum = __riscv_vfadd_vv_f32m2(sum, term, vl);
                        }
                    }
                    sum = __riscv_vfadd_vf_f32m2(sum, bias[channel], vl);
                    __riscv_vse32_v_f32m2(out + y * ow + x, sum, vl);
                    x += vl;
                }
            }
        }
    }
#ifdef YOLO_N550_BOARD
    if (clp_output) board_finish_write(output, bytes);
    else board_fence();
#endif
    ++rvv_depthwise3x3_calls;
    return 0;
}

/* Separate multiply/add keeps the scalar baseline's rounding contract.
 * FMA can be introduced as a separately measured numerical scheme later. */
void rvv_matmul_ordered(const float *a, const float *b, float *c,
                        size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc) {
    if (!m || !n) return;
    (void)rvv_read_buffer(a, k ? ((m-1)*lda+k)*4 : 0);
    const float *right_source = rvv_read_buffer(b, k ? ((k-1)*ldb+n)*4 : 0);
    size_t output_bytes = ((m-1)*ldc+n)*4;
    float *dest = rvv_write_buffer(c, output_bytes);
    for (size_t row = 0; row < m; row += 4) {
        for (size_t col = 0; col < n;) {
            size_t vl = __riscv_vsetvl_e32m2(n - col);
            vfloat32m2_t s0 = __riscv_vfmv_v_f_f32m2(0.0f, vl), s1 = s0, s2 = s0, s3 = s0;
            for (size_t inner = 0; inner < k; ++inner) {
                vfloat32m2_t right = __riscv_vle32_v_f32m2(right_source+inner*ldb+col, vl);
                s0 = __riscv_vfadd_vv_f32m2(s0, __riscv_vfmul_vf_f32m2(right, a[row*lda+inner], vl), vl);
                if (row+1 < m) s1 = __riscv_vfadd_vv_f32m2(s1, __riscv_vfmul_vf_f32m2(right, a[(row+1)*lda+inner], vl), vl);
                if (row+2 < m) s2 = __riscv_vfadd_vv_f32m2(s2, __riscv_vfmul_vf_f32m2(right, a[(row+2)*lda+inner], vl), vl);
                if (row+3 < m) s3 = __riscv_vfadd_vv_f32m2(s3, __riscv_vfmul_vf_f32m2(right, a[(row+3)*lda+inner], vl), vl);
            }
            __riscv_vse32_v_f32m2(dest+row*ldc+col, s0, vl);
            if (row+1 < m) __riscv_vse32_v_f32m2(dest+(row+1)*ldc+col, s1, vl);
            if (row+2 < m) __riscv_vse32_v_f32m2(dest+(row+2)*ldc+col, s2, vl);
            if (row+3 < m) __riscv_vse32_v_f32m2(dest+(row+3)*ldc+col, s3, vl);
            col += vl;
        }
    }
    rvv_publish_buffer(c, output_bytes);
}

void rvv_copy_bytes(void *output, const void *input, size_t bytes) {
    if (!bytes || output == input) return;
    const unsigned char *source = rvv_read_buffer(input, bytes);
    unsigned char *dest = rvv_write_buffer(output, bytes);
    for (size_t i = 0; i < bytes;) {
        size_t vl = __riscv_vsetvl_e8m2(bytes - i);
        __riscv_vse8_v_u8m2(dest + i, __riscv_vle8_v_u8m2(source + i, vl), vl);
        i += vl;
    }
    rvv_publish_buffer(output, bytes);
}

static vfloat32m2_t exp_neg(vfloat32m2_t x, size_t vl) {
    vbool16_t tiny = __riscv_vmfle_vf_f32m2_b16(x, -80.0f, vl);
    x = __riscv_vfmax_vf_f32m2(x, -80.0f, vl);
    vfloat32m2_t nf = __riscv_vfsub_vf_f32m2(__riscv_vfmul_vf_f32m2(x, 1.4426950408889634f, vl), 0.5f, vl);
    vint32m2_t n = __riscv_vfcvt_rtz_x_f_v_i32m2(nf, vl);
    nf = __riscv_vfcvt_f_x_v_f32m2(n, vl);
    vfloat32m2_t r = __riscv_vfsub_vv_f32m2(x, __riscv_vfmul_vf_f32m2(nf, 0.693145751953125f, vl), vl);
    r = __riscv_vfsub_vv_f32m2(r, __riscv_vfmul_vf_f32m2(nf, 0.000001428606765330187f, vl), vl);
    const float coefficients[] = {1.0f / 120.0f, 1.0f / 24.0f, 1.0f / 6.0f, 0.5f, 1.0f, 1.0f};
    vfloat32m2_t p = __riscv_vfmv_v_f_f32m2(1.0f / 720.0f, vl);
    for (size_t i = 0; i < sizeof(coefficients) / sizeof(coefficients[0]); ++i)
        p = __riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(r, p, vl), coefficients[i], vl);
    vuint32m2_t bits = __riscv_vreinterpret_v_i32m2_u32m2(__riscv_vadd_vx_i32m2(n, 127, vl));
    bits = __riscv_vsll_vx_u32m2(bits, 23, vl);
    p = __riscv_vfmul_vv_f32m2(p, __riscv_vreinterpret_v_u32m2_f32m2(bits), vl);
    return __riscv_vfmerge_vfm_f32m2(p, 0.0f, tiny, vl);
}

void rvv_nonlinear(const float *input, float *output, size_t count, int silu) {
    const float *source = rvv_read_buffer(input, count*4);
    float *dest = rvv_write_buffer(output, count*4);
    for (size_t i = 0; i < count;) {
        size_t vl = __riscv_vsetvl_e32m2(count - i);
        vfloat32m2_t x = __riscv_vle32_v_f32m2(source + i, vl);
        vfloat32m2_t e = exp_neg(__riscv_vfneg_v_f32m2(__riscv_vfabs_v_f32m2(x, vl), vl), vl);
        vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(e, 1.0f, vl);
        vbool16_t positive = __riscv_vmfge_vf_f32m2_b16(x, 0.0f, vl);
        vfloat32m2_t numerator;
        if (silu) numerator = __riscv_vmerge_vvm_f32m2(__riscv_vfmul_vv_f32m2(x, e, vl), x, positive, vl);
        else numerator = __riscv_vfmerge_vfm_f32m2(e, 1.0f, positive, vl);
        __riscv_vse32_v_f32m2(dest + i, __riscv_vfdiv_vv_f32m2(numerator, denominator, vl), vl);
        i += vl;
    }
    rvv_publish_buffer(output, count*4);
}

void rvv_softmax(const float *input, float *output, size_t rows, size_t width) {
    const float *source = rvv_read_buffer(input, rows*width*4);
    for (size_t row = 0; row < rows; ++row) {
        const float *scalar_source = input+row*width;
        float *scalar_dest = output+row*width;
        float maximum = scalar_source[0], sum = 0.0f;
        for (size_t i = 1; i < width; ++i) if (scalar_source[i] > maximum) maximum = scalar_source[i];
        float *dest = rvv_write_buffer(scalar_dest, width*4);
        for (size_t i = 0; i < width;) {
            size_t vl = __riscv_vsetvl_e32m2(width - i);
            vfloat32m2_t x = __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(source+row*width+i, vl), maximum, vl);
            __riscv_vse32_v_f32m2(dest + i, exp_neg(x, vl), vl);
            i += vl;
        }
        rvv_publish_buffer(scalar_dest, width*4);
        /* Ordered scalar reduction deliberately retains the baseline contract. */
        for (size_t i = 0; i < width; ++i) sum += scalar_dest[i];
        dest = rvv_write_buffer(scalar_dest, width*4);
        for (size_t i = 0; i < width;) {
            size_t vl = __riscv_vsetvl_e32m2(width - i);
            __riscv_vse32_v_f32m2(dest + i, __riscv_vfdiv_vf_f32m2(__riscv_vle32_v_f32m2(dest + i, vl), sum, vl), vl);
            i += vl;
        }
        rvv_publish_buffer(scalar_dest, width*4);
    }
}
