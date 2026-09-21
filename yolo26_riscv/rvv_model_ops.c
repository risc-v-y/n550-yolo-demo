#include "rvv_model_ops.h"
#include <stdint.h>
#include <riscv_vector.h>

/* Separate multiply/add keeps the scalar baseline's rounding contract.
 * FMA can be introduced as a separately measured numerical scheme later. */
void rvv_matmul_ordered(const float *a, const float *b, float *c,
                        size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc) {
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n;) {
            size_t vl = __riscv_vsetvl_e32m1(n - col);
            vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (size_t inner = 0; inner < k; ++inner) {
                vfloat32m1_t right = __riscv_vle32_v_f32m1(b + inner * ldb + col, vl);
                vfloat32m1_t term = __riscv_vfmul_vf_f32m1(right, a[row * lda + inner], vl);
                sum = __riscv_vfadd_vv_f32m1(sum, term, vl);
            }
            __riscv_vse32_v_f32m1(c + row * ldc + col, sum, vl);
            col += vl;
        }
    }
}

void rvv_copy_bytes(void *output, const void *input, size_t bytes) {
    unsigned char *dest = output;
    const unsigned char *source = input;
    for (size_t i = 0; i < bytes;) {
        size_t vl = __riscv_vsetvl_e8m1(bytes - i);
        __riscv_vse8_v_u8m1(dest + i, __riscv_vle8_v_u8m1(source + i, vl), vl);
        i += vl;
    }
}

static vfloat32m1_t exp_neg(vfloat32m1_t x, size_t vl) {
    vbool32_t tiny = __riscv_vmfle_vf_f32m1_b32(x, -80.0f, vl);
    x = __riscv_vfmax_vf_f32m1(x, -80.0f, vl);
    vfloat32m1_t nf = __riscv_vfsub_vf_f32m1(__riscv_vfmul_vf_f32m1(x, 1.4426950408889634f, vl), 0.5f, vl);
    vint32m1_t n = __riscv_vfcvt_rtz_x_f_v_i32m1(nf, vl);
    nf = __riscv_vfcvt_f_x_v_f32m1(n, vl);
    vfloat32m1_t r = __riscv_vfsub_vv_f32m1(x, __riscv_vfmul_vf_f32m1(nf, 0.693145751953125f, vl), vl);
    r = __riscv_vfsub_vv_f32m1(r, __riscv_vfmul_vf_f32m1(nf, 0.000001428606765330187f, vl), vl);
    const float coefficients[] = {1.0f / 120.0f, 1.0f / 24.0f, 1.0f / 6.0f, 0.5f, 1.0f, 1.0f};
    vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(1.0f / 720.0f, vl);
    for (size_t i = 0; i < sizeof(coefficients) / sizeof(coefficients[0]); ++i)
        p = __riscv_vfadd_vf_f32m1(__riscv_vfmul_vv_f32m1(r, p, vl), coefficients[i], vl);
    vuint32m1_t bits = __riscv_vreinterpret_v_i32m1_u32m1(__riscv_vadd_vx_i32m1(n, 127, vl));
    bits = __riscv_vsll_vx_u32m1(bits, 23, vl);
    p = __riscv_vfmul_vv_f32m1(p, __riscv_vreinterpret_v_u32m1_f32m1(bits), vl);
    return __riscv_vfmerge_vfm_f32m1(p, 0.0f, tiny, vl);
}

void rvv_nonlinear(const float *input, float *output, size_t count, int silu) {
    for (size_t i = 0; i < count;) {
        size_t vl = __riscv_vsetvl_e32m1(count - i);
        vfloat32m1_t x = __riscv_vle32_v_f32m1(input + i, vl);
        vfloat32m1_t e = exp_neg(__riscv_vfneg_v_f32m1(__riscv_vfabs_v_f32m1(x, vl), vl), vl);
        vfloat32m1_t denominator = __riscv_vfadd_vf_f32m1(e, 1.0f, vl);
        vbool32_t positive = __riscv_vmfge_vf_f32m1_b32(x, 0.0f, vl);
        vfloat32m1_t numerator;
        if (silu) numerator = __riscv_vmerge_vvm_f32m1(__riscv_vfmul_vv_f32m1(x, e, vl), x, positive, vl);
        else numerator = __riscv_vfmerge_vfm_f32m1(e, 1.0f, positive, vl);
        __riscv_vse32_v_f32m1(output + i, __riscv_vfdiv_vv_f32m1(numerator, denominator, vl), vl);
        i += vl;
    }
}

void rvv_softmax(const float *input, float *output, size_t rows, size_t width) {
    for (size_t row = 0; row < rows; ++row) {
        const float *source = input + row * width;
        float *dest = output + row * width;
        float maximum = source[0], sum = 0.0f;
        for (size_t i = 1; i < width; ++i) if (source[i] > maximum) maximum = source[i];
        for (size_t i = 0; i < width;) {
            size_t vl = __riscv_vsetvl_e32m1(width - i);
            vfloat32m1_t x = __riscv_vfsub_vf_f32m1(__riscv_vle32_v_f32m1(source + i, vl), maximum, vl);
            __riscv_vse32_v_f32m1(dest + i, exp_neg(x, vl), vl);
            i += vl;
        }
        /* Ordered scalar reduction deliberately retains the baseline contract. */
        for (size_t i = 0; i < width; ++i) sum += dest[i];
        for (size_t i = 0; i < width;) {
            size_t vl = __riscv_vsetvl_e32m1(width - i);
            __riscv_vse32_v_f32m1(dest + i, __riscv_vfdiv_vf_f32m1(__riscv_vle32_v_f32m1(dest + i, vl), sum, vl), vl);
            i += vl;
        }
    }
}
