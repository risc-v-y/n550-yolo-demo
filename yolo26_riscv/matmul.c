#include "matmul.h"

#include <riscv_vector.h>

/* Keep a scalar implementation independent of the vector indexing. */
__attribute__((noinline))
void matmul_fp32_scalar(const float *a, const float *b, float *c,
                       size_t m, size_t n, size_t k,
                       size_t lda, size_t ldb, size_t ldc) {
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (size_t inner = 0; inner < k; ++inner) {
                sum += a[row * lda + inner] * b[inner * ldb + col];
            }
            c[row * ldc + col] = sum;
        }
    }
}

/* Vectorize adjacent output columns. Runtime VL handles every N tail and
 * works with different hardware vector lengths without recompilation.
 * The explicit FMA may round differently from the scalar multiply/add.
 */
__attribute__((noinline))
void matmul_fp32_rvv(const float *a, const float *b, float *c,
                    size_t m, size_t n, size_t k,
                    size_t lda, size_t ldb, size_t ldc) {
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n;) {
            size_t vl = __riscv_vsetvl_e32m1(n - col);
            vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (size_t inner = 0; inner < k; ++inner) {
                vfloat32m1_t right = __riscv_vle32_v_f32m1(b + inner * ldb + col, vl);
                sum = __riscv_vfmacc_vf_f32m1(sum, a[row * lda + inner], right, vl);
            }
            __riscv_vse32_v_f32m1(c + row * ldc + col, sum, vl);
            col += vl;
        }
    }
}
