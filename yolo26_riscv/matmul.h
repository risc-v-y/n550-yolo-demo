#ifndef YOLO26_MATMUL_H_
#define YOLO26_MATMUL_H_

#include <stddef.h>

/* Row-major C[M,N] = A[M,K] * B[K,N]. Leading dimensions are in floats.
 * Buffers must not overlap. lda >= K, ldb >= N, ldc >= N.
 * M=0 or N=0 writes nothing; K=0 writes zeros in the active output region.
 * This first implementation accepts finite FP32 inputs and overwrites C.
 */
void matmul_fp32_scalar(const float *a, const float *b, float *c,
                       size_t m, size_t n, size_t k,
                       size_t lda, size_t ldb, size_t ldc);
void matmul_fp32_rvv(const float *a, const float *b, float *c,
                    size_t m, size_t n, size_t k,
                    size_t lda, size_t ldb, size_t ldc);

#endif
