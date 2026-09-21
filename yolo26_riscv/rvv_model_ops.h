#ifndef YOLO26_RVV_MODEL_OPS_H_
#define YOLO26_RVV_MODEL_OPS_H_
#include <stddef.h>
void rvv_matmul_ordered(const float *a, const float *b, float *c,
                        size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
void rvv_copy_bytes(void *output, const void *input, size_t bytes);
void rvv_nonlinear(const float *input, float *output, size_t count, int silu);
void rvv_softmax(const float *input, float *output, size_t rows, size_t width);
#endif
