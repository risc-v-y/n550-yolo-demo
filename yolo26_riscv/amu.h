#ifndef YOLO26_AMU_H_
#define YOLO26_AMU_H_
#include <stddef.h>
#include <stdint.h>

/* Independently implemented FP32 -> IEEE binary16, round-to-nearest-even.
 * Reject non-finite/overflowing inputs instead of silently changing precision. */
int amu_half(float value, uint16_t *bits);
int amu_init(void);
int amu_matmul_fp16(const float *a, const float *b, float *c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
extern uint64_t amu_instruction_count;
#endif
