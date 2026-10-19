#ifndef YOLO26_AMU_H_
#define YOLO26_AMU_H_
#include <stddef.h>
#include <stdint.h>

/* Independently implemented FP32 -> IEEE binary16, round-to-nearest-even.
 * Reject non-finite/overflowing inputs instead of silently changing precision. */
int amu_half(float value, uint16_t *bits);
/* Immutable [ceil(M/32),ceil(K/32),32,32] tiles, zero-padded, 64-byte aligned.
 * pack publishes CPU writes before any AMU load on the board. Source and
 * storage must not overlap. Descriptor is valid only after successful pack. */
typedef struct {
    const uint16_t *data;
    size_t m, k, bytes;
} AmuPackedWeights;
size_t amu_packed_weight_bytes(size_t m, size_t k);
int amu_pack_weights(const float *a, size_t m, size_t k, size_t lda,
                     uint16_t *storage, size_t capacity, AmuPackedWeights *packed);
int amu_init(void);
int amu_matmul_fp16(const float *a, const float *b, float *c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
int amu_matmul_prepacked_fp16(const AmuPackedWeights *packed, const float *b, float *c,
                             size_t n, size_t ldb, size_t ldc);
extern uint64_t amu_instruction_count;
extern uint64_t amu_runtime_weight_conversions;
#endif
