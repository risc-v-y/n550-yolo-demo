#ifndef YOLO26_AMU_H_
#define YOLO26_AMU_H_
#ifndef YOLO_FAST
#define YOLO_FAST 0
#endif
#include <stddef.h>
#include <stdint.h>
#include "conv2d.h"

/* Independently implemented FP32 -> IEEE binary16, round-to-nearest-even.
 * Reject non-finite/overflowing inputs instead of silently changing precision. */
int amu_half(float value, uint16_t *bits);
/* Configuration is changed only while idle, then weights must be repacked.
 * K always advances by 32: tiling never changes the accumulation order. */
enum { AMU_MAX_ROWS=128 };
extern size_t amu_tile_m, amu_tile_n, amu_capacity_rows;
int amu_configure(size_t m, size_t n);
size_t amu_packed_input_bytes(size_t n, size_t k);
/* Immutable [ceil(M/tile_m),ceil(K/32),tile_m,32] tiles, zero-padded, 64-byte aligned.
 * pack publishes CPU writes before any AMU load on the board. Source and
 * storage must not overlap. Descriptor is valid only after successful pack. */
typedef struct {
    const uint16_t *data;
    size_t m, k, bytes;
} AmuPackedWeights;
size_t amu_packed_weight_bytes(size_t m, size_t k);
int amu_pack_weights(const float *a, size_t m, size_t k, size_t lda,
                     uint16_t *storage, size_t capacity, AmuPackedWeights *packed);
/* One spatial block, <=128 positions: [ceil(K/32),round_up(N,32),32 K].
 * Source is NCHW FP32, groups=1. Storage is scalar-written then cleaned;
 * all output-channel blocks share it until the synchronous multiply returns.
 * Source/storage must not overlap. Failure leaves the descriptor invalid. */
enum { AMU_INPUT_MAX_K=4096 };
typedef struct {
    const uint16_t *data;
    size_t k, n, bytes;
} AmuPackedInput;
int amu_pack_conv_input(const Conv2D *p, const float *input, size_t batch,
                        size_t first, size_t columns, uint16_t *storage,
                        size_t capacity, AmuPackedInput *packed);
int amu_conv_input_shape(const Conv2D *p, size_t *k, size_t *oh, size_t *ow);
/* Alternate physical input; logical graph tensors retain FP32 metadata.
 * The UINT8 binding is frame-scoped; prepare is called once per convolution. */
extern unsigned amu_input_flags;
enum { AMU_INPUT_PRECONVERT=1, AMU_INPUT_SPATIAL=2, AMU_INPUT_TILED=4,
       AMU_INPUT_U8_REUSE=8, AMU_INPUT_OWNED=16, AMU_INPUT_MASK=31 };
void amu_input_bind_u8(const float *logical, const uint8_t *physical);
int amu_input_prepare(const Conv2D *p,const float *input,uint16_t *half,size_t count);
/* Dedicated, disjoint CLP workspaces only; synchronous AMU consumes each block
 * before its next write. Acquire/release surround one complete convolution.
 * Other pointers, diagnostic probes and scalar fallbacks keep per-call CBO. */
int amu_input_acquire(uint16_t *scratch,size_t scratch_bytes,uint16_t *half,size_t half_bytes);
void amu_input_release(void);
/* Same bits/layout as scalar packing. On the board the caller must publish
 * the input before CLP reads (the graph boundary already does so). Output
 * cache ownership is transferred once per block; no FP16 ISA is required. */
int amu_pack_conv_input_rvv(const Conv2D *p, const float *input, size_t batch,
                            size_t first, size_t columns, uint16_t *storage,
                            size_t capacity, AmuPackedInput *packed);
int amu_init(void);
int amu_matmul_fp16(const float *a, const float *b, float *c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
int amu_matmul_prepacked_fp16(const AmuPackedWeights *packed, const float *b, float *c,
                             size_t n, size_t ldb, size_t ldc);
int amu_matmul_packed_fp16(const AmuPackedWeights *weight, const AmuPackedInput *input,
                          float *c, size_t ldc);
extern uint64_t amu_instruction_count;
extern uint64_t amu_runtime_weight_conversions;
#endif
