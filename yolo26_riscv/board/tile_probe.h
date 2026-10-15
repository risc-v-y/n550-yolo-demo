#ifndef YOLO_TILE_PROBE_H
#define YOLO_TILE_PROBE_H
#include <stddef.h>
#include <stdint.h>
/* Diagnostic-only wire format, little endian. No normal inference ABI change. */
typedef struct {
    uint32_t meta[16];
    uint16_t left[1024], right[1024];
    float before[512], after[512], bias[16];
} TileRecord;
typedef struct { uint32_t header[16]; TileRecord records[3]; } TileProbe;
extern TileProbe board_tile_probe;
void tile_probe_begin(const void *input, size_t bytes, const void *weights, size_t weight_bytes);
int tile_probe_select(size_t first);
int tile_probe_amu(const uint16_t *left, const uint16_t *right, const float *output,
                   size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
void tile_probe_bias(const float *output, const float *bias, size_t stride);
int tile_probe_finish(void);
#endif
