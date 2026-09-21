#include "amu.h"
#include "model_profile.h"
#include <xewmatrix_intrinsic.h>
#ifdef YOLO_N550_BOARD
#include "board/cache.h"
#endif

enum { TILE_M = 32, TILE_N = 32, TILE_K = 32 };
static uint16_t left_tile[TILE_M * TILE_K] __attribute__((aligned(64)));
static uint16_t right_tile[TILE_N * TILE_K] __attribute__((aligned(64)));
uint64_t amu_instruction_count;

int amu_half(float value, uint16_t *bits) {
    union { float f; uint32_t u; } x = {.f = value};
    uint32_t sign = (x.u >> 16) & 0x8000u;
    uint32_t exponent = (x.u >> 23) & 255u, fraction = x.u & 0x7fffffu;
    if (exponent == 255u) return -1;
    if (exponent < 102u) { *bits = (uint16_t)sign; return 0; }
    if (exponent < 113u) {
        uint32_t mantissa = fraction | 0x800000u;
        unsigned shift = 126u - exponent;
        uint32_t half = mantissa >> shift, residue = mantissa & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1u);
        half += residue > halfway || (residue == halfway && (half & 1u));
        *bits = (uint16_t)(sign | half);
        return 0;
    }
    if (exponent > 142u) return -1;
    uint32_t half = ((exponent - 112u) << 10) | (fraction >> 13);
    uint32_t residue = fraction & 8191u;
    half += residue > 4096u || (residue == 4096u && (half & 1u));
    if (half >= 0x7c00u) return -1;
    *bits = (uint16_t)(sign | half);
    return 0;
}

/* HSC: matrix CSR reads drain the AMU ROB. Compiler barriers are needed:
 * the vendor load/store macros themselves do not declare memory clobbers.
 * Platform cache maintenance will be supplied by the platform adapter. */
static void complete(void) {
    uint64_t start = model_ticks();
    uintptr_t flags;
    __asm__ volatile ("csrr %0, xmfflags" : "=r"(flags) : : "memory");
    (void)flags;
    model_profile_end(PROFILE_MATRIX_WAIT, start);
}

int amu_init(void) {
    uintptr_t capabilities, row_bytes, tile_bytes;
    __asm__ volatile ("csrs mstatus, %0" : : "r"((uintptr_t)0x60000000u) : "memory");
    __asm__ volatile ("csrw xmcsr, zero" : : : "memory");
    __asm__ volatile ("csrr %0, xmisa" : "=r"(capabilities));
    __asm__ volatile ("csrr %0, xtrlenb" : "=r"(row_bytes));
    __asm__ volatile ("csrr %0, xtlenb" : "=r"(tile_bytes));
    if (!(capabilities & 0x80u) || row_bytes < TILE_K * 2 || tile_bytes / row_bytes < TILE_M) return -1;
    amu_instruction_count = 0;
    return 0;
}

int amu_matmul_fp16(const float *a, const float *b, float *c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc) {
    if (!a || !b || !c || lda < k || ldb < n || ldc < n) return -1;
    for (size_t row = 0; row < m; row += TILE_M) {
        size_t rows = m - row < TILE_M ? m - row : TILE_M;
        for (size_t col = 0; col < n; col += TILE_N) {
            size_t cols = n - col < TILE_N ? n - col : TILE_N;
            uint64_t start = model_ticks();
            uintptr_t old;
            __riscv_msettilem(old, rows, w);
            __riscv_msettilen(old, cols, w);
            __riscv_mzero("acc0");
            model_profile_end(PROFILE_MATRIX_SUBMIT, start);
            for (size_t inner = 0; inner < k; inner += TILE_K) {
                size_t count = k - inner < TILE_K ? k - inner : TILE_K;
                start = model_ticks();
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < count; ++j)
                        if (amu_half(a[(row + i) * lda + inner + j], left_tile + i * TILE_K + j)) return -1;
                model_profile_end(PROFILE_WEIGHT_PREPARE, start);
                start = model_ticks();
                /* TR-B contains N rows of K elements, packed from row-major B[K,N]. */
                for (size_t i = 0; i < cols; ++i)
                    for (size_t j = 0; j < count; ++j)
                        if (amu_half(b[(inner + j) * ldb + col + i], right_tile + i * TILE_K + j)) return -1;
                model_profile_end(PROFILE_INPUT_PACK, start);
                start = model_ticks();
                __asm__ volatile ("fence rw, rw" : : : "memory");
#ifdef YOLO_N550_BOARD
                board_clean(left_tile, sizeof(left_tile));
                board_clean(right_tile, sizeof(right_tile));
#endif
                __riscv_msettilek(old, count, w);
                __riscv_mlae16("tr0", left_tile, TILE_K * sizeof(uint16_t));
                __riscv_mlbe16("tr1", right_tile, TILE_K * sizeof(uint16_t));
                __riscv_mfmacc_s_h("acc0", "tr1", "tr0");
                model_profile_end(PROFILE_MATRIX_SUBMIT, start);
                complete();
                ++amu_instruction_count;
            }
            start = model_ticks();
#ifdef YOLO_N550_BOARD
            for (size_t i = 0; i < rows; ++i)
                board_prepare_write(c + (row+i)*ldc + col, cols*sizeof(float));
#endif
            __riscv_msce32("acc0", c + row * ldc + col, ldc * sizeof(float));
            model_profile_end(PROFILE_MATRIX_SUBMIT, start);
            complete();
            __asm__ volatile ("fence rw, rw" : : : "memory");
#ifdef YOLO_N550_BOARD
            for (size_t i = 0; i < rows; ++i)
                board_finish_write(c + (row+i)*ldc + col, cols*sizeof(float));
#endif
            (void)old;
        }
    }
    return 0;
}
