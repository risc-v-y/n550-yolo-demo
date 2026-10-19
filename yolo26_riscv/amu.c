#include "amu.h"
#include "model_profile.h"
#ifndef AMU_PACK_ONLY
#include <xewmatrix_intrinsic.h>
#ifdef YOLO_N550_BOARD
#include "board/cache.h"
#include "board/memory.h"
#include "board/diagnostics.h"
#define DIAG_PHASE(p) board_diag_phase(p)
#else
#define DIAG_PHASE(p) ((void)0)
#endif

enum { TILE_M = 32, TILE_N = 32, TILE_K = 32 };
#ifdef YOLO_N550_BOARD
static uint16_t left_tile[TILE_M * TILE_K] BOARD_CLP_DATA;
static uint16_t right_tile[TILE_N * TILE_K] BOARD_CLP_DATA;
#else
static uint16_t left_tile[TILE_M * TILE_K] __attribute__((aligned(64)));
static uint16_t right_tile[TILE_N * TILE_K] __attribute__((aligned(64)));
#endif
uint64_t amu_instruction_count;
uint64_t amu_runtime_weight_conversions;
#endif

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

size_t amu_packed_weight_bytes(size_t m, size_t k) {
    if(!m || !k) return 0;
    size_t mt=m/32+(m%32!=0), kt=k/32+(k%32!=0);
    if(mt>SIZE_MAX/kt || mt*kt>SIZE_MAX/2048) return 0;
    return mt*kt*2048;
}
int amu_pack_weights(const float *a,size_t m,size_t k,size_t lda,
                     uint16_t *storage,size_t capacity,AmuPackedWeights *packed) {
    if(!packed) return -1;
    *packed=(AmuPackedWeights){0};
    size_t bytes=amu_packed_weight_bytes(m,k);
    if(!a || !storage || !bytes || lda<k || capacity<bytes || (uintptr_t)storage%64 ||
       m>SIZE_MAX/lda/sizeof(float)) return -1;
    for(size_t i=0;i<bytes/2;++i) storage[i]=0;
    size_t kt=k/32+(k%32!=0);
    for(size_t row=0;row<m;row+=32) {
        size_t rows=m-row<32 ? m-row : 32;
        for(size_t inner=0;inner<k;inner+=32) {
            size_t count=k-inner<32 ? k-inner : 32;
            uint16_t *tile=storage+((row/32)*kt+inner/32)*1024;
            for(size_t i=0;i<rows;++i) for(size_t j=0;j<count;++j)
                if(amu_half(a[(row+i)*lda+inner+j],tile+i*32+j)) return -1;
        }
    }
#ifdef YOLO_N550_BOARD
    board_clean(storage,bytes);
#endif
    *packed=(AmuPackedWeights){storage,m,k,bytes};
    return 0;
}

#ifndef AMU_PACK_ONLY

/* HSC: matrix CSR reads drain the AMU ROB. Compiler barriers are needed:
 * the vendor load/store macros themselves do not declare memory clobbers.
 * Platform cache maintenance will be supplied by the platform adapter. */
static void complete(void) {
    DIAG_PHASE(PHASE_AMU_WAIT);
    uint64_t start = model_ticks();
    uintptr_t flags;
    __asm__ volatile ("csrr %0, xmfflags" : "=r"(flags) : : "memory");
    (void)flags;
#ifdef YOLO_N550_BOARD
    board_diag.amu_flags=flags;
    board_clean((const void *)&board_diag.amu_flags,sizeof(board_diag.amu_flags));
#endif
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
    amu_runtime_weight_conversions = 0;
    return 0;
}

static int matmul_fp16(const float *a,const AmuPackedWeights *packed,const float *b,float *c,
                       size_t m,size_t n,size_t k,size_t lda,size_t ldb,size_t ldc) {
    if ((!a && !packed) || !b || !c || lda < k || ldb < n || ldc < n) return -1;
    for (size_t row = 0; row < m; row += TILE_M) {
        size_t rows = m - row < TILE_M ? m - row : TILE_M;
        for (size_t col = 0; col < n; col += TILE_N) {
            size_t cols = n - col < TILE_N ? n - col : TILE_N;
            uint64_t start = model_ticks();
            uintptr_t old;
            DIAG_PHASE(PHASE_AMU_SUBMIT);
            __riscv_msettilem(old, rows, w);
            __riscv_msettilen(old, cols, w);
            __riscv_mzero("acc0");
            model_profile_end(PROFILE_MATRIX_SUBMIT, start);
            for (size_t inner = 0; inner < k; inner += TILE_K) {
                DIAG_PHASE(PHASE_AMU_PACK);
                size_t count = k - inner < TILE_K ? k - inner : TILE_K;
                const uint16_t *left=left_tile;
                if(packed) {
                    size_t kt=k/TILE_K+(k%TILE_K!=0);
                    left=packed->data+((row/TILE_M)*kt+inner/TILE_K)*TILE_M*TILE_K;
                } else {
                    start = model_ticks();
                    for (size_t i = 0; i < rows; ++i)
                        for (size_t j = 0; j < count; ++j)
                            if (amu_half(a[(row + i) * lda + inner + j], left_tile + i * TILE_K + j)) return -1;
                    amu_runtime_weight_conversions+=rows*count;
                    model_profile_end(PROFILE_WEIGHT_PREPARE, start);
                }
                start = model_ticks();
                /* TR-B contains N rows of K elements, packed from row-major B[K,N]. */
                for (size_t i = 0; i < cols; ++i)
                    for (size_t j = 0; j < count; ++j)
                        if (amu_half(b[(inner + j) * ldb + col + i], right_tile + i * TILE_K + j)) return -1;
                model_profile_end(PROFILE_INPUT_PACK, start);
                start = model_ticks();
                __asm__ volatile ("fence rw, rw" : : : "memory");
#ifdef YOLO_N550_BOARD
                if(!packed) board_clean(left_tile, sizeof(left_tile));
                board_clean(right_tile, sizeof(right_tile));
#endif
                DIAG_PHASE(PHASE_AMU_SUBMIT);
                __riscv_msettilek(old, count, w);
                __riscv_mlae16("tr0", left, TILE_K * sizeof(uint16_t));
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
            DIAG_PHASE(PHASE_AMU_STORE);
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
int amu_matmul_fp16(const float *a,const float *b,float *c,
                    size_t m,size_t n,size_t k,size_t lda,size_t ldb,size_t ldc) {
    return matmul_fp16(a,0,b,c,m,n,k,lda,ldb,ldc);
}
int amu_matmul_prepacked_fp16(const AmuPackedWeights *packed,const float *b,float *c,
                             size_t n,size_t ldb,size_t ldc) {
    if(!packed || !packed->data || (uintptr_t)packed->data%64 ||
       !amu_packed_weight_bytes(packed->m,packed->k) ||
       packed->bytes!=amu_packed_weight_bytes(packed->m,packed->k)) return -1;
    return matmul_fp16(0,packed,b,c,packed->m,n,packed->k,packed->k,ldb,ldc);
}
#endif
