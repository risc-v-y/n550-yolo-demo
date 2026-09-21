#ifndef YOLO26_MODEL_PROFILE_H_
#define YOLO26_MODEL_PROFILE_H_

#include <stdint.h>
#ifndef MODEL_PROFILE
#define MODEL_PROFILE 0
#endif

/* Disjoint internal stages; their sum is a subset of graph time.
 * rdcycle is an emulator counter under QEMU, NOT FPGA execution cycles.
 * Timer reads and accounting add overhead. No conversion to FPS is valid. */
enum {
    PROFILE_WEIGHT_PREPARE,
    PROFILE_INPUT_IM2COL,
    PROFILE_INPUT_PACK,
    PROFILE_MATRIX_SUBMIT,
    PROFILE_MATRIX_WAIT,
    PROFILE_BIAS_DEQUANT,
    PROFILE_PHASE_COUNT
};
typedef struct { uint64_t ticks, calls; } ModelProfilePhase;
#if MODEL_PROFILE
extern ModelProfilePhase model_profile_phases[PROFILE_PHASE_COUNT];
static inline uint64_t model_ticks(void) {
    uint64_t ticks;
    __asm__ volatile ("rdcycle %0" : "=r"(ticks) : : "memory");
    return ticks;
}
static inline void model_profile_end(unsigned phase, uint64_t start) {
    model_profile_phases[phase].ticks += model_ticks() - start;
    ++model_profile_phases[phase].calls;
}
#else
static inline uint64_t model_ticks(void) { return 0; }
static inline void model_profile_end(unsigned phase, uint64_t start) {
    (void)phase; (void)start;
}
#endif
#endif
