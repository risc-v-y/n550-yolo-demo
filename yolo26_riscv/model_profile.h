#ifndef YOLO26_MODEL_PROFILE_H_
#define YOLO26_MODEL_PROFILE_H_

#include <stdint.h>
#ifndef MODEL_PROFILE
#define MODEL_PROFILE 0
#endif

/* Disjoint internal stages; their sum is a subset of node compute time.
 * rdcycle is a CPU cycle counter on hardware, an emulator counter in QEMU.
 * Only physical-board cycles and its actual core frequency give board time.
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
extern ModelProfilePhase model_profile_phases[PROFILE_PHASE_COUNT];
#if MODEL_PROFILE
#ifdef YOLO_N550_BOARD
extern int model_profile_enabled;
#endif
static inline uint64_t model_ticks(void) {
#ifdef YOLO_N550_BOARD
    if(!model_profile_enabled) return 0;
#endif
    uint64_t ticks;
    __asm__ volatile ("rdcycle %0" : "=r"(ticks) : : "memory");
    return ticks;
}
static inline void model_profile_end(unsigned phase, uint64_t start) {
#ifdef YOLO_N550_BOARD
    if(!model_profile_enabled) return;
#endif
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
