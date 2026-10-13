/* Test production wrappers with cache-operation spies. QEMU has no S2C cache. */
#include "../memory.h"
#define YOLO_BOARD_CACHE_H
#define BOARD_INLINE static inline __attribute__((always_inline))
static unsigned cleans,prepares,finishes,fences,bad_address;
static void check(const void *p,size_t bytes) {
    if(!board_is_clp(p) || !bytes) ++bad_address;
    __asm__ volatile("" ::: "memory");
}
BOARD_INLINE void board_clean(const void *p,size_t n) { check(p,n); ++cleans; }
BOARD_INLINE void board_prepare_write(const void *p,size_t n) { check(p,n); ++prepares; }
BOARD_INLINE void board_finish_write(const void *p,size_t n) { check(p,n); ++finishes; }
BOARD_INLINE void board_fence(void) { ++fences; __asm__ volatile("fence iorw,iorw" ::: "memory"); }
#include "../coherent_rvv.h"
static float shared[32] BOARD_CLP_DATA;
static float cached[16] __attribute__((aligned(64)));
extern void clp_qemu_exit(uintptr_t code) __attribute__((noreturn));
__attribute__((noreturn)) void platform_exit(uintptr_t code) { clp_qemu_exit(code); }
__attribute__((noreturn)) void platform_trap(uintptr_t a,uintptr_t b,uintptr_t c,const uintptr_t *r) {
    (void)a; (void)b; (void)c; (void)r; clp_qemu_exit(99);
}
static void reset(void) { cleans=prepares=finishes=fences=bad_address=0; }
int main(void) {
    for(unsigned i=0;i<16;++i) cached[i]=123;
    for(unsigned i=0;i<32;++i) shared[i]=123;
    size_t vl=__riscv_vsetvl_e32m1(3);
    vfloat32m1_t value=__riscv_vfmv_v_f_f32m1(2,vl);
    __riscv_vse32_v_f32m1(cached+1,value,vl);
    if(prepares || finishes || cleans || fences!=1 || cached[0]!=123 || cached[1]!=2 || cached[4]!=123) return 1;
    reset();
    __riscv_vse32_v_f32m1(shared+1,value,vl);
    if(prepares!=1 || finishes!=1 || bad_address || shared[0]!=123 || shared[1]!=2 || shared[4]!=123) return 2;
    reset();
    value=__riscv_vle32_v_f32m1(shared+1,vl);
    __riscv_vse32_v_f32m1(cached+1,value,vl);
    if(cleans!=1 || prepares || finishes || bad_address || cached[1]!=2) return 3;
    reset();
    value=__riscv_vlse32_v_f32m1(shared+1,8,vl);
    __riscv_vsse32_v_f32m1(shared+16,8,value,vl);
    if(cleans!=3 || prepares!=3 || finishes!=3 || bad_address || shared[17]!=123 || shared[16]!=2) return 4;
    reset();
    const uint32_t index[3]={8,0,4};
    board_clean(shared,16); /* Same source ownership handoff as graph boundary. */
    vuint32m1_t ix=__riscv_vle32_v_u32m1(index,vl);
    value=__riscv_vluxei32_v_f32m1(shared,ix,vl);
    __riscv_vse32_v_f32m1(cached,value,vl);
    if(cleans!=1 || prepares || finishes || bad_address || cached[0]!=2 || cached[1]!=123 || cached[2]!=2) return 5;
    return 0;
}
