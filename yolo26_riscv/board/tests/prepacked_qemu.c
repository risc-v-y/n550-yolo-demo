/* QEMU regression: unchanged AMU arithmetic, tails and two-frame reuse. */
#include "../../amu.h"
#include "../../model_profile.h"
static float a[64*1152],b[1152*35],legacy[64*35],packed_output[64*35];
static uint16_t weights[64*1152] __attribute__((aligned(64)));
ModelProfilePhase model_profile_phases[PROFILE_PHASE_COUNT];
static uintptr_t host_call(uintptr_t op,uintptr_t arg) {
    register uintptr_t a0 __asm__("a0")=op;
    register uintptr_t a1 __asm__("a1")=arg;
    __asm__ volatile(".option push\n.option norvc\n.balign 16\nslli zero,zero,31\n"
                     "ebreak\nsrai zero,zero,7\n.option pop" : "+r"(a0),"+r"(a1) :: "memory");
    return a0;
}
__attribute__((noreturn)) void platform_exit(uintptr_t code) {
    if(code) host_call(4,(uintptr_t)"PREPACKED AMU FAILED\n");
    uintptr_t args[2]={0x20026,code}; host_call(0x20,(uintptr_t)args);
    for(;;) __asm__ volatile("wfi");
}
void platform_trap(uintptr_t cause,uintptr_t pc,uintptr_t value,const uintptr_t *regs) {
    (void)cause; (void)pc; (void)value; (void)regs; platform_exit(99);
}
static uint64_t fingerprint(const uint16_t *p,size_t n) {
    uint64_t result=1;
    for(size_t i=0;i<n;++i) result=(result^p[i])*UINT64_C(1099511628211);
    return result;
}
int main(void) {
    if(amu_init()) platform_exit(1);
    static const size_t shapes[][3]={{1,1,1},{3,5,16},{3,5,17},{16,35,27},
        {32,32,31},{32,32,32},{33,35,33},{33,35,65},{5,7,96},{1,17,1152},{64,2,576}};
    for(unsigned test=0;test<sizeof(shapes)/sizeof(shapes[0]);++test) {
        size_t m=shapes[test][0],n=shapes[test][1],k=shapes[test][2];
        for(size_t i=0;i<m*k;++i) a[i]=(float)((int)(i%97)-48)/19.0f;
        AmuPackedWeights descriptor;
        if(amu_pack_weights(a,m,k,k,weights,sizeof(weights),&descriptor)) platform_exit(2);
        uint64_t hash=fingerprint(weights,descriptor.bytes/2);
        for(unsigned frame=0;frame<2;++frame) {
            for(size_t i=0;i<k*n;++i) b[i]=(float)((int)((i*13+frame*7)%101)-50)/23.0f;
            /* Long-K cancellation, including positive/negative values. */
            if(frame) for(size_t i=0;i<k*n;i+=2) b[i]=-b[i];
            if(amu_matmul_fp16(a,b,legacy,m,n,k,k,n,n)) platform_exit(3);
            uint64_t conversions=amu_runtime_weight_conversions;
            if(!conversions || amu_matmul_prepacked_fp16(&descriptor,b,packed_output,n,n,n)) platform_exit(4);
            if(amu_runtime_weight_conversions!=conversions || fingerprint(weights,descriptor.bytes/2)!=hash)
                platform_exit(5);
            for(size_t i=0;i<m*n;++i) {
                union { float f; uint32_t u; } old={.f=legacy[i]},new={.f=packed_output[i]};
                if(old.u!=new.u) platform_exit(6);
            }
        }
    }
    host_call(4,(uintptr_t)"PREPACKED AMU PASS: 11 shapes, two inputs each, exact FP32, immutable weights\n");
    platform_exit(0);
}
