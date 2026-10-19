#ifndef YOLO_AMU_HALF_RVV_H
#define YOLO_AMU_HALF_RVV_H
#include <riscv_vector.h>
#include <stdint.h>
/* Integer RVV implements amu_half's RNE bit rule, independently of frm.
 * Caller rejects abs(bits)>=0x477ff000 (overflow, infinity or NaN).
 * e32,m2 -> e16,m1 preserves VL; it uses no vector FP16 arithmetic.
 * ISA/intrinsic references: docs.riscv.org/reference/isa/unpriv/v-st-ext
 * and github.com/riscv-non-isa/riscv-rvv-intrinsic-doc. */
static inline vuint16m1_t amu_half_bits_rvv(vuint32m2_t bits,size_t vl) {
    vuint32m2_t absolute=__riscv_vand_vx_u32m2(bits,0x7fffffffu,vl);
    vuint32m2_t sign=__riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(bits,16,vl),0x8000u,vl);
    vuint32m2_t odd=__riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(absolute,13,vl),1,vl);
    vuint32m2_t rounded=__riscv_vadd_vv_u32m2(__riscv_vadd_vx_u32m2(absolute,0xfffu,vl),odd,vl);
    rounded=__riscv_vsrl_vx_u32m2(__riscv_vsub_vx_u32m2(rounded,0x38000000u,vl),13,vl);
    vbool16_t normal=__riscv_vmsgeu_vx_u32m2_b16(absolute,0x38800000u,vl);
    vuint32m2_t result=__riscv_vmerge_vvm_u32m2(__riscv_vmv_v_x_u32m2(0,vl),rounded,normal,vl);
    vbool16_t sub=__riscv_vmand_mm_b16(__riscv_vmsgeu_vx_u32m2_b16(absolute,0x33000000u,vl),
                                    __riscv_vmsltu_vx_u32m2_b16(absolute,0x38800000u,vl),vl);
    if(__riscv_vfirst_m_b16(sub,vl)>=0) {
        vuint32m2_t exponent=__riscv_vsrl_vx_u32m2(absolute,23,vl);
        vuint32m2_t shift=__riscv_vrsub_vx_u32m2(exponent,126,vl);
        shift=__riscv_vmaxu_vx_u32m2(__riscv_vminu_vx_u32m2(shift,24,vl),1,vl);
        vuint32m2_t mantissa=__riscv_vor_vx_u32m2(__riscv_vand_vx_u32m2(absolute,0x7fffffu,vl),0x800000u,vl);
        vuint32m2_t truncated=__riscv_vsrl_vv_u32m2(mantissa,shift,vl);
        vuint32m2_t halfway=__riscv_vsll_vv_u32m2(__riscv_vmv_v_x_u32m2(1,vl),
                                              __riscv_vsub_vx_u32m2(shift,1,vl),vl);
        vuint32m2_t bias=__riscv_vadd_vv_u32m2(__riscv_vsub_vx_u32m2(halfway,1,vl),
                                           __riscv_vand_vx_u32m2(truncated,1,vl),vl);
        vuint32m2_t small=__riscv_vsrl_vv_u32m2(__riscv_vadd_vv_u32m2(mantissa,bias,vl),shift,vl);
        result=__riscv_vmerge_vvm_u32m2(result,small,sub,vl);
    }
    return __riscv_vnsrl_wx_u16m1(__riscv_vor_vv_u32m2(result,sign,vl),0,vl);
}
#endif
