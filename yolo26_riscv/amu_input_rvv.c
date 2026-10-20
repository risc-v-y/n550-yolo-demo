#include "amu.h"
#include "amu_half_rvv.h"
#ifdef YOLO_N550_BOARD
#include "board/cache.h"
#include "board/memory.h"
#include "board/diagnostics.h"
#endif

/* Private cached-DDR metadata, not an activation or AMU buffer. Rebuilt only
 * when the input/kernel shape changes. CLP reservation remains unchanged. */
static uint32_t indices[3][AMU_INPUT_MAX_K] __attribute__((aligned(64)));
static size_t plan_k,plan_h,plan_w,plan_kh,plan_kw;
uint64_t amu_rvv_input_blocks;

static void zero_half(uint16_t *dst,size_t count) {
    while(count) {
        size_t vl=__riscv_vsetvl_e16m1(count);
        __riscv_vse16_v_u16m1(dst,__riscv_vmv_v_x_u16m1(0,vl),vl);
        dst+=vl; count-=vl;
    }
}

int amu_pack_conv_input_rvv(const Conv2D *p,const float *input,size_t batch,
                            size_t first,size_t columns,uint16_t *storage,
                            size_t capacity,AmuPackedInput *packed) {
    if(!packed) return -1;
    *packed=(AmuPackedInput){0};
    size_t k,oh,ow;
    if(!input || !storage || (uintptr_t)storage%64 || !columns || columns>32 ||
       amu_conv_input_shape(p,&k,&oh,&ow) || batch>=p->batch || first>=oh*ow ||
       columns>oh*ow-first) return -1;
    size_t bytes=amu_packed_weight_bytes(32,k),plane=(size_t)p->in_h*p->in_w;
    if(capacity<bytes) return -1;
    /* EI32 byte offsets and unsigned boundary masks cover every frozen
     * shape. Preserve the scalar API for shapes outside that address range. */
    uint64_t largest=(uint64_t)(p->in_channels-1)*plane+
                     (uint64_t)(p->kernel_h-1)*p->in_w+p->kernel_w-1;
    if((uint64_t)p->in_h+2ull*p->pad_h>INT32_MAX ||
       (uint64_t)p->in_w+2ull*p->pad_w>INT32_MAX || largest>UINT32_MAX/4)
        return amu_pack_conv_input(p,input,batch,first,columns,storage,capacity,packed);
#ifdef YOLO_N550_BOARD
    board_diag_phase(PHASE_AMU_PACK);
#endif
    int pointwise=p->kernel_h==1 && p->kernel_w==1 && p->stride_h==1 &&
                  p->stride_w==1 && !p->pad_h && !p->pad_w;
    if(!pointwise && (plan_k!=k || plan_h!=p->in_h || plan_w!=p->in_w ||
       plan_kh!=p->kernel_h || plan_kw!=p->kernel_w)) {
        size_t inner=0;
        for(size_t ic=0;ic<p->in_channels;++ic)
            for(size_t ky=0;ky<p->kernel_h;++ky) for(size_t kx=0;kx<p->kernel_w;++kx,++inner) {
                indices[0][inner]=(uint32_t)((ic*plane+ky*p->in_w+kx)*4);
                indices[1][inner]=(uint32_t)ky;
                indices[2][inner]=(uint32_t)kx;
            }
#ifdef YOLO_N550_BOARD
        for(unsigned i=0;i<3;++i) board_clean(indices[i],k*4);
#endif
        plan_k=k; plan_h=p->in_h; plan_w=p->in_w;
        plan_kh=p->kernel_h; plan_kw=p->kernel_w;
    }
    const uint32_t *source=(const uint32_t *)(input+batch*p->in_channels*plane);
    uint16_t *destination=storage;
#ifdef YOLO_N550_BOARD
    /* Graph boundary has already published input. Transfer only scratch;
     * raw u32m2/u16m1 intrinsics intentionally bypass per-vector wrappers. */
    board_prepare_write(storage,bytes);
    source=(const uint32_t *)board_rvv_address(source);
    destination=(uint16_t *)board_rvv_address(storage);
#endif
    size_t kt=(k+31)/32;
    for(size_t block=0;block<kt;++block)
        zero_half(destination+block*1024+columns*32,(32-columns)*32);
    if(k%32) for(size_t col=0;col<columns;++col)
        zero_half(destination+(kt-1)*1024+col*32+k%32,32-k%32);
    for(size_t col=0;col<columns;++col) {
        size_t pos=first+col;
        uint32_t y0=(uint32_t)(pos/ow*p->stride_h-p->pad_h);
        uint32_t x0=(uint32_t)(pos%ow*p->stride_w-p->pad_w);
        uint32_t base=(y0*p->in_w+x0)*4;
        int interior=y0<p->in_h && x0<p->in_w &&
                     p->kernel_h<=p->in_h-y0 && p->kernel_w<=p->in_w-x0;
        for(size_t inner=0;inner<k;) {
            size_t count=k-inner<32-inner%32 ? k-inner : 32-inner%32;
            size_t vl=__riscv_vsetvl_e32m2(count);
            vuint32m2_t bits;
            if(pointwise) {
                bits=__riscv_vlse32_v_u32m2(source+inner*plane+pos,(ptrdiff_t)(plane*4),vl);
            } else {
                vuint32m2_t offset=__riscv_vadd_vx_u32m2(__riscv_vle32_v_u32m2(indices[0]+inner,vl),base,vl);
                if(interior) bits=__riscv_vluxei32_v_u32m2(source,offset,vl);
                else {
                    vuint32m2_t y=__riscv_vadd_vx_u32m2(__riscv_vle32_v_u32m2(indices[1]+inner,vl),y0,vl);
                    vuint32m2_t x=__riscv_vadd_vx_u32m2(__riscv_vle32_v_u32m2(indices[2]+inner,vl),x0,vl);
                    vbool16_t valid=__riscv_vmand_mm_b16(__riscv_vmsltu_vx_u32m2_b16(y,p->in_h,vl),
                                                       __riscv_vmsltu_vx_u32m2_b16(x,p->in_w,vl),vl);
                    bits=__riscv_vluxei32_v_u32m2_m(valid,source,offset,vl);
                    bits=__riscv_vmerge_vvm_u32m2(__riscv_vmv_v_x_u32m2(0,vl),bits,valid,vl);
                }
            }
            vuint32m2_t absolute=__riscv_vand_vx_u32m2(bits,0x7fffffffu,vl);
            if(__riscv_vfirst_m_b16(__riscv_vmsgeu_vx_u32m2_b16(absolute,0x477ff000u,vl),vl)>=0)
                goto invalid;
            __riscv_vse16_v_u16m1(destination+(inner/32)*1024+col*32+inner%32,
                                  amu_half_bits_rvv(bits,vl),vl);
            inner+=vl;
        }
    }
#ifdef YOLO_N550_BOARD
    board_finish_write(storage,bytes);
#endif
    *packed=(AmuPackedInput){storage,k,columns,bytes};
    ++amu_rvv_input_blocks;
    return 0;
invalid:
#ifdef YOLO_N550_BOARD
    board_finish_write(storage,bytes);
#endif
    return -1;
}
