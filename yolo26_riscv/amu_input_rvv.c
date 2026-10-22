#include "amu.h"
#include "amu_half_rvv.h"
#ifdef YOLO_N550_BOARD
#include "board/cache.h"
#include "board/memory.h"
#include "board/diagnostics.h"
#endif

/* Shape-only offsets; no converted activation rows or per-pixel row tags. */
static uint32_t indices[3][AMU_INPUT_MAX_K] __attribute__((aligned(64)));
static size_t plan_k,plan_h,plan_w,plan_kh,plan_kw;
uint64_t amu_rvv_input_blocks;
unsigned amu_input_flags;
static const float *u8_logical, *prepared_input;
static const uint8_t *u8_physical;
static const void *prepared_source;
static unsigned prepared_bits=32;
static uint16_t normalized[256] __attribute__((aligned(64)));
static int normalized_ready;

void amu_input_bind_u8(const float *logical,const uint8_t *physical) {
    u8_logical=logical; u8_physical=physical;
}
int amu_input_prepare(const Conv2D *p,const float *input,uint16_t *half,size_t capacity) {
    prepared_input=input; prepared_source=input; prepared_bits=32;
    if(u8_physical && input==u8_logical) {
        if(!normalized_ready) {
            for(unsigned i=0;i<256;++i)
                if(amu_half((float)i/255.0f,normalized+i)) return -1;
#ifdef YOLO_N550_BOARD
            board_clean(normalized,sizeof(normalized));
#endif
            normalized_ready=1;
        }
        prepared_source=u8_physical; prepared_bits=8;
        /* Consume the frame binding exactly once. The arena may reuse this
         * logical address for an unrelated later convolution's FP32 tensor. */
        u8_physical=0;
        return 0;
    }
    if(!(amu_input_flags&AMU_INPUT_PRECONVERT) || p->kernel_h*p->kernel_w==1) return 0;
    size_t count=(size_t)p->batch*p->in_channels*p->in_h*p->in_w;
    if(!half || count>capacity) return -1;
    const uint32_t *src=(const uint32_t *)input;
    uint16_t *dst=half;
#ifdef YOLO_N550_BOARD
    board_clean(input,count*4);
    board_prepare_write(half,count*2);
    src=board_rvv_address(input); dst=board_rvv_address(half);
#endif
    int status=0;
    for(size_t i=0;i<count;) {
        size_t vl=__riscv_vsetvl_e32m2(count-i);
        vuint32m2_t bits=__riscv_vle32_v_u32m2(src+i,vl);
        vuint32m2_t absolute=__riscv_vand_vx_u32m2(bits,0x7fffffffu,vl);
        if(__riscv_vfirst_m_b16(__riscv_vmsgeu_vx_u32m2_b16(absolute,0x477ff000u,vl),vl)>=0) {
            status=-1; break;
        }
        __riscv_vse16_v_u16m1(dst+i,amu_half_bits_rvv(bits,vl),vl);
        i+=vl;
    }
#ifdef YOLO_N550_BOARD
    board_finish_write(half,count*2);
#endif
    if(status) return status;
    prepared_source=half; prepared_bits=16;
    return 0;
}

static void zero_half(uint16_t *dst,size_t count) {
    while(count) {
        size_t vl=__riscv_vsetvl_e16m1(count);
        __riscv_vse16_v_u16m1(dst,__riscv_vmv_v_x_u16m1(0,vl),vl);
        dst+=vl; count-=vl;
    }
}

/* Inline with a literal probe flag at both call sites. Normal packing has
 * no checkpoint branches/calls inside its vector loop. */
static inline __attribute__((always_inline)) int pack_input(
        const Conv2D *p,const float *input,size_t batch,size_t first,size_t columns,
        uint16_t *storage,AmuPackedInput *packed,size_t k,size_t ow,int probe) {
    (void)probe;
#ifdef YOLO_N550_BOARD
    static const size_t tags[3]={SIZE_MAX,SIZE_MAX,SIZE_MAX};
    uint64_t converted=0;
#define PACK_MARK(step,c,i,v) do { \
    if(probe && board_pack_mark(step,c,i,-1,v,tags,converted)) goto invalid; \
} while(0)
#else
#define PACK_MARK(step,c,i,v) do { } while(0)
#endif
    size_t bytes=amu_packed_input_bytes(columns,k),plane=(size_t)p->in_h*p->in_w;
    size_t padded=(columns+31)/32*32, tile=padded*32;
    PACK_MARK(PACK_PREPARE_BEGIN,0,0,0);
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
    unsigned source_bits=input==prepared_input ? prepared_bits : 32;
    const void *physical=input==prepared_input ? prepared_source : input;
    const unsigned char *source=(const unsigned char *)physical+batch*p->in_channels*plane*(source_bits/8);
    uint16_t *destination=storage;
#ifdef YOLO_N550_BOARD
    board_prepare_write(storage,bytes);
    source=board_rvv_address(source);
    destination=(uint16_t *)board_rvv_address(storage);
#endif
    size_t kt=(k+31)/32;
    for(size_t block=0;block<kt;++block)
        zero_half(destination+block*tile+columns*32,(padded-columns)*32);
    if(k%32) for(size_t col=0;col<columns;++col)
        zero_half(destination+(kt-1)*tile+col*32+k%32,32-k%32);
    PACK_MARK(PACK_PREPARE_DONE,0,0,0);
    if(pointwise && (amu_input_flags&AMU_INPUT_SPATIAL) && !probe) {
        /* Read adjacent spatial values of one channel; transpose directly
         * into [position,K] with a 64-byte strided half store. */
        for(size_t inner=0;inner<k;++inner) for(size_t col=0;col<columns;) {
            size_t vl=__riscv_vsetvl_e32m2(columns-col);
            size_t at=inner*plane+first+col;
            vuint16m1_t half;
            if(source_bits==32) {
                vuint32m2_t bits=__riscv_vle32_v_u32m2((const uint32_t *)source+at,vl);
                vuint32m2_t absolute=__riscv_vand_vx_u32m2(bits,0x7fffffffu,vl);
                if(__riscv_vfirst_m_b16(__riscv_vmsgeu_vx_u32m2_b16(absolute,0x477ff000u,vl),vl)>=0)
                    goto invalid;
                half=amu_half_bits_rvv(bits,vl);
            } else if(source_bits==16) half=__riscv_vle16_v_u16m1((const uint16_t *)source+at,vl);
            else {
                vuint16m1_t v=__riscv_vzext_vf2_u16m1(__riscv_vle8_v_u8mf2(source+at,vl),vl);
                half=__riscv_vluxei16_v_u16m1(normalized,__riscv_vsll_vx_u16m1(v,1,vl),vl);
            }
            __riscv_vsse16_v_u16m1(destination+(inner/32)*tile+col*32+inner%32,64,half,vl);
            col+=vl;
        }
        goto published;
    }
    for(size_t col=0;col<columns;++col) {
        size_t pos=first+col;
        uint32_t y0=(uint32_t)(pos/ow*p->stride_h-p->pad_h);
        uint32_t x0=(uint32_t)(pos%ow*p->stride_w-p->pad_w);
        uint32_t base=(y0*p->in_w+x0)*4;
        int interior=y0<p->in_h && x0<p->in_w &&
                     p->kernel_h<=p->in_h-y0 && p->kernel_w<=p->in_w-x0;
        for(size_t inner=0;inner<k;) {
            PACK_MARK(PACK_INDEX_BEGIN,col,inner,0);
            size_t count=k-inner<32-inner%32 ? k-inner : 32-inner%32;
            size_t vl=__riscv_vsetvl_e32m2(count);
            vuint32m2_t bits;
            vuint16m1_t half;
            if(source_bits!=32) {
                vuint32m2_t offset;
                vbool16_t valid=__riscv_vmset_m_b16(vl);
                if(pointwise) {
                    offset=__riscv_vadd_vx_u32m2(__riscv_vid_v_u32m2(vl),(uint32_t)inner,vl);
                    offset=__riscv_vadd_vx_u32m2(__riscv_vmul_vx_u32m2(offset,(uint32_t)plane,vl),(uint32_t)pos,vl);
                } else {
                    offset=__riscv_vadd_vx_u32m2(__riscv_vle32_v_u32m2(indices[0]+inner,vl),base,vl);
                    offset=__riscv_vsrl_vx_u32m2(offset,2,vl);
                    if(!interior) {
                        vuint32m2_t y=__riscv_vadd_vx_u32m2(__riscv_vle32_v_u32m2(indices[1]+inner,vl),y0,vl);
                        vuint32m2_t x=__riscv_vadd_vx_u32m2(__riscv_vle32_v_u32m2(indices[2]+inner,vl),x0,vl);
                        valid=__riscv_vmand_mm_b16(__riscv_vmsltu_vx_u32m2_b16(y,p->in_h,vl),
                                                 __riscv_vmsltu_vx_u32m2_b16(x,p->in_w,vl),vl);
                    }
                }
                if(source_bits==16) {
                    offset=__riscv_vsll_vx_u32m2(offset,1,vl);
                    half=__riscv_vluxei32_v_u16m1_m(valid,(const uint16_t *)source,offset,vl);
                } else {
                    vuint8mf2_t pixels=__riscv_vluxei32_v_u8mf2_m(valid,source,offset,vl);
                    vuint16m1_t values=__riscv_vzext_vf2_u16m1(pixels,vl);
                    half=__riscv_vluxei16_v_u16m1_m(valid,normalized,
                                                   __riscv_vsll_vx_u16m1(values,1,vl),vl);
                }
                half=__riscv_vmerge_vvm_u16m1(__riscv_vmv_v_x_u16m1(0,vl),half,valid,vl);
                goto store_half;
            }
            if(pointwise) {
                PACK_MARK(PACK_INDEX_DONE,col,inner,vl);
                PACK_MARK(PACK_GATHER_BEGIN,col,inner,vl);
                bits=__riscv_vlse32_v_u32m2((const uint32_t *)source+inner*plane+pos,(ptrdiff_t)(plane*4),vl);
            } else {
                vuint32m2_t offset=__riscv_vadd_vx_u32m2(__riscv_vle32_v_u32m2(indices[0]+inner,vl),base,vl);
                if(interior) {
                    PACK_MARK(PACK_INDEX_DONE,col,inner,vl);
                    PACK_MARK(PACK_GATHER_BEGIN,col,inner,vl);
                    bits=__riscv_vluxei32_v_u32m2((const uint32_t *)source,offset,vl);
                } else {
                    vuint32m2_t y=__riscv_vadd_vx_u32m2(__riscv_vle32_v_u32m2(indices[1]+inner,vl),y0,vl);
                    vuint32m2_t x=__riscv_vadd_vx_u32m2(__riscv_vle32_v_u32m2(indices[2]+inner,vl),x0,vl);
                    vbool16_t valid=__riscv_vmand_mm_b16(__riscv_vmsltu_vx_u32m2_b16(y,p->in_h,vl),
                                                       __riscv_vmsltu_vx_u32m2_b16(x,p->in_w,vl),vl);
                    PACK_MARK(PACK_INDEX_DONE,col,inner,vl);
                    PACK_MARK(PACK_GATHER_BEGIN,col,inner,vl);
                    bits=__riscv_vluxei32_v_u32m2_m(valid,(const uint32_t *)source,offset,vl);
                    bits=__riscv_vmerge_vvm_u32m2(__riscv_vmv_v_x_u32m2(0,vl),bits,valid,vl);
                }
            }
            PACK_MARK(PACK_GATHER_DONE,col,inner,vl);
            vuint32m2_t absolute=__riscv_vand_vx_u32m2(bits,0x7fffffffu,vl);
            if(__riscv_vfirst_m_b16(__riscv_vmsgeu_vx_u32m2_b16(absolute,0x477ff000u,vl),vl)>=0)
                goto invalid;
            half=amu_half_bits_rvv(bits,vl);
store_half:
            PACK_MARK(PACK_STORE_BEGIN,col,inner,vl);
            __riscv_vse16_v_u16m1(destination+(inner/32)*tile+col*32+inner%32,
                                  half,vl);
#ifdef YOLO_N550_BOARD
            if(probe) converted+=vl;
#endif
            PACK_MARK(PACK_STORE_DONE,col,inner,vl);
            inner+=vl;
        }
        PACK_MARK(PACK_COLUMN_DONE,col,k,0);
    }
published:
#ifdef YOLO_N550_BOARD
    board_finish_write(storage,bytes);
#endif
    PACK_MARK(PACK_BLOCK_DONE,columns,k,0);
    *packed=(AmuPackedInput){storage,k,columns,bytes};
    ++amu_rvv_input_blocks;
#undef PACK_MARK
    return 0;
invalid:
#ifdef YOLO_N550_BOARD
    board_finish_write(storage,bytes);
#endif
    return -1;
}

int amu_pack_conv_input_rvv(const Conv2D *p,const float *input,size_t batch,
                            size_t first,size_t columns,uint16_t *storage,
                            size_t capacity,AmuPackedInput *packed) {
    if(!packed) return -1;
    *packed=(AmuPackedInput){0};
    size_t k,oh,ow;
    if(!input || !storage || (uintptr_t)storage%64 || !columns || columns>AMU_MAX_ROWS ||
       amu_conv_input_shape(p,&k,&oh,&ow) || batch>=p->batch || first>=oh*ow ||
       columns>oh*ow-first) return -1;
    size_t bytes=amu_packed_input_bytes(columns,k),plane=(size_t)p->in_h*p->in_w;
    uintptr_t source=(uintptr_t)input,dest=(uintptr_t)storage;
    size_t source_bytes=(size_t)p->batch*p->in_channels*plane*4;
    if(capacity<bytes || source_bytes>UINTPTR_MAX-source || bytes>UINTPTR_MAX-dest ||
       (source<dest+bytes && dest<source+source_bytes)) return -1;
    uint64_t largest=(uint64_t)(p->in_channels-1)*plane+
                     (uint64_t)(p->kernel_h-1)*p->in_w+p->kernel_w-1;
    if((uint64_t)p->in_h+2ull*p->pad_h>INT32_MAX ||
       (uint64_t)p->in_w+2ull*p->pad_w>INT32_MAX || largest>UINT32_MAX/4)
        return amu_pack_conv_input(p,input,batch,first,columns,storage,capacity,packed);
#ifdef YOLO_N550_BOARD
    board_diag_phase(PHASE_AMU_PACK);
    if((board_diag.options&DIAG_TRACE) && !board_diag.node && !first &&
       board_pack_begin(first,columns,k,input,0,storage,0))
        return pack_input(p,input,batch,first,columns,storage,packed,k,ow,1);
#endif
    return pack_input(p,input,batch,first,columns,storage,packed,k,ow,0);
}
