#include "amu.h"
#include "amu_half_rvv.h"
#include "rvv_access.h"
#ifdef YOLO_N550_BOARD
#include "board/cache.h"
#include "board/memory.h"
#include "board/diagnostics.h"
#endif

/* Private cached-DDR metadata, not an activation or AMU buffer. Rebuilt only
 * when the input/kernel shape changes. CLP reservation remains unchanged. */
static uint32_t indices[3][AMU_INPUT_MAX_K] __attribute__((aligned(64)));
static size_t plan_k,plan_h,plan_w,plan_kh,plan_kw;
static int plan_rows;
uint64_t amu_rvv_input_blocks;
uint64_t amu_rvv_cached_blocks,amu_rvv_row_values;

static int overlap(const void *a,size_t na,const void *b,size_t nb) {
    uintptr_t aa=(uintptr_t)a,bb=(uintptr_t)b;
    return na>UINTPTR_MAX-aa || nb>UINTPTR_MAX-bb || (aa<bb+nb && bb<aa+na);
}

int amu_input_rows_begin(AmuInputRows *rows,const Conv2D *p,const float *input,
                          size_t batch,uint16_t *storage,size_t capacity) {
    if(!rows) return -1;
    *rows=(AmuInputRows){0};
    size_t k,oh,ow;
    if(!input || !storage || (uintptr_t)storage%64 ||
       amu_conv_input_shape(p,&k,&oh,&ow) || batch>=p->batch ||
       p->kernel_h!=3 || p->kernel_w!=3 || p->stride_h>2 || p->stride_w>2)
        return -1;
    size_t count=(size_t)p->in_channels*p->in_w;
    if(count>SIZE_MAX/6 || count*6>capacity || count*6>UINT32_MAX ||
       overlap(storage,count*6,input,(size_t)p->batch*p->in_channels*p->in_h*p->in_w*4))
        return -1;
    *rows=(AmuInputRows){.shape=*p,.input=input,.data=storage,.batch=batch,
                         .bytes=count*6,.row={SIZE_MAX,SIZE_MAX,SIZE_MAX}};
    (void)rvv_write_buffer(storage,rows->bytes);
    return 0;
}

void amu_input_rows_end(AmuInputRows *rows) {
    if(rows && rows->data) {
        rvv_publish_buffer(rows->data,rows->bytes);
        rows->data=0;
    }
}

static int cache_row(AmuInputRows *rows,size_t y,int probe,size_t column) {
#ifdef YOLO_N550_BOARD
    if(probe && board_pack_mark(PACK_ROW_CHECK,column,0,(int64_t)y,0,rows->row,rows->converted)) return -1;
#else
    (void)probe; (void)column;
#endif
    size_t slot=y%3;
    if(rows->row[slot]==y) {
#ifdef YOLO_N550_BOARD
        if(probe && board_pack_mark(PACK_ROW_HIT,column,0,(int64_t)y,0,rows->row,rows->converted)) return -1;
#endif
        return 0;
    }
#ifdef YOLO_N550_BOARD
    if(probe && board_pack_mark(PACK_ROW_FILL_BEGIN,column,0,(int64_t)y,0,rows->row,rows->converted)) return -1;
#endif
    const Conv2D *p=&rows->shape;
    size_t plane=(size_t)p->in_h*p->in_w;
    const uint32_t *source=(const uint32_t *)rows->input;
    uint16_t *dest=rows->data+slot*p->in_channels*p->in_w;
#ifdef YOLO_N550_BOARD
    source=(const uint32_t *)board_rvv_address(source);
    dest=(uint16_t *)board_rvv_address(dest);
#endif
    for(size_t c=0;c<p->in_channels;++c) for(size_t x=0;x<p->in_w;) {
        size_t vl=__riscv_vsetvl_e32m2(p->in_w-x);
        vuint32m2_t bits=__riscv_vle32_v_u32m2(source+(rows->batch*p->in_channels+c)*plane+y*p->in_w+x,vl);
        vbool16_t invalid=__riscv_vmsgeu_vx_u32m2_b16(__riscv_vand_vx_u32m2(bits,0x7fffffffu,vl),0x477ff000u,vl);
        /* No valid half is infinite. Keep a sentinel rather than rejecting
         * pixels that the selected stride/padding window never consumes. */
        vuint16m1_t half=amu_half_bits_rvv(bits,vl);
        half=__riscv_vmerge_vxm_u16m1(half,0x7c00u,invalid,vl);
        __riscv_vse16_v_u16m1(dest+c*p->in_w+x,half,vl);
        x+=vl;
    }
    __asm__ volatile("fence rw,rw" ::: "memory");
    rows->row[slot]=y;
    rows->converted+=(size_t)p->in_channels*p->in_w;
    amu_rvv_row_values+=(size_t)p->in_channels*p->in_w;
#ifdef YOLO_N550_BOARD
    if(probe && board_pack_mark(PACK_ROW_FILL_DONE,column,0,(int64_t)y,0,rows->row,rows->converted)) return -1;
#endif
    return 0;
}

static void zero_half(uint16_t *dst,size_t count) {
    while(count) {
        size_t vl=__riscv_vsetvl_e16m1(count);
        __riscv_vse16_v_u16m1(dst,__riscv_vmv_v_x_u16m1(0,vl),vl);
        dst+=vl; count-=vl;
    }
}

static int pack_input(const Conv2D *p,const float *input,size_t batch,
                            size_t first,size_t columns,uint16_t *storage,
                            size_t capacity,AmuPackedInput *packed,AmuInputRows *rows) {
    if(!packed) return -1;
    *packed=(AmuPackedInput){0};
    size_t k,oh,ow;
    if(!input || !storage || (uintptr_t)storage%64 || !columns || columns>32 ||
       amu_conv_input_shape(p,&k,&oh,&ow) || batch>=p->batch || first>=oh*ow ||
       columns>oh*ow-first) return -1;
    size_t bytes=amu_packed_weight_bytes(32,k),plane=(size_t)p->in_h*p->in_w;
    if(capacity<bytes) return -1;
    if(rows && (!rows->data || overlap(storage,bytes,rows->data,rows->bytes) ||
                 overlap(storage,bytes,input,(size_t)p->batch*p->in_channels*plane*4))) return -1;
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
    int probe=0;
#ifdef YOLO_N550_BOARD
    if(rows) probe=board_pack_begin(first,columns,k,input,rows->data,storage,rows->bytes);
#define PACK_MARK(step,c,i,r,v) do { \
    if(probe && board_pack_mark(step,c,i,r,v,rows->row,rows->converted)) goto invalid; \
} while(0)
#else
#define PACK_MARK(step,c,i,r,v) do { } while(0)
#endif
    PACK_MARK(PACK_PREPARE_BEGIN,0,0,-1,0);
    int pointwise=p->kernel_h==1 && p->kernel_w==1 && p->stride_h==1 &&
                  p->stride_w==1 && !p->pad_h && !p->pad_w;
    if(!pointwise && (plan_k!=k || plan_h!=p->in_h || plan_w!=p->in_w ||
       plan_kh!=p->kernel_h || plan_kw!=p->kernel_w || plan_rows!=(rows!=0))) {
        size_t inner=0;
        for(size_t ic=0;ic<p->in_channels;++ic)
            for(size_t ky=0;ky<p->kernel_h;++ky) for(size_t kx=0;kx<p->kernel_w;++kx,++inner) {
                indices[0][inner]=rows ? (uint32_t)((ic*p->in_w+kx)*2) :
                                        (uint32_t)((ic*plane+ky*p->in_w+kx)*4);
                indices[1][inner]=(uint32_t)ky;
                indices[2][inner]=(uint32_t)kx;
            }
#ifdef YOLO_N550_BOARD
        for(unsigned i=0;i<3;++i) board_clean(indices[i],k*4);
#endif
        plan_k=k; plan_h=p->in_h; plan_w=p->in_w;
        plan_kh=p->kernel_h; plan_kw=p->kernel_w;
        plan_rows=rows!=0;
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
    PACK_MARK(PACK_PREPARE_DONE,0,0,-1,0);
    for(size_t col=0;col<columns;++col) {
        size_t pos=first+col;
        uint32_t y0=(uint32_t)(pos/ow*p->stride_h-p->pad_h);
        uint32_t x0=(uint32_t)(pos%ow*p->stride_w-p->pad_w);
        uint32_t base=(y0*p->in_w+x0)*4;
        int interior=y0<p->in_h && x0<p->in_w &&
                     p->kernel_h<=p->in_h-y0 && p->kernel_w<=p->in_w-x0;
        uint32_t slot=0;
        const uint16_t *cached=rows ? rows->data : 0;
        if(rows) {
            int64_t iy=(int64_t)(pos/ow*p->stride_h)-p->pad_h;
            slot=(uint32_t)((iy%3+3)%3);
            for(unsigned ky=0;ky<3;++ky)
                if(iy+ky>=0 && iy+ky<p->in_h && cache_row(rows,(size_t)(iy+ky),probe,col)) goto invalid;
            PACK_MARK(PACK_ROWS_DONE,col,0,iy,0);
#ifdef YOLO_N550_BOARD
            cached=(const uint16_t *)board_rvv_address(cached);
#endif
        }
        for(size_t inner=0;inner<k;) {
            PACK_MARK(PACK_INDEX_BEGIN,col,inner,-1,0);
            size_t count=k-inner<32-inner%32 ? k-inner : 32-inner%32;
            size_t vl=__riscv_vsetvl_e32m2(count);
            if(rows) {
                vuint32m2_t ky=__riscv_vle32_v_u32m2(indices[1]+inner,vl);
                vuint32m2_t s=__riscv_vadd_vx_u32m2(ky,slot,vl);
                s=__riscv_vmerge_vvm_u32m2(s,__riscv_vsub_vx_u32m2(s,3,vl),
                                         __riscv_vmsgeu_vx_u32m2_b16(s,3,vl),vl);
                vuint32m2_t offset=__riscv_vadd_vv_u32m2(__riscv_vle32_v_u32m2(indices[0]+inner,vl),
                    __riscv_vmul_vx_u32m2(s,(uint32_t)((size_t)p->in_channels*p->in_w*2),vl),vl);
                offset=__riscv_vadd_vx_u32m2(offset,x0*2,vl);
                PACK_MARK(PACK_INDEX_DONE,col,inner,-1,vl);
                vuint16m1_t half;
                if(interior) {
                    PACK_MARK(PACK_GATHER_BEGIN,col,inner,-1,vl);
                    half=__riscv_vluxei32_v_u16m1(cached,offset,vl);
                }
                else {
                    vuint32m2_t y=__riscv_vadd_vx_u32m2(ky,y0,vl);
                    vuint32m2_t x=__riscv_vadd_vx_u32m2(__riscv_vle32_v_u32m2(indices[2]+inner,vl),x0,vl);
                    vbool16_t valid=__riscv_vmand_mm_b16(__riscv_vmsltu_vx_u32m2_b16(y,p->in_h,vl),
                                                       __riscv_vmsltu_vx_u32m2_b16(x,p->in_w,vl),vl);
                    PACK_MARK(PACK_GATHER_BEGIN,col,inner,-1,vl);
                    half=__riscv_vluxei32_v_u16m1_m(valid,cached,offset,vl);
                    half=__riscv_vmerge_vvm_u16m1(__riscv_vmv_v_x_u16m1(0,vl),half,valid,vl);
                }
                PACK_MARK(PACK_GATHER_DONE,col,inner,-1,vl);
                vuint16m1_t absolute=__riscv_vand_vx_u16m1(half,0x7fffu,vl);
                if(__riscv_vfirst_m_b16(__riscv_vmsgeu_vx_u16m1_b16(absolute,0x7c00u,vl),vl)>=0)
                    goto invalid;
                PACK_MARK(PACK_STORE_BEGIN,col,inner,-1,vl);
                __riscv_vse16_v_u16m1(destination+(inner/32)*1024+col*32+inner%32,half,vl);
                PACK_MARK(PACK_STORE_DONE,col,inner,-1,vl);
                inner+=vl;
                continue;
            }
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
        PACK_MARK(PACK_COLUMN_DONE,col,k,-1,0);
    }
#ifdef YOLO_N550_BOARD
    board_finish_write(storage,bytes);
#endif
    PACK_MARK(PACK_BLOCK_DONE,columns,k,-1,0);
    *packed=(AmuPackedInput){storage,k,columns,bytes};
    ++amu_rvv_input_blocks;
    if(rows) ++amu_rvv_cached_blocks;
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
    return pack_input(p,input,batch,first,columns,storage,capacity,packed,0);
}

int amu_pack_conv_rows_rvv(AmuInputRows *rows,size_t first,size_t columns,
                          uint16_t *storage,size_t capacity,AmuPackedInput *packed) {
    if(!rows || !rows->data) {
        if(packed) *packed=(AmuPackedInput){0};
        return -1;
    }
    return pack_input(&rows->shape,rows->input,rows->batch,first,columns,storage,capacity,packed,rows);
}
