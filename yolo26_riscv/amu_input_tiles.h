#ifndef YOLO_AMU_INPUT_TILES_H_
#define YOLO_AMU_INPUT_TILES_H_

/* Eight K rows x up to eight adjacent output positions. Segment stores
 * transpose the eight vectors into eight contiguous halves per position.
 * Each segment stays inside one 32-half AMU K row; no scalar scratch writes.
 * Interior loads are unit/constant stride. Only image edges use a mask. */
static inline vuint16m1_t spatial_half(const Conv2D *p,const unsigned char *source,
        unsigned bits,size_t batch_plane,size_t inner,size_t k,size_t pos,
        size_t ow,size_t vl,int interior,int *invalid) {
    vuint16m1_t zero=__riscv_vmv_v_x_u16m1(0,vl);
    if(inner>=k) return zero;
    size_t kernel=(size_t)p->kernel_h*p->kernel_w;
    size_t channel=inner/kernel,ky=inner%kernel/p->kernel_w,kx=inner%p->kernel_w;
    int64_t y=(int64_t)(pos/ow*p->stride_h+ky)-p->pad_h;
    int64_t x=(int64_t)(pos%ow*p->stride_w+kx)-p->pad_w;
    if(y<0 || y>=p->in_h) return zero;
    size_t row=channel*batch_plane+(size_t)y*p->in_w;
    vbool16_t valid=__riscv_vmset_m_b16(vl);
    vuint32m2_t offsets=__riscv_vmv_v_x_u32m2(0,vl);
    if(!interior) {
        offsets=__riscv_vadd_vx_u32m2(__riscv_vmul_vx_u32m2(
            __riscv_vid_v_u32m2(vl),p->stride_w,vl),(uint32_t)x,vl);
        valid=__riscv_vmsltu_vx_u32m2_b16(offsets,p->in_w,vl);
    }
    vuint16m1_t half;
    if(bits==32) {
        const uint32_t *s=(const uint32_t *)source+row;
        vuint32m2_t v=interior ? __riscv_vlse32_v_u32m2(s+(size_t)x,p->stride_w*4,vl) :
            __riscv_vluxei32_v_u32m2_m(valid,s,__riscv_vsll_vx_u32m2(offsets,2,vl),vl);
        if(!interior) v=__riscv_vmerge_vvm_u32m2(__riscv_vmv_v_x_u32m2(0,vl),v,valid,vl);
        if(__riscv_vfirst_m_b16(__riscv_vmsgeu_vx_u32m2_b16(
            __riscv_vand_vx_u32m2(v,0x7fffffffu,vl),0x477ff000u,vl),vl)>=0) *invalid=1;
        half=amu_half_bits_rvv(v,vl);
    } else if(bits==16) {
        const uint16_t *s=(const uint16_t *)source+row;
        half=interior ? __riscv_vlse16_v_u16m1(s+(size_t)x,p->stride_w*2,vl) :
            __riscv_vluxei32_v_u16m1_m(valid,s,__riscv_vsll_vx_u32m2(offsets,1,vl),vl);
    } else {
        const uint8_t *s=source+row;
        vuint8mf2_t v=interior ? __riscv_vlse8_v_u8mf2(s+(size_t)x,p->stride_w,vl) :
            __riscv_vluxei32_v_u8mf2_m(valid,s,offsets,vl);
        vuint16m1_t index=__riscv_vsll_vx_u16m1(__riscv_vzext_vf2_u16m1(v,vl),1,vl);
        half=__riscv_vluxei16_v_u16m1_m(valid,normalized,index,vl);
    }
    return interior ? half : __riscv_vmerge_vvm_u16m1(zero,half,valid,vl);
}

static int pack_spatial_tiles(const Conv2D *p,const unsigned char *source,unsigned bits,
        uint16_t *dest,size_t first,size_t columns,size_t k,size_t ow,size_t tile) {
    size_t plane=(size_t)p->in_h*p->in_w;
    for(size_t col=0;col<columns;) {
        size_t pos=first+col,count=columns-col;
        if(count>ow-pos%ow) count=ow-pos%ow;
        if(count>8) count=8;
        size_t vl=__riscv_vsetvl_e16m1(count);
        int64_t y=(int64_t)(pos/ow*p->stride_h)-p->pad_h;
        int64_t x=(int64_t)(pos%ow*p->stride_w)-p->pad_w;
        int interior=y>=0 && x>=0 && y+p->kernel_h<=p->in_h &&
            x+(int64_t)(vl-1)*p->stride_w+p->kernel_w<=p->in_w;
        for(size_t inner=0;inner<k;inner+=8) {
            int invalid=0;
#define ROW(i) spatial_half(p,source,bits,plane,inner+(i),k,pos,ow,vl,interior,&invalid)
            vuint16m1_t r0=ROW(0),r1=ROW(1),r2=ROW(2),r3=ROW(3);
            vuint16m1_t r4=ROW(4),r5=ROW(5),r6=ROW(6),r7=ROW(7);
#undef ROW
            if(invalid) return -1;
            vuint16m1x8_t rows=__riscv_vcreate_v_u16m1x8(r0,r1,r2,r3,r4,r5,r6,r7);
            __riscv_vssseg8e16_v_u16m1x8(dest+(inner/32)*tile+col*32+inner%32,64,rows,vl);
        }
        col+=vl;
    }
    return 0;
}
#endif
