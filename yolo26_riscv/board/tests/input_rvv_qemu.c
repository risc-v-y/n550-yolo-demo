/* Actual integer RVV conversion and gather, compared with scalar packing. */
#include "../../amu.h"
#include "../../amu_half_rvv.h"
#include "../../scalar_model.h"
#include "model_generated.h"
extern void platform_exit(uintptr_t code) __attribute__((noreturn));
static float source[MODEL_ARENA_BYTES/4];
static uint16_t reference[AMU_INPUT_MAX_K*32] __attribute__((aligned(64)));
static struct {
    uint16_t before[32],data[128*1024],after[32];
} row_buffer __attribute__((aligned(64)));
static struct {
    uint16_t before[32],data[AMU_INPUT_MAX_K*32],after[32];
} actual __attribute__((aligned(64)));

static void check_words(const uint32_t *words,size_t n) {
    uint16_t result[64];
    for(size_t i=0;i<n;) {
        size_t vl=__riscv_vsetvl_e32m2(n-i);
        __riscv_vse16_v_u16m1(result,amu_half_bits_rvv(__riscv_vle32_v_u32m2(words+i,vl),vl),vl);
        for(size_t j=0;j<vl;++j) {
            union { uint32_t u; float f; } value={.u=words[i+j]};
            uint16_t expected;
            if(amu_half(value.f,&expected) || expected!=result[j]) platform_exit(40);
        }
        i+=vl;
    }
}
static void conversion_cases(void) {
    static const uint32_t edge[]={0,0x80000000,1,0x807fffff,0x32ffffff,0x33000000,
        0x33000001,0x33800000,0x387fc000,0x387fffff,0x38800000,0x3f801000,
        0x3f803000,0x477fe000,0x477fefff,0xc77fefff};
    uint32_t words[64],rng=20261019;
    for(unsigned round=0;round<4096;++round) {
        /* Integer conversion must be independent of floating rounding mode. */
        __asm__ volatile("csrw frm,%0" :: "r"((uintptr_t)(round%4)) : "memory");
        for(unsigned i=0;i<64;++i) {
            rng^=rng<<13; rng^=rng>>17; rng^=rng<<5;
            words[i]=(rng&0x7fffffff)<0x477ff000u ? rng : rng&0x807fffffu;
            if(i<sizeof(edge)/sizeof(edge[0])) words[i]=edge[i];
        }
        check_words(words,64);
    }
    __asm__ volatile("csrw fcsr,zero" ::: "memory");
    /* Around every positive/negative normal binary16 midpoint. */
    for(uint32_t h=0x400;h<0x7c00;++h) {
        uint32_t bits=((h/1024+112)<<23)|((h%1024)<<13);
        for(unsigned i=0;i<6;++i) {
            uint32_t at=bits+0xfff+i%3;
            words[i]=(at<0x477ff000 ? at : bits)|(i>=3 ? 0x80000000u : 0);
        }
        check_words(words,6);
    }
}
static void compare_block(const Conv2D *p,size_t batch,size_t first,size_t count) {
    AmuPackedInput old,new;
    for(unsigned i=0;i<32;++i) actual.before[i]=actual.after[i]=0x3555;
    if(amu_pack_conv_input(p,source,batch,first,count,reference,sizeof(reference),&old) ||
       amu_pack_conv_input_rvv(p,source,batch,first,count,actual.data,sizeof(actual.data),&new))
        platform_exit(41);
    if(old.k!=new.k || old.n!=new.n || old.bytes!=new.bytes) platform_exit(42);
    for(size_t i=0;i<old.bytes/2;++i) if(reference[i]!=actual.data[i]) platform_exit(43);
    for(unsigned i=0;i<32;++i) if(actual.before[i]!=0x3555 || actual.after[i]!=0x3555) platform_exit(44);
}
static void check_cached(const Conv2D *p,size_t batch,int full) {
    AmuInputRows rows;
    size_t k,oh,ow;
    if(amu_conv_input_shape(p,&k,&oh,&ow)) platform_exit(47);
    for(unsigned i=0;i<32;++i) row_buffer.before[i]=row_buffer.after[i]=0x3555;
    if(amu_input_rows_begin(&rows,p,source,batch,row_buffer.data,sizeof(row_buffer.data))) platform_exit(48);
    size_t area=oh*ow,starts[]={0,ow-1,area>32 ? area-32 : 0,area-1};
    for(size_t j=0;j<(full ? (area+31)/32 : 4);++j) {
        size_t first=full ? j*32 : starts[j],count=area-first<32 ? area-first : 32;
        AmuPackedInput old,new;
        for(unsigned i=0;i<32;++i) actual.before[i]=actual.after[i]=0x3555;
        if(amu_pack_conv_input(p,source,batch,first,count,reference,sizeof(reference),&old) ||
           amu_pack_conv_rows_rvv(&rows,first,count,actual.data,sizeof(actual.data),&new)) platform_exit(49);
        if(old.k!=new.k || old.n!=new.n || old.bytes!=new.bytes) platform_exit(50);
        for(size_t i=0;i<old.bytes/2;++i) if(reference[i]!=actual.data[i]) platform_exit(51);
        for(unsigned i=0;i<32;++i) if(actual.before[i]!=0x3555 || actual.after[i]!=0x3555) platform_exit(52);
    }
    if(!rows.converted || (full && rows.converted>(size_t)p->in_channels*p->in_h*p->in_w)) platform_exit(53);
    amu_input_rows_end(&rows);
    for(unsigned i=0;i<32;++i) if(row_buffer.before[i]!=0x3555 || row_buffer.after[i]!=0x3555) platform_exit(54);
    AmuPackedInput invalid={.data=actual.data};
    if(!amu_pack_conv_rows_rvv(&rows,0,1,actual.data,sizeof(actual.data),&invalid) || invalid.data) platform_exit(55);
}

static void cached_edges(void) {
    Conv2D p={2,3,5,7,16,3,3,1,1,1,1,1};
    for(unsigned frame=0;frame<2;++frame) {
        for(unsigned i=0;i<2*3*5*7;++i) source[i]=frame ? -(float)(i+1) : (float)i;
        check_cached(&p,frame,1);
    }
    p=(Conv2D){1,1,1,3,1,3,3,1,1,1,1,1};
    source[0]=1.0f; source[1]=2.0f;
    union { uint32_t u; float f; } bad={.u=0x7fc00001}; source[2]=bad.f;
    AmuInputRows rows;
    AmuPackedInput packed;
    if(amu_input_rows_begin(&rows,&p,source,0,row_buffer.data,sizeof(row_buffer.data)) ||
       amu_pack_conv_rows_rvv(&rows,0,1,actual.data,sizeof(actual.data),&packed)) platform_exit(56);
    if(!amu_pack_conv_rows_rvv(&rows,1,1,actual.data,sizeof(actual.data),&packed) || packed.data) platform_exit(57);
    amu_input_rows_end(&rows);
    if(!amu_input_rows_begin(&rows,&p,source,0,row_buffer.data,1) || rows.data) platform_exit(58);
    if(!amu_input_rows_begin(&rows,&p,source,0,(uint16_t *)source,sizeof(source)) || rows.data) platform_exit(59);
    if(amu_input_rows_begin(&rows,&p,source,0,row_buffer.data,sizeof(row_buffer.data))) platform_exit(60);
    if(!amu_pack_conv_rows_rvv(&rows,0,1,row_buffer.data,sizeof(row_buffer.data),&packed) || packed.data) platform_exit(61);
    amu_input_rows_end(&rows);
    source[2]=0.0f;
}
void test_input_rvv(void) {
    conversion_cases();
    static const uint32_t pattern[]={0,0x80000000,0x33800000,0xb3800000,0x3f801000,
        0x3f803000,0x387fc000,0x38800000,0x477fefff,0xc77fefff,0x3eaaaaab,0xbf123456,0x33000001};
    for(size_t i=0;i<sizeof(source)/sizeof(source[0]);++i) {
        union { uint32_t u; float f; } value={.u=pattern[i%13]}; source[i]=value.f;
    }
    unsigned layers=0;
    for(unsigned i=0;i<MODEL_NODE_COUNT;++i) {
        const ModelNode *node=model_nodes+i;
        if(node->op!=OP_CONV || node->p[4]!=1) continue;
        const ModelTensor *a=model_tensors+node->in[0],*w=model_tensors+node->in[1];
        const ModelTensor *out=model_tensors+node->out[0];
        Conv2D p={a->shape[0],a->shape[1],a->shape[2],a->shape[3],w->shape[0],
            w->shape[2],w->shape[3],node->p[0],node->p[1],node->p[2],node->p[3],1};
        size_t area=out->shape[2]*out->shape[3];
        size_t starts[]={0,out->shape[3]-1,area>32 ? area-32 : 0,area-1};
        for(unsigned j=0;j<4;++j) {
            size_t count=area-starts[j]<32 ? area-starts[j] : 32;
            compare_block(&p,0,starts[j],count);
        }
        if(p.kernel_h==3 && p.kernel_w==3) check_cached(&p,0,1);
        ++layers;
    }
    if(layers!=94) platform_exit(45);
    if(!amu_rvv_cached_blocks || !amu_rvv_row_values) platform_exit(62);
    cached_edges();
    Conv2D p={2,3,5,7,33,3,1,2,2,1,0,1};
    compare_block(&p,1,0,12); compare_block(&p,1,11,1);
    p=(Conv2D){1,1,1,1,1,1,1,1,1,0,0,1};
    static const uint32_t invalid[]={0x477ff000,0xc77ff000,0x7f800000,0xff800000,0x7fc00000};
    for(unsigned i=0;i<sizeof(invalid)/sizeof(invalid[0]);++i) {
        union { uint32_t u; float f; } value={.u=invalid[i]}; source[0]=value.f;
        AmuPackedInput new;
        if(!amu_pack_conv_input_rvv(&p,source,0,0,1,actual.data,sizeof(actual.data),&new) || new.data)
            platform_exit(46);
    }
}
