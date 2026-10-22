/* QEMU regression: unchanged AMU arithmetic, tails and two-frame reuse. */
#include "../../amu.h"
#include "../../model_profile.h"
static float a[129*1152],b[1152*129],legacy[129*129],packed_output[129*129];
static uint16_t weights[256*1152] __attribute__((aligned(64)));
static uint16_t inputs[1152*128] __attribute__((aligned(64)));
static float columns[1152*32];
ModelProfilePhase model_profile_phases[PROFILE_PHASE_COUNT];
extern void test_input_rvv(void);
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
    test_input_rvv();
    /* Tile M/N may change only independent output dimensions. Compare every
     * candidate with 32x32, including K=33 and rows/columns beyond 128. */
    for(size_t m=32;m<=128;m*=2) for(size_t n=32;n<=128;n*=2) {
        for(size_t i=0;i<129*33;++i) a[i]=(float)((int)(i%37)-18)/17.0f;
        for(size_t i=0;i<33*129;++i) b[i]=(float)((int)(i%41)-20)/19.0f;
        if(amu_configure(32,32) || amu_matmul_fp16(a,b,legacy,129,129,33,33,129,129)) platform_exit(63);
        if(amu_configure(m,n)) platform_exit(64);
        AmuPackedWeights w;
        if(amu_pack_weights(a,129,33,33,weights,sizeof(weights),&w)) platform_exit(65);
        Conv2D p={1,33,1,129,129,1,1,1,1,0,0,1};
        for(size_t first=0;first<129;first+=n) {
            AmuPackedInput in;
            size_t count=129-first<n ? 129-first : n;
            if(amu_pack_conv_input_rvv(&p,b,0,first,count,inputs,sizeof(inputs),&in) ||
               amu_matmul_packed_fp16(&w,&in,packed_output+first,129)) platform_exit(66);
        }
        for(size_t i=0;i<129*129;++i) {
            union {float f; uint32_t u;} x={.f=legacy[i]},y={.f=packed_output[i]};
            if(x.u!=y.u) platform_exit(67);
        }
    }
    if(amu_configure(32,32)) platform_exit(68);
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
            Conv2D p={1,k,1,n,m,1,1,1,1,0,0,1};
            for(size_t first=0;first<n;first+=32) {
                AmuPackedInput input;
                size_t count=n-first<32 ? n-first : 32;
                if(amu_pack_conv_input_rvv(&p,b,0,first,count,inputs,sizeof(inputs),&input)) platform_exit(7);
                uint64_t before=fingerprint(inputs,input.bytes/2);
                uint64_t calls=model_profile_phases[PROFILE_INPUT_PACK].calls;
                if(amu_matmul_packed_fp16(&descriptor,&input,packed_output+first,n) ||
                   before!=fingerprint(inputs,input.bytes/2) ||
                   calls!=model_profile_phases[PROFILE_INPUT_PACK].calls) platform_exit(8);
            }
            for(size_t i=0;i<m*n;++i) {
                union { float f; uint32_t u; } old={.f=legacy[i]},new={.f=packed_output[i]};
                if(old.u!=new.u) platform_exit(9);
            }
        }
    }
    /* Real windows, padding and spatial tails; compare old im2col+packing. */
    static const Conv2D convs[]={
        {1,3,5,7,16,3,3,1,1,1,1,1},{1,64,5,7,64,3,3,1,1,1,1,1},
        {1,128,5,7,33,3,3,2,2,1,1,1},{2,3,5,7,33,3,1,2,2,1,0,1}};
    for(unsigned test=0;test<sizeof(convs)/sizeof(convs[0]);++test) {
        const Conv2D *p=convs+test;
        size_t oh=(p->in_h+2*p->pad_h-p->kernel_h)/p->stride_h+1;
        size_t ow=(p->in_w+2*p->pad_w-p->kernel_w)/p->stride_w+1;
        size_t area=oh*ow,k=p->in_channels*p->kernel_h*p->kernel_w,m=p->out_channels;
        for(size_t i=0;i<m*k;++i) a[i]=(float)((int)(i%97)-48)/19.0f;
        for(size_t i=0;i<p->batch*p->in_channels*p->in_h*p->in_w;++i)
            b[i]=(float)((int)((i*13)%101)-50)/23.0f;
        AmuPackedWeights weight;
        if(amu_pack_weights(a,m,k,k,weights,sizeof(weights),&weight)) platform_exit(10);
        for(size_t batch=0;batch<p->batch;++batch) for(size_t first=0;first<area;first+=32) {
            size_t count=area-first<32 ? area-first : 32;
            for(size_t ic=0;ic<p->in_channels;++ic)
                for(size_t ky=0;ky<p->kernel_h;++ky) for(size_t kx=0;kx<p->kernel_w;++kx)
                    for(size_t col=0;col<count;++col) {
                        size_t pos=first+col,inner=(ic*p->kernel_h+ky)*p->kernel_w+kx;
                        int y=(int)(pos/ow*p->stride_h+ky)-(int)p->pad_h;
                        int x=(int)(pos%ow*p->stride_w+kx)-(int)p->pad_w;
                        columns[inner*32+col]=(y>=0 && x>=0 && y<(int)p->in_h && x<(int)p->in_w) ?
                            b[((batch*p->in_channels+ic)*p->in_h+(size_t)y)*p->in_w+(size_t)x] : 0.0f;
                    }
            AmuPackedInput input;
            if(amu_matmul_fp16(a,columns,legacy,m,count,k,k,32,35) ||
               amu_pack_conv_input_rvv(p,b,batch,first,count,inputs,sizeof(inputs),&input) ||
               amu_matmul_packed_fp16(&weight,&input,packed_output,35)) platform_exit(11);
            for(size_t row=0;row<m;++row) for(size_t col=0;col<count;++col) {
                size_t at=row*35+col;
                union { float f; uint32_t u; } old={.f=legacy[at]},new={.f=packed_output[at]};
                if(old.u!=new.u) platform_exit(12);
                float bias=(float)((int)(row%7)-3)/13.0f;
                old.f+=bias; new.f+=bias;
                if(old.u!=new.u) platform_exit(13);
            }
        }
    }
    host_call(4,(uintptr_t)"PREPACKED AMU PASS: RVV bits/94 layouts, 11 matrix shapes x2, 4 convolution shapes\n");
    platform_exit(0);
}
