/* Linux x86_64 numerical auditor. Never linked into board firmware.
 * Non-AMU operations use our existing scalar definitions, validated against
 * every frozen QEMU tensor. Mode 16 reproduces the finite-window RTL PE;
 * mode 32 preserves the independent historical QEMU comparison. */
#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>
#include "scalar_model.h"
#include "model_generated.h"
#include "conv2d.h"
typedef __int128 I128;
typedef unsigned __int128 U128;
static unsigned mode;
static uint32_t word(float f) { uint32_t u; memcpy(&u,&f,4); return u; }
static float number(uint32_t u) { float f; memcpy(&f,&u,4); return f; }
static int half(float f,uint16_t *out) {
    uint32_t u=word(f),s=(u>>16)&0x8000,e=(u>>23)&255,t=u&0x7fffff;
    if(e==255 || e>142) return -1;
    if(e<102) { *out=s; return 0; }
    if(e<113) {
        unsigned shift=126-e; uint32_t m=t|0x800000,q=m>>shift,r=m&((1u<<shift)-1),h=1u<<(shift-1);
        *out=s|(q+(r>h || (r==h && (q&1)))); return 0;
    }
    uint32_t q=((e-112)<<10)|(t>>13),r=t&8191;
    q+=r>4096 || (r==4096 && (q&1));
    if(q>=0x7c00) return -1;
    *out=s|q; return 0;
}
static int64_t fixed(uint16_t h) {
    unsigned e=(h>>10)&31;
    int64_t n=e ? (int64_t)(1024+(h&1023))<<(e-1) : h&1023;
    return h&0x8000 ? -n : n;
}
static unsigned length(U128 n) {
    uint64_t high=(uint64_t)(n>>64);
    return high ? 128u-(unsigned)__builtin_clzll(high) : 64u-(unsigned)__builtin_clzll((uint64_t)n);
}
/* FP16 products share the exact quantum 2^-48. K<=4096 bounds the sum
 * below 2^93 in these units. Rounded accumulators stay on the same grid. */
static I128 round_fixed(I128 n) {
    if(!n) return 0;
    U128 a=n<0 ? (U128)-n : (U128)n;
    unsigned bits=length(a);
    if(bits>24) {
        unsigned shift=bits-24;
        U128 q=a>>shift,r=a-((a>>shift)<<shift),h=(U128)1<<(shift-1);
        q+=r>h || (r==h && (q&1)); a=q<<shift;
    }
    return n<0 ? -(I128)a : (I128)a;
}
/* Finite FP16 path of pe_top.sv at AMU e9c0aa3.
 * P2: 29 main fractional bits + 5 residual bits; signed tails.
 * P4/P5: normalize to bit 34. RTL does not jam right-shifted-out
 * normalization bits into sticky. NaN/Inf/BF16 are outside this capture.
 * Integers use the exact FP16-product grid, 2^-48. */
static int half_exp_fixed(int64_t v) {
    uint64_t a=v<0 ? (uint64_t)-v : (uint64_t)v;
    int e=63-__builtin_clzll(a)-24;
    return e < -14 ? -14 : e;
}
static I128 pe_round(I128 n,int tail) {
    if(!n) return 0;
    U128 a=n<0 ? (U128)-n : (U128)n;
    unsigned bits=length(a);
    if(bits>24) {
        unsigned shift=bits-24;
        U128 q=a>>shift,r=a-(q<<shift),h=(U128)1<<(shift-1);
        q+=r>h || (r==h && (tail || (q&1)));
        a=q<<shift;
    }
    return n<0 ? -(I128)a : (I128)a;
}
static I128 pe16_step(const int64_t *w,const int64_t *x,size_t stride,size_t count,I128 acc) {
    I128 products[17]; int exponents[17]; size_t used=0;
    int maximum=-1000;
    for(size_t j=0;j<count;++j) {
        int64_t a=w[j],b=x[j*stride];
        if(!a || !b) continue;
        products[used]=(I128)a*b;
        exponents[used]=half_exp_fixed(a)+half_exp_fixed(b);
        if(exponents[used]>maximum) maximum=exponents[used];
        ++used;
    }
    if(acc) {
        products[used]=acc;
        U128 a=acc<0 ? (U128)-acc : (U128)acc;
        exponents[used]=(int)length(a)-1-48;
        if(exponents[used]>maximum) maximum=exponents[used];
        ++used;
    }
    if(!used) return 0;
    int shift=maximum+48-34;
    unsigned quant_shift=shift>0 ? (unsigned)shift : 0;
    U128 mask=((U128)1<<quant_shift)-1;
    I128 sum=0; int pos_tail=0,neg_tail=0;
    for(size_t j=0;j<used;++j) {
        I128 n=products[j]; U128 a=n<0 ? (U128)-n : (U128)n;
        int diff=maximum-exponents[j];
        U128 q=a>>quant_shift;
        int tail=(a&mask)!=0;
        /* P2 drops main data for exponent difference >29, saturates
         * the 5-bit residual if it overflows, and keeps tail at >=37. */
        if(diff>=37) { q=0; tail=1; }
        else if(diff>29 && q>31) { q=31; tail=1; }
        sum+=n<0 ? -(I128)q : (I128)q;
        if(tail) { if(n<0) neg_tail=1; else pos_tail=1; }
    }
    sum*=((I128)1<<quant_shift);
    if(!sum) return 0;
    U128 mag=sum<0 ? (U128)-sum : (U128)sum;
    unsigned bits=length(mag);
    unsigned discarded=bits>35 ? bits-35 : 0;
    mag=(mag>>discarded)<<discarded;
    I128 truncated=sum<0 ? -(I128)mag : (I128)mag;
    int tail=sum<0 ? neg_tail&&!pos_tail : pos_tail&&!neg_tail;
    return pe_round(truncated,tail);
}

static float fixed_float(I128 n) {
    if(!n) return 0.0f;
    U128 a=n<0 ? (U128)-n : (U128)n;
    unsigned bits=length(a);
    U128 mantissa=bits>24 ? a>>(bits-24) : a<<(24-bits);
    return number((n<0 ? 0x80000000u : 0)|((bits-1-48+127)<<23)|((uint32_t)mantissa&0x7fffff));
}
void matmul_fp32_scalar(const float *a,const float *b,float *c,size_t m,size_t n,size_t k,size_t lda,size_t ldb,size_t ldc) {
    for(size_t r=0;r<m;++r) for(size_t col=0;col<n;++col) {
        float sum=0;
        for(size_t i=0;i<k;++i) { float term=a[r*lda+i]*b[i*ldb+col]; sum=sum+term; }
        c[r*ldc+col]=sum;
    }
}
int conv2d_output_shape(const Conv2D *p,size_t *h,size_t *w) {
    if(!p->groups || !p->stride_h || !p->stride_w || p->in_channels%p->groups || p->out_channels%p->groups) return -1;
    *h=(p->in_h+2*p->pad_h-p->kernel_h)/p->stride_h+1;
    *w=(p->in_w+2*p->pad_w-p->kernel_w)/p->stride_w+1; return 0;
}
int conv2d_fp32_scalar(const Conv2D *p,const float *x,const float *w,const float *b,float *out) {
    size_t oh,ow; if(conv2d_output_shape(p,&oh,&ow)) return -1;
    size_t icg=p->in_channels/p->groups,ocg=p->out_channels/p->groups;
    for(size_t n=0;n<p->batch;++n) for(size_t oc=0;oc<p->out_channels;++oc)
        for(size_t y=0;y<oh;++y) for(size_t col=0;col<ow;++col) {
            float sum=0;
            for(size_t ic=0;ic<icg;++ic) for(size_t ky=0;ky<p->kernel_h;++ky) for(size_t kx=0;kx<p->kernel_w;++kx) {
                long iy=(long)(y*p->stride_h+ky)-p->pad_h,ix=(long)(col*p->stride_w+kx)-p->pad_w;
                float v=0;
                if(iy>=0 && ix>=0 && (size_t)iy<p->in_h && (size_t)ix<p->in_w)
                    v=x[((n*p->in_channels+oc/ocg*icg+ic)*p->in_h+(size_t)iy)*p->in_w+(size_t)ix];
                float term=v*w[((oc*icg+ic)*p->kernel_h+ky)*p->kernel_w+kx]; sum=sum+term;
            }
            out[((n*p->out_channels+oc)*oh+y)*ow+col]=sum+b[oc];
        }
    return 0;
}
static int amu_conv(const ModelNode *node,const ModelTensor *t,unsigned char *arena) {
    const ModelTensor *xt=t+node->in[0],*wt=t+node->in[1],*yt=t+node->out[0];
    const float *x=(float *)(arena+xt->offset),*w=(float *)(arena+wt->offset),*bias=(float *)(arena+t[node->in[2]].offset);
    float *out=(float *)(arena+yt->offset);
    size_t ci=xt->shape[1],ih=xt->shape[2],iw=xt->shape[3],co=wt->shape[0],kh=wt->shape[2],kw=wt->shape[3];
    size_t oh=yt->shape[2],ow=yt->shape[3],area=oh*ow,k=ci*kh*kw;
    if(!k || k>4096 || area>SIZE_MAX/k/sizeof(int64_t)) return -1;
    int64_t *weights=malloc(co*k*sizeof(int64_t)),*patch=malloc(k*area*sizeof(int64_t));
    if(!weights || !patch) { free(weights); free(patch); return -2; }
    int result=0;
    for(size_t i=0;i<co*k;++i) { uint16_t h; if(half(w[i],&h)) { result=-3; goto done; } weights[i]=fixed(h); }
    for(size_t n=0;n<xt->shape[0];++n) {
        for(size_t ic=0;ic<ci;++ic) for(size_t ky=0;ky<kh;++ky) for(size_t kx=0;kx<kw;++kx)
            for(size_t pos=0;pos<area;++pos) {
                long iy=(long)((pos/ow)*node->p[0]+ky)-node->p[2],ix=(long)((pos%ow)*node->p[1]+kx)-node->p[3];
                float v=0;
                if(iy>=0 && ix>=0 && (size_t)iy<ih && (size_t)ix<iw) v=x[((n*ci+ic)*ih+(size_t)iy)*iw+(size_t)ix];
                uint16_t h; if(half(v,&h)) { result=-3; goto done; }
                patch[((ic*kh+ky)*kw+kx)*area+pos]=fixed(h);
            }
        for(size_t oc=0;oc<co;++oc) for(size_t pos=0;pos<area;++pos) {
            I128 exact=0; float accumulator=0;
            for(size_t first=0;first<k;first+=mode) {
                size_t end=first+mode<k ? first+mode : k;
                if(mode==16) {
                    exact=pe16_step(weights+oc*k+first,patch+first*area+pos,area,end-first,exact);
                } else {
                    double sum=0;
                    for(size_t j=first;j<end;++j) sum+=(double)weights[oc*k+j]*(double)patch[j*area+pos];
                    accumulator=(float)((double)accumulator+sum*0x1p-48);
                }
            }
            if(mode==16) accumulator=fixed_float(exact);
            out[(n*co+oc)*area+pos]=accumulator+bias[oc];
        }
    }
done: free(weights); free(patch); return result;
}
static int transfer(void *p,size_t bytes,int writing) {
    return (writing ? fwrite(p,1,bytes,stdout) : fread(p,1,bytes,stdin))==bytes ? 0 : -1;
}
static int selftest(void) {
    I128 one=(I128)1<<48,halfway=(I128)1<<24;
    if(word(fixed_float(round_fixed(one+halfway)))!=0x3f800000 ||
       word(fixed_float(round_fixed(one+halfway+1)))!=0x3f800001 ||
       word(fixed_float(round_fixed(-one-halfway-1)))!=0xbf800001) return 1;
    /* Captured node 2: losing normalization sticky changes c084d0ab
     * into c084d0aa. Preserve RTL behavior independently of exact RNE. */
    const uint16_t ah[16]={0x3995,0x3e29,0x3e55,0x3fc4,0x3f6f,0x418d,0x4327,0x3ef5,0x418a,0x4358,0x8004,0x801b,0x8013,0x434f,0x3edc,0x3f09},bh[16]={0x2db6,0xa06d,0x9e66,0xa0a6,0xa0d5,0x1e11,0x2cce,0x1a41,0xa9fb,0xa94c,0x9d18,0x3218,0xad5b,0xb4b7,0xb93d,0x20d9};
    int64_t aa[16],bb[16];
    for(unsigned i=0;i<16;++i) { aa[i]=fixed(ah[i]); bb[i]=fixed(bh[i]); }
    if(word(fixed_float(pe16_step(aa,bb,1,16,(I128)-557377971552256LL)))!=0xc084d0aa) return 1;
    puts("{\"version\":1,\"arch\":\"linux-x86_64\",\"nodes\":334,\"operators\":21,\"selftest\":\"passed\"}"); return 0;
}
int main(int argc,char **argv) {
    if(fesetround(FE_TONEAREST)) return 2;
    _mm_setcsr(_mm_getcsr()&~((1u<<15)|(1u<<6))); /* gradual underflow */
    if(argc==2 && !strcmp(argv[1],"--selftest")) return selftest();
    if(argc!=2 || strcmp(argv[1],"--serve")) return 2;
    if(selftest()) return 3;
    fflush(stdout);
    for(;;) {
        uint32_t header[4];
        if(fread(header,1,sizeof(header),stdin)!=sizeof(header)) return feof(stdin) ? 0 : 4;
        if(header[0]!=0x31415259 || header[1]>=334 || (header[2]!=16 && header[2]!=32)) return 5;
        const ModelNode *node=model_nodes+header[1];
        if(header[3]!=node->inputs) return 6;
        mode=header[2]; ModelTensor tensors[MODEL_TENSOR_COUNT]; memcpy(tensors,model_tensors,sizeof(tensors));
        size_t bytes=64;
        for(unsigned i=0;i<node->inputs;++i) bytes+=(model_tensors[node->in[i]].bytes+63)&~(size_t)63;
        for(unsigned i=0;i<node->outputs;++i) if(node->out[i]>=0) bytes+=(model_tensors[node->out[i]].bytes+63)&~(size_t)63;
        unsigned char *arena=calloc(1,bytes); if(!arena) return 7;
        size_t at=64;
        for(unsigned i=0;i<node->inputs;++i) {
            uint32_t descriptor[2]; if(transfer(descriptor,8,0)) return 8;
            int id=node->in[i]; ModelTensor *t=tensors+id;
            if(descriptor[0]!=(uint32_t)id || descriptor[1]!=t->bytes) return 9;
            t->constant=0; t->offset=at;
            if(transfer(arena+at,t->bytes,0)) return 10;
            if(t->dtype==1) for(size_t j=0;j<t->count;++j) if((word(((float *)(arena+at))[j])&0x7f800000)==0x7f800000) return 11;
            at+=(t->bytes+63)&~(size_t)63;
        }
        for(unsigned i=0;i<node->outputs;++i) if(node->out[i]>=0) {
            ModelTensor *t=tensors+node->out[i]; t->constant=0; t->offset=at; at+=(t->bytes+63)&~(size_t)63;
        }
        int status=node->op==OP_CONV && node->p[4]==1 ? amu_conv(node,tensors,arena) : scalar_model_node(node,tensors,arena,arena);
        uint32_t count=0; if(!status) for(unsigned i=0;i<node->outputs;++i) count+=node->out[i]>=0;
        int32_t response[2]={status,(int32_t)count}; if(transfer(response,8,1)) return 12;
        if(!status) for(unsigned i=0;i<node->outputs;++i) if(node->out[i]>=0) {
            int id=node->out[i]; uint32_t descriptor[2]={(uint32_t)id,tensors[id].bytes};
            if(transfer(descriptor,8,1) || transfer(arena+tensors[id].offset,tensors[id].bytes,1)) return 13;
        }
        free(arena); if(fflush(stdout)) return 14;
    }
}
