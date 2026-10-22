/* Candidate arithmetic has its own reference; it need not equal unfused math. */
static float scalar_fma(float a,float b,float c) {
    float out;
    __asm__ volatile("fmadd.s %0,%1,%2,%3" : "=f"(out) : "f"(a),"f"(b),"f"(c));
    return out;
}
static void candidate_cases(void) {
    float *a=(float *)ram.data,*b=a+4096,*out=b+16384,*expected=(float *)old.data;
    static const size_t inner[]={1,16,31,32,33,169};
    __asm__ volatile("csrw fcsr,zero" ::: "memory");
    rvv_compute_flags=RVV_FMA;
    for(unsigned shape=0;shape<sizeof(inner)/sizeof(inner[0]);++shape) {
        size_t k=inner[shape],m=5,n=37,lda=k+1,ldb=n+3,ldc=n+5;
        for(size_t i=0;i<m*lda;++i) a[i]=value(i,0);
        for(size_t i=0;i<k*ldb;++i) b[i]=value(i+7,0);
        for(size_t i=0;i<m*ldc;++i) out[i]=expected[i]=-123.0f;
        for(size_t r=0;r<m;++r) for(size_t c=0;c<n;++c) {
            float sum=0.0f;
            for(size_t t=0;t<k;++t) sum=scalar_fma(a[r*lda+t],b[t*ldb+c],sum);
            expected[r*ldc+c]=sum;
        }
        rvv_matmul_ordered(a,b,out,m,n,k,lda,ldb,ldc);
        for(size_t i=0;i<m*ldc;++i) require(bits(out[i])==bits(expected[i]),30,i,bits(out[i]),bits(expected[i]));
    }
    Conv2D p={.batch=2,.in_channels=3,.in_h=5,.in_w=37,.out_channels=3,
        .kernel_h=3,.kernel_w=3,.stride_h=1,.stride_w=1,.pad_h=1,.pad_w=1,.groups=3};
    float *bias=b+64;
    for(size_t i=0;i<1110;++i) a[i]=value(i,0);
    for(size_t i=0;i<27;++i) b[i]=value(i+5,0);
    for(size_t i=0;i<3;++i) bias[i]=value(i+2,0);
    for(size_t batch=0;batch<2;++batch) for(size_t ch=0;ch<3;++ch)
        for(size_t y=0;y<5;++y) for(size_t x=0;x<37;++x) {
            float sum=0.0f;
            for(size_t ky=0;ky<3;++ky) for(size_t kx=0;kx<3;++kx) {
                size_t iy=y+ky-1,ix=x+kx-1;
                float v=iy<5 && ix<37 ? a[(batch*3+ch)*185+iy*37+ix] : 0.0f;
                sum=scalar_fma(v,b[ch*9+ky*3+kx],sum);
            }
            expected[(batch*3+ch)*185+y*37+x]=sum+bias[ch];
        }
    require(!rvv_depthwise3x3(&p,a,b,bias,out),31,0,0,0);
    for(size_t i=0;i<1110;++i) require(bits(out[i])==bits(expected[i]),32,i,bits(out[i]),bits(expected[i]));
    rvv_compute_flags=RVV_REDUCE;
    for(unsigned shape=0;shape<sizeof(inner)/sizeof(inner[0]);++shape) {
        size_t n=inner[shape];
        for(size_t i=0;i<3*n;++i) a[i]=17.0f;
        rvv_softmax(a,out,3,n);
        volatile float denominator=(float)n;
        float want=1.0f/denominator;
        for(size_t i=0;i<3*n;++i) require(bits(out[i])==bits(want),33,i,bits(out[i]),bits(want));
    }
    rvv_compute_flags=RVV_BLOCK_TRANSPOSE;
    for(size_t i=0;i<2*33*35;++i) a[i]=value(i,0);
    rvv_transpose(a,out,2,33,35);
    for(size_t batch=0;batch<2;++batch) for(size_t r=0;r<33;++r) for(size_t c=0;c<35;++c)
        require(bits(out[batch*1155+c*33+r])==bits(a[batch*1155+r*35+c]),34,r*35+c,0,0);
    rvv_compute_flags=0;
    for(size_t i=0;i<3*37;++i) out[i]=expected[i]=value(i,0);
    rvv_fuse_bias_silu=0;
    require(!rvv_bias_channels(expected,bias,3,37),35,0,0,0);
    rvv_nonlinear(expected,expected,111,1);
    rvv_fuse_bias_silu=1;
    require(!rvv_bias_channels(out,bias,3,37),36,0,0,0);
    for(size_t i=0;i<111;++i) require(bits(out[i])==bits(expected[i]),37,i,bits(out[i]),bits(expected[i]));
    rvv_fuse_bias_silu=0;
    print("CANDIDATES PASS FMA BMM/DW, reduction, transpose, fused bias/SiLU\n");
}
