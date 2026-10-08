#include "cache.h"
#include "diagnostics.h"
#include <xewmatrix_intrinsic.h>

/* Direct bypass-to-bypass handoffs: no scalar repacking or diagnostic calls
 * between producer and consumer. All reference values are exact integers. */
enum { HALF_COUNT=7*32, FLOAT_COUNT=7*16, ROUNDS=32 };
static uint16_t seed_a[HALF_COUNT] __attribute__((aligned(64)));
static uint16_t seed_b[HALF_COUNT] __attribute__((aligned(64)));
static uint16_t tile_a[HALF_COUNT] __attribute__((aligned(64)));
static uint16_t tile_b[HALF_COUNT] __attribute__((aligned(64)));
static float matrix_out[FLOAT_COUNT] __attribute__((aligned(64)));
static float vector_out[FLOAT_COUNT] __attribute__((aligned(64)));

static int value(unsigned round,unsigned row,unsigned k,unsigned side) {
    return (int)((round*3+row*5+k*(side ? 3 : 2)+side)%7)-3;
}
static uint16_t half_integer(int x) {
    const uint16_t magnitude[]={0,0x3c00,0x4000,0x4200};
    return magnitude[x<0 ? -x : x] | (x<0 ? 0x8000 : 0);
}
static void vector_copy_half(uint16_t *dst,const uint16_t *src,size_t count) {
    while(count) {
        size_t vl;
        __asm__ volatile("vsetvli %0,%3,e16,m1,ta,ma\n"
                         "vle16.v v8,(%1)\nvse16.v v8,(%2)"
                         : "=&r"(vl) : "r"(src),"r"(dst),"r"(count) : "v8","memory");
        src+=vl; dst+=vl; count-=vl;
    }
}
static void vector_add(float *dst,const float *src,size_t count,float tag) {
    while(count) {
        size_t vl;
        __asm__ volatile("vsetvli %0,%3,e32,m1,ta,ma\n"
                         "vle32.v v8,(%1)\nvfadd.vf v8,v8,%4\nvse32.v v8,(%2)"
                         : "=&r"(vl) : "r"(src),"r"(dst),"r"(count),"f"(tag)
                         : "v8","memory");
        src+=vl; dst+=vl; count-=vl;
    }
}
static int mismatch(unsigned buffer,unsigned index,uint32_t got,uint32_t want) {
    board_diag.tensor=buffer;
    board_diag.element=index;
    board_diag.observed=got;
    board_diag.expected=want;
    board_diag_error(4,-100-(int)buffer);
    return -100-(int)buffer;
}

int board_sync_selftest(void) {
    const unsigned shapes[][3]={{3,5,17},{1,1,1},{2,3,32},{3,5,31}};
    for(unsigned round=0;round<ROUNDS;++round) {
        unsigned m=shapes[round%4][0],n=shapes[round%4][1],k=shapes[round%4][2];
        board_diag.frame=round;
        board_diag.phase=PHASE_COMPUTE;
        board_diag_commit();
        /* Dirty scalar lines include row padding. Flush must preserve it. */
        for(unsigned i=0;i<HALF_COUNT;++i) {
            seed_a[i]=seed_b[i]=0;
            tile_a[i]=tile_b[i]=0x3555;
        }
        for(unsigned i=0;i<FLOAT_COUNT;++i) matrix_out[i]=vector_out[i]=-1234;
        for(unsigned r=0;r<m;++r)
            for(unsigned j=0;j<k;++j) seed_a[32+r*32+j]=half_integer(value(round,r,j,0));
        for(unsigned r=0;r<n;++r)
            for(unsigned j=0;j<k;++j) seed_b[32+r*32+j]=half_integer(value(round,r,j,1));
        board_clean(seed_a,sizeof(seed_a));
        board_clean(seed_b,sizeof(seed_b));
        board_prepare_write(tile_a,sizeof(tile_a));
        board_prepare_write(tile_b,sizeof(tile_b));
        board_prepare_write(matrix_out,sizeof(matrix_out));
        board_prepare_write(vector_out,sizeof(vector_out));
        for(unsigned r=0;r<m;++r) vector_copy_half(tile_a+32+r*32,seed_a+32+r*32,k);
        for(unsigned r=0;r<n;++r) vector_copy_half(tile_b+32+r*32,seed_b+32+r*32,k);
        board_fence(); /* RVV stores -> AMU loads (RAW). */
        uintptr_t old,flags;
        __riscv_msettilem(old,m,w);
        __riscv_msettilen(old,n,w);
        __riscv_msettilek(old,k,w);
        __riscv_mzero("acc0");
        __riscv_mlae16("tr0",tile_a+32,64);
        __riscv_mlbe16("tr1",tile_b+32,64);
        __riscv_mfmacc_s_h("acc0","tr1","tr0");
        __riscv_msce32("acc0",matrix_out+16,64);
        /* Same N550 completion convention as amu.c; FENCE is not a generic
         * accelerator-completion primitive. Hardware must validate both. */
        __asm__ volatile("csrr %0,xmfflags" : "=r"(flags) : : "memory");
        board_fence(); /* AMU stores -> RVV loads (RAW). */
        for(unsigned r=0;r<m;++r)
            vector_add(vector_out+16+r*16,matrix_out+16+r*16,n,(float)(round+1));
        board_fence(); /* Finish RVV accesses before scalar checks/reuse. */
        board_finish_write(tile_a,sizeof(tile_a));
        board_finish_write(tile_b,sizeof(tile_b));
        board_finish_write(matrix_out,sizeof(matrix_out));
        board_finish_write(vector_out,sizeof(vector_out));
        board_diag.amu_flags=flags;
        board_diag.phase=PHASE_CHECK;
        board_diag_commit();
        (void)old;
        for(unsigned i=0;i<HALF_COUNT;++i) {
            unsigned row=i/32,col=i%32;
            uint16_t a=row>=1 && row<=m && col<k ? half_integer(value(round,row-1,col,0)) : 0x3555;
            uint16_t b=row>=1 && row<=n && col<k ? half_integer(value(round,row-1,col,1)) : 0x3555;
            if(tile_a[i]!=a) return mismatch(0,i,tile_a[i],a);
            if(tile_b[i]!=b) return mismatch(1,i,tile_b[i],b);
        }
        for(unsigned i=0;i<FLOAT_COUNT;++i) {
            unsigned row=i/16,col=i%16;
            int active=row>=1 && row<=m && col<n;
            int expected=0;
            if(active)
                for(unsigned j=0;j<k;++j) expected+=value(round,row-1,j,0)*value(round,col,j,1);
            union { float f; uint32_t u; } c={.f=matrix_out[i]},v={.f=vector_out[i]};
            union { float f; uint32_t u; } want_c={.f=active ? (float)expected : -1234.0f};
            union { float f; uint32_t u; } want_v={.f=active ? (float)(expected+(int)round+1) : -1234.0f};
            if(c.u!=want_c.u) return mismatch(2,i,c.u,want_c.u);
            if(v.u!=want_v.u) return mismatch(3,i,v.u,want_v.u);
        }
    }
    board_diag.frame=0;
    board_diag_commit();
    return 0;
}
