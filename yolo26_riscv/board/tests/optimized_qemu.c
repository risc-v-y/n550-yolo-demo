/* Actual v5/v6 dispatchers side by side, frozen non-convolution shapes.
 * QEMU RAM does not prove physical CLP/cache coherency or performance. */
#include "../../accel_model.h"
#include "../../rvv_model_ops.h"
#include "../memory.h"
#include "../diagnostics.h"
#include "../model_io.h"
#include "model_generated.h"
#include <stdint.h>

enum { CAPACITY = 6*1024*1024, GUARD = 64 };
typedef struct { unsigned char data[CAPACITY]; } Buffer;
static Buffer clp BOARD_CLP_DATA, ram __attribute__((aligned(64))), old __attribute__((aligned(64)));
volatile BoardDiagnostics board_diag;
ModelProfilePhase model_profile_phases[PROFILE_PHASE_COUNT];
void board_diag_phase(unsigned phase) { (void)phase; }
int board_pack_begin(size_t first,size_t columns,size_t k,const void *input,
                     const void *cache,void *output,size_t cache_bytes) {
    (void)first; (void)columns; (void)k; (void)input; (void)cache;
    (void)output; (void)cache_bytes; return 0;
}
int board_pack_mark(unsigned stage,size_t column,size_t inner,int64_t row,
                    size_t vl,const size_t tags[3],uint64_t converted) {
    (void)stage; (void)column; (void)inner; (void)row; (void)vl;
    (void)tags; (void)converted; return -1;
}
extern int accel_model_node_v5(const ModelNode *, const ModelTensor *, unsigned char *, const unsigned char *);
extern uint32_t board_crc32_v5(const void *, size_t);
static unsigned cases, current_node, current_frame, current_mode;
void *memset(void *p, int value, size_t bytes) {
    unsigned char *dest=p;
    for (size_t i=0;i<bytes;++i) dest[i]=(unsigned char)value;
    return p;
}

static uintptr_t host_call(uintptr_t op, uintptr_t arg) {
    register uintptr_t a0 __asm__("a0")=op, a1 __asm__("a1")=arg;
    __asm__ volatile(".option push\n.option norvc\n.balign 16\nslli zero,zero,31\n"
                     "ebreak\nsrai zero,zero,7\n.option pop" : "+r"(a0), "+r"(a1) :: "memory");
    return a0;
}
static void print(const char *text) { host_call(4, (uintptr_t)text); }
static void hex(uint64_t value) {
    char text[19]="0x0000000000000000";
    for (unsigned i=0; i<16; ++i) text[17-i]="0123456789abcdef"[(value>>(4*i))&15];
    print(text);
}
__attribute__((noreturn)) void platform_exit(uintptr_t status) {
    uintptr_t args[2]={0x20026,status}; host_call(0x20,(uintptr_t)args);
    for (;;) __asm__ volatile("wfi");
}
void platform_trap(uintptr_t cause, uintptr_t pc, uintptr_t value, const uintptr_t *regs) {
    (void)regs; print("OPT TRAP "); hex(cause); hex(pc); hex(value); platform_exit(99);
}
static void require(int valid, unsigned stage, size_t at, uint64_t got, uint64_t expected) {
    if (valid) return;
    print("OPT FAIL node="); hex(current_node); print(" frame="); hex(current_frame);
    print(" memory="); hex(current_mode); print(" stage="); hex(stage);
    print(" at="); hex(at); print(" got="); hex(got); print(" expected="); hex(expected); print("\n");
    platform_exit(stage);
}
static uint32_t bits(float value) { union { float f; uint32_t u; } v={.f=value}; return v.u; }
static float value(size_t at, unsigned frame) {
    static const uint32_t fixed[]={0,0x80000000,0x3f800000,0xbf800000,0x3eaaaaab,0xbeaaaaab,
        1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f800001,0xbf7fffff};
    union { float f; uint32_t u; } v={.u=fixed[at%14]};
    return frame ? (float)((int)((at*1103515245u+12345u)>>8)%1023)/64.0f : v.f;
}
static int same_case(const ModelNode *a, const ModelNode *b) {
    if (a->op!=b->op || a->inputs!=b->inputs || a->outputs!=b->outputs) return 0;
    for (unsigned i=0;i<8;++i) if (a->p[i]!=b->p[i]) return 0;
    for (unsigned io=0;io<2;++io) for (unsigned i=0;i<(io?a->outputs:a->inputs);++i) {
        int ai=io?a->out[i]:a->in[i], bi=io?b->out[i]:b->in[i];
        if ((ai<0)!=(bi<0)) return 0;
        if (ai<0) continue;
        const ModelTensor *x=model_tensors+ai,*y=model_tensors+bi;
        if (x->rank!=y->rank || x->dtype!=y->dtype) return 0;
        for (unsigned d=0;d<x->rank;++d) if (x->shape[d]!=y->shape[d]) return 0;
    }
    return 1;
}
static void node_case(const ModelNode *source, unsigned frame, unsigned mode) {
    ModelTensor tensors[6]={{0}};
    ModelNode node=*source;
    size_t end=GUARD;
    for (unsigned io=0;io<2;++io) for (unsigned i=0;i<(io?node.outputs:node.inputs);++i) {
        int id=io?source->out[i]:source->in[i];
        if (id<0) continue;
        unsigned target=io?4+i:i;
        tensors[target]=model_tensors[id]; tensors[target].constant=0;
        tensors[target].offset=(uint32_t)end;
        if (io) node.out[i]=(int32_t)target; else node.in[i]=(int32_t)target;
        end=(end+tensors[target].bytes+GUARD+63)&~(size_t)63;
    }
    require(end<=CAPACITY,1,0,end,CAPACITY);
    for (size_t i=0;i<end;++i) old.data[i]=0xa5;
    for (unsigned slot=0;slot<node.inputs;++slot) {
        const ModelTensor *t=tensors+slot;
        for (size_t i=0;i<t->count;++i) {
            if (t->dtype==1) ((float *)(old.data+t->offset))[i]=value(i+slot*7,frame);
            else {
                int64_t selected=(int64_t)(i%24000);
                if (node.op==OP_GATHER && slot==1) {
                    const ModelTensor *from=tensors;
                    size_t inner=1;
                    for (size_t d=(size_t)node.p[0]+1;d<from->rank;++d) inner*=from->shape[d];
                    selected=(int64_t)((frame?i:i/inner)*37%from->shape[node.p[0]]);
                }
                ((int64_t *)(old.data+t->offset))[i]=selected;
            }
        }
    }
    unsigned char *actual=mode?clp.data:ram.data;
    for (size_t i=0;i<end;++i) actual[i]=old.data[i];
    current_frame=frame; current_mode=mode; ++cases;
    require(!accel_model_node_v5(&node,tensors,old.data,old.data),2,0,0,0);
    require(!accel_model_node(&node,tensors,actual,actual),3,0,0,0);
    for (size_t i=0;i<end;++i) require(actual[i]==old.data[i],4,i,actual[i],old.data[i]);
}
static void binary_edges(void) {
    float *a=(float *)ram.data,*b=a+256,*dest=b+256,*expected=dest+256;
    static const size_t counts[]={1,3,7,8,17,31,32,33,63,65,169};
    for (unsigned frm=0;frm<5;++frm) {
        __asm__ volatile("csrw frm,%0" :: "r"(frm):"memory");
        for (unsigned shape=0;shape<sizeof(counts)/sizeof(counts[0]);++shape) {
            size_t n=counts[shape];
            for (unsigned op=0;op<3;++op) for (unsigned scalar=0;scalar<2;++scalar) {
                for (size_t i=0;i<n;++i) { a[i]=value(i,0); b[i]=value(i+3,0); }
                for (size_t i=0;i<n;++i) {
                    volatile float x=a[i],y=b[scalar?0:i];
                    expected[i]=op==0?x+y:op==1?x-y:x*y;
                }
                require(!rvv_binary_periodic(a,b,dest,n,scalar?1:n,op),5,0,0,0);
                for (size_t i=0;i<n;++i) require(bits(dest[i])==bits(expected[i]),6,i,bits(dest[i]),bits(expected[i]));
                require(!rvv_binary_periodic(a,b,a,n,scalar?1:n,op),7,0,0,0);
                for (size_t i=0;i<n;++i) require(bits(a[i])==bits(expected[i]),8,i,bits(a[i]),bits(expected[i]));
            }
        }
    }
    __asm__ volatile("csrw fcsr,zero" ::: "memory");
    require(rvv_binary_periodic(a,b,a+1,32,32,0)==-1,9,0,0,0);
    require(rvv_binary_periodic(a,b,dest,33,2,0)==-1,10,0,0,0);
}
static void bias_cases(void) {
    for(unsigned node=0;node<MODEL_NODE_COUNT;++node) {
        const ModelNode *n=model_nodes+node;
        if(n->op!=OP_CONV || n->p[4]!=1) continue;
        const ModelTensor *t=model_tensors+n->out[0];
        size_t channels=t->shape[1],columns=t->shape[2]*t->shape[3];
        size_t bias_at=(GUARD+t->bytes+GUARD+63)&~(size_t)63;
        size_t end=bias_at+channels*4+GUARD;
        require(end<=CAPACITY,17,node,0,0);
        for(unsigned mode=0;mode<2;++mode) for(unsigned frame=0;frame<2;++frame) {
            unsigned char *data=mode ? clp.data : ram.data;
            for(size_t i=0;i<end;++i) data[i]=old.data[i]=0xa5;
            float *dest=(float *)(data+GUARD),*expected=(float *)(old.data+GUARD);
            float *bias=(float *)(data+bias_at);
            for(size_t c=0;c<channels;++c) bias[c]=value(c+7,frame);
            for(size_t i=0;i<t->count;++i) {
                dest[i]=value(i,frame);
                volatile float x=dest[i],y=bias[i/columns];
                expected[i]=x+y;
            }
            for(size_t i=bias_at;i<end;++i) old.data[i]=data[i];
            require(!rvv_bias_channels(dest,bias,channels,columns),18,node,0,0);
            for(size_t i=0;i<end;++i) require(data[i]==old.data[i],19,i,data[i],old.data[i]);
        }
    }
    float *dest=(float *)(ram.data+64),*bias=(float *)(ram.data+2048);
    for(unsigned frm=0;frm<5;++frm) {
        __asm__ volatile("csrw frm,%0" :: "r"((uintptr_t)frm) : "memory");
        for(size_t c=0;c<3;++c) bias[c]=value(c,0);
        for(size_t i=0;i<111;++i) {
            dest[i]=value(i+3,0);
            volatile float x=dest[i],y=bias[i/37];
            ((float *)old.data)[i]=x+y;
        }
        require(!rvv_bias_channels(dest,bias,3,37),20,frm,0,0);
        for(size_t i=0;i<111;++i) require(bits(dest[i])==bits(((float *)old.data)[i]),21,i,0,0);
    }
    __asm__ volatile("csrw fcsr,zero" ::: "memory");
    require(rvv_bias_channels(dest,dest+1,3,37)==-1,22,0,0,0);
    require(rvv_bias_channels(dest,bias,SIZE_MAX,2)==-1,23,0,0,0);
    require(rvv_bias_channels(dest,bias,3,0)==-1,24,0,0,0);
}

static void topk_edges(void) {
    ModelTensor tensors[3]={{.offset=64,.rank=2,.dtype=1,.shape={1,67}},
        {.offset=1024,.rank=2,.dtype=1,.shape={1,1}},
        {.offset=4096,.rank=2,.dtype=2,.shape={1,1}}};
    ModelNode node={.op=OP_TOPK,.inputs=1,.outputs=2,.in={0},.out={1,2},.p={1,1}};
    for (unsigned sample=0;sample<7;++sample) for (unsigned k=1;k<=67;k+=11) {
        tensors[0].count=67; tensors[0].bytes=268;
        tensors[1].count=tensors[2].count=k; tensors[1].bytes=k*4; tensors[2].bytes=k*8;
        tensors[1].shape[1]=tensors[2].shape[1]=k; node.p[1]=(int32_t)k;
        for (size_t i=0;i<6000;++i) old.data[i]=ram.data[i]=0xa5;
        for (unsigned i=0;i<67;++i) {
            union {float f;uint32_t u;} v={.f=sample==0?1.0f:sample==1?(float)i:sample==2?-(float)i:value(i,0)};
            if (sample==4 && i%7==0) v.u=0x7f800000;
            if (sample==5 && i%7==0) v.u=0xff800000;
            if (sample==6 && i%7==0) v.u=0x7fc00017;
            ((float *)(old.data+64))[i]=((float *)(ram.data+64))[i]=v.f;
        }
        require(!accel_model_node_v5(&node,tensors,old.data,old.data),11,k,0,0);
        require(!accel_model_node(&node,tensors,ram.data,ram.data),12,k,0,0);
        for (size_t i=0;i<6000;++i) require(old.data[i]==ram.data[i],13,i,ram.data[i],old.data[i]);
    }
}
static void crc_cases(void) {
    for (size_t i=0;i<65540;++i) ram.data[i]=(unsigned char)(i*97u+(i>>8));
    for (unsigned offset=0;offset<4;++offset) {
        for (size_t length=0;length<256;++length)
            require(board_crc32(ram.data+offset,length)==board_crc32_v5(ram.data+offset,length),14,length,0,0);
        require(board_crc32(ram.data+offset,65536)==board_crc32_v5(ram.data+offset,65536),15,0,0,0);
    }
    require(board_crc32("123456789",9)==0xcbf43926u,16,0,0,0);
}
int main(void) {
    unsigned shapes=0;
    for (unsigned i=0;i<MODEL_NODE_COUNT;++i) {
        const ModelNode *node=model_nodes+i;
        if (node->op==OP_CONV) continue;
        int duplicate=0;
        for (unsigned j=0;j<i;++j) if (same_case(node,model_nodes+j)) duplicate=1;
        if (duplicate) continue;
        current_node=i; ++shapes;
        for (unsigned frame=0;frame<2;++frame) for (unsigned mode=0;mode<2;++mode) node_case(node,frame,mode);
    }
    binary_edges(); bias_cases(); topk_edges(); crc_cases();
    print("OPTIMIZED PASS frozen unique non-conv shapes="); hex(shapes); print(" cases="); hex(cases);
    print("; v5/v6 exact including inputs/guards, cached/CLP, 5 frm, Top-K ties/NaNs, CRC\n");
    platform_exit(0);
}
