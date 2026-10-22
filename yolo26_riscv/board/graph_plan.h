#ifndef YOLO_BOARD_GRAPH_PLAN_H
#define YOLO_BOARD_GRAPH_PLAN_H
#include "../scalar_model.h"
#include "model_generated.h"

enum { GRAPH_VIEWS=1, GRAPH_FUSE_SILU=2, GRAPH_FUSE_NEAREST=4, GRAPH_OWNERSHIP=8 };
static ModelTensor execution_tensors[MODEL_TENSOR_COUNT];
static unsigned graph_root[MODEL_TENSOR_COUNT],graph_relative[MODEL_TENSOR_COUNT];
static int graph_first[MODEL_TENSOR_COUNT],graph_last[MODEL_TENSOR_COUNT];
static unsigned graph_uses[MODEL_TENSOR_COUNT],graph_group[MODEL_NODE_COUNT];
static unsigned graph_flags;
static size_t graph_peak;

/* Rebuild intervals from logical edges, including every alias descendant.
 * First-fit storage is cache-line aligned and inputs live through their last
 * consumer (not merely until that node starts). No old arena offsets are
 * assumed safe after extending a root's lifetime. */
static int graph_plan(unsigned flags) {
    if(flags&~15u) return -1;
    graph_flags=flags; graph_peak=0;
    for(unsigned t=0;t<MODEL_TENSOR_COUNT;++t) {
        execution_tensors[t]=model_tensors[t]; graph_root[t]=t; graph_relative[t]=0;
        graph_first[t]=-2; graph_last[t]=-2; graph_uses[t]=0;
    }
    graph_first[0]=graph_last[0]=-1;
    for(unsigned n=0;n<MODEL_NODE_COUNT;++n) {
        const ModelNode *p=model_nodes+n; graph_group[n]=n;
        for(unsigned j=0;j<p->inputs;++j) ++graph_uses[p->in[j]];
        for(unsigned j=0;j<p->outputs;++j) if(p->out[j]>=0 && !model_tensors[p->out[j]].constant)
            graph_first[p->out[j]]=graph_last[p->out[j]]=(int)n;
    }
    if(!flags) { graph_peak=MODEL_ARENA_BYTES; return 0; }
    for(unsigned n=0;n<MODEL_NODE_COUNT;++n) {
        const ModelNode *p=model_nodes+n;
        if(!p->inputs || p->out[0]<0) continue;
        unsigned a=p->in[0],z=p->out[0];
        const ModelTensor *in=model_tensors+a,*out=model_tensors+z;
        if(out->constant || in->constant) continue;
        int alias=p->op==OP_COPY && in->bytes==out->bytes;
        size_t delta=0;
        if(flags&GRAPH_VIEWS) {
            if(p->op==OP_EXPAND && in->bytes==out->bytes && in->dtype==out->dtype) alias=1;
            if(p->op==OP_SLICE && p->p[0]>=0 && (unsigned)p->p[0]<in->rank && p->p[1]>=0) {
                size_t outer=1,inner=1,axis=(unsigned)p->p[0];
                for(size_t d=0;d<axis;++d) outer*=in->shape[d];
                for(size_t d=axis+1;d<in->rank;++d) inner*=in->shape[d];
                delta=(size_t)p->p[1]*inner*(in->dtype==2 ? 8 : 4);
                alias=outer==1 && delta<=in->bytes && out->bytes<=in->bytes-delta;
            }
        }
        if((flags&GRAPH_FUSE_SILU) && p->op==OP_SILU && n && graph_uses[a]==1 &&
           model_nodes[n-1].op==OP_CONV && model_nodes[n-1].p[4]==1 &&
           model_nodes[n-1].out[0]==(int)a && in->bytes==out->bytes) {
            alias=1; graph_group[n]=n-1;
        }
        if(alias) {
            graph_root[z]=graph_root[a]; graph_relative[z]=graph_relative[a]+(unsigned)delta;
        }
    }
    /* The nearest output can be a segment of the following Concat root.
     * Only channel concatenation with one outer batch and one consumer. */
    if(flags&GRAPH_FUSE_NEAREST) for(unsigned n=0;n+1<MODEL_NODE_COUNT;++n) {
        const ModelNode *p=model_nodes+n,*q=p+1;
        if(p->op!=OP_NEAREST || q->op!=OP_CONCAT || q->p[0]!=1 || q->out[0]<0) continue;
        unsigned a=p->out[0],z=q->out[0];
        if(graph_uses[a]!=1 || model_tensors[a].shape[0]!=1 || graph_root[a]!=a || graph_root[z]!=z) continue;
        size_t offset=0;
        for(unsigned j=0;j<q->inputs;++j) {
            if(q->in[j]==(int)a) {
                graph_root[a]=z; graph_relative[a]=(unsigned)offset;
                graph_first[z]=(int)n; graph_group[n+1]=n; break;
            }
            offset+=model_tensors[q->in[j]].bytes;
        }
    }
    /* Flatten roots after a Concat redirection, then propagate liveness. */
    for(unsigned t=0;t<MODEL_TENSOR_COUNT;++t) {
        unsigned r=graph_root[t];
        for(unsigned guard=0;graph_root[r]!=r;++guard) {
            if(guard==MODEL_TENSOR_COUNT) return -1;
            graph_relative[t]+=graph_relative[r]; r=graph_root[r];
        }
        graph_root[t]=r;
    }
    for(unsigned n=0;n<MODEL_NODE_COUNT;++n) {
        const ModelNode *p=model_nodes+n;
        for(unsigned j=0;j<p->inputs;++j) {
            unsigned r=graph_root[p->in[j]];
            if(graph_last[r]<(int)n) graph_last[r]=(int)n;
        }
    }
    graph_last[graph_root[MODEL_OUTPUT_ID]]=MODEL_NODE_COUNT;
    for(int n=-1;n<(int)MODEL_NODE_COUNT;++n) for(unsigned t=0;t<MODEL_TENSOR_COUNT;++t) {
        if(model_tensors[t].constant || graph_root[t]!=t || graph_first[t]!=n) continue;
        size_t bytes=(model_tensors[t].bytes+63u)&~(size_t)63,at=0;
        for(;;) {
            int collision=0;
            for(unsigned other=0;other<MODEL_TENSOR_COUNT;++other) {
                if(other==t || model_tensors[other].constant || graph_root[other]!=other ||
                   graph_first[other]<-1 || graph_first[other]>n ||
                   (graph_first[other]==n && other>t) || graph_last[other]<n) continue;
                size_t start=execution_tensors[other].offset;
                size_t end=start+((model_tensors[other].bytes+63u)&~(size_t)63);
                if(at<end && start<at+bytes) {at=end; collision=1; break;}
            }
            if(!collision) break;
        }
        if(at+bytes>MODEL_ARENA_BYTES) return -1;
        execution_tensors[t].offset=(uint32_t)at;
        if(at+bytes>graph_peak) graph_peak=at+bytes;
    }
    for(unsigned t=0;t<MODEL_TENSOR_COUNT;++t) if(!model_tensors[t].constant && graph_root[t]!=t) {
        unsigned r=graph_root[t];
        if(graph_relative[t]>model_tensors[r].bytes ||
           model_tensors[t].bytes>model_tensors[r].bytes-graph_relative[t]) return -1;
        execution_tensors[t].offset=execution_tensors[r].offset+graph_relative[t];
    }
    return 0;
}
#endif
