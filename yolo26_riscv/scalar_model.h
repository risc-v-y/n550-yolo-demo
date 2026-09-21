#ifndef YOLO26_SCALAR_MODEL_H_
#define YOLO26_SCALAR_MODEL_H_

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t offset, ref_offset, bytes, count, rank, dtype, constant, shape[4];
} ModelTensor;
typedef enum {
    OP_CONV, OP_SILU, OP_ADD, OP_SUB, OP_MUL, OP_COPY, OP_SLICE, OP_CONCAT, OP_POOL,
    OP_NEAREST, OP_PERMUTE, OP_EXPAND, OP_BMM, OP_SOFTMAX, OP_SIGMOID, OP_MAX,
    OP_TOPK, OP_GATHER, OP_REMAINDER, OP_FLOORDIV, OP_CAST
} ModelOp;
typedef struct {
    ModelOp op;
    uint32_t inputs, outputs;
    int32_t in[4], out[2], p[8];
    const char *module;
} ModelNode;

/* Frozen graph tensors use contiguous logical order. Dtype 1=FP32, 2=int64.
 * This executor only calls scalar kernels, with FP32 accumulation and no FMA.
 */
int scalar_model_node(const ModelNode *node, const ModelTensor *tensors,
                      unsigned char *arena, const unsigned char *constants);
float scalar_model_exp_neg(float x);

#endif
