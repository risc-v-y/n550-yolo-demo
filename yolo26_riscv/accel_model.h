#ifndef YOLO26_ACCEL_MODEL_H_
#define YOLO26_ACCEL_MODEL_H_
#include "scalar_model.h"
#define MODEL_PRECISION_FP16 1
#define MODEL_PRECISION_MIXED 2
#ifndef MODEL_PRECISION
#define MODEL_PRECISION MODEL_PRECISION_FP16
#endif
#if MODEL_PRECISION != MODEL_PRECISION_FP16 && MODEL_PRECISION != MODEL_PRECISION_MIXED
#error Unsupported MODEL_PRECISION
#endif
extern uint32_t accel_rvv_nodes, accel_amu_nodes, accel_scalar_nodes, accel_alias_nodes;
/* Called once before READY. Immutable FP16 weights stay in ordinary DDR. */
extern uint64_t accel_weight_bytes, accel_weight_values, accel_weight_cycles, accel_weight_layers;
int accel_model_prepare(const ModelNode *nodes, size_t node_count,
                        const ModelTensor *tensors, size_t tensor_count,
                        const unsigned char *constants, size_t constant_bytes);
int accel_model_node(const ModelNode *node, const ModelTensor *tensors,
                     unsigned char *arena, const unsigned char *constants);
#endif
