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
int accel_model_node(const ModelNode *node, const ModelTensor *tensors,
                     unsigned char *arena, const unsigned char *constants);
#endif
