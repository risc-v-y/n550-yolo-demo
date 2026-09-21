#include "accel_model.h"
#include "amu.h"
#include "basic.h"
#include "conv2d.h"
#include "rvv_model_ops.h"
#include "model_profile.h"
#include <riscv_vector.h>

uint32_t accel_rvv_nodes, accel_amu_nodes, accel_scalar_nodes, accel_alias_nodes;
enum { CONV_COLUMNS = 32, MAX_CONV_K = 4096 };
static float conv_scratch[MAX_CONV_K * CONV_COLUMNS] __attribute__((aligned(64)));

static size_t product(const ModelTensor *t, size_t begin, size_t end) {
    size_t value = 1;
    for (size_t i = begin; i < end; ++i) value *= t->shape[i];
    return value;
}
static size_t broadcast(const ModelTensor *from, const ModelTensor *to, size_t index) {
    size_t at = 0, stride = 1;
    for (size_t d = to->rank; d > 0; --d) {
        size_t coordinate = index % to->shape[d - 1]; index /= to->shape[d - 1];
        int axis = (int)d - 1 - ((int)to->rank - (int)from->rank);
        if (axis >= 0) {
            if (from->shape[axis] != 1) at += coordinate * stride;
            stride *= from->shape[axis];
        }
    }
    return at;
}
static int done(int status, int amu) {
    if (!status) {
        if (amu) ++accel_amu_nodes;
        else ++accel_rvv_nodes;
    }
    return status;
}
static int matmul(int amu, const float *a, const float *b, float *c,
                  size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc) {
    if (amu) return amu_matmul_fp16(a, b, c, m, n, k, lda, ldb, ldc);
    rvv_matmul_ordered(a, b, c, m, n, k, lda, ldb, ldc);
    return 0;
}
static void add_bias(float *dest, const float *bias, size_t channels, size_t columns, size_t stride) {
    uint64_t start = model_ticks();
    for (size_t oc = 0; oc < channels; ++oc) {
        for (size_t i = 0; i < columns;) {
            size_t vl = __riscv_vsetvl_e32m1(columns - i);
            float *at = dest + oc * stride + i;
            __riscv_vse32_v_f32m1(at, __riscv_vfadd_vf_f32m1(__riscv_vle32_v_f32m1(at, vl), bias[oc], vl), vl);
            i += vl;
        }
    }
    model_profile_end(PROFILE_BIAS_DEQUANT, start);
}
static int convolution(const Conv2D *p, const float *input, const float *weight,
                        const float *bias, float *output, int amu) {
    size_t oh, ow;
    if (conv2d_output_shape(p, &oh, &ow)) return -1;
    size_t icg = p->in_channels / p->groups, ocg = p->out_channels / p->groups;
    size_t k = icg * p->kernel_h * p->kernel_w, area = oh * ow;
    if (k > MAX_CONV_K) return -1;
    for (size_t batch = 0; batch < p->batch; ++batch) {
        for (size_t group = 0; group < p->groups; ++group) {
            const float *weights = weight + group * ocg * k;
            const float *biases = bias + group * ocg;
            float *dest = output + (batch * p->out_channels + group * ocg) * area;
            if (p->kernel_h == 1 && p->kernel_w == 1 && p->stride_h == 1 && p->stride_w == 1 && !p->pad_h && !p->pad_w) {
                if (matmul(amu, weights, input + (batch * p->in_channels + group * icg) * area,
                           dest, ocg, area, k, k, area, area)) return -1;
                add_bias(dest, biases, ocg, area, area);
                continue;
            }
            for (size_t first = 0; first < area; first += CONV_COLUMNS) {
                size_t count = area - first < CONV_COLUMNS ? area - first : CONV_COLUMNS;
                uint64_t start = model_ticks();
                size_t inner = 0;
                for (size_t ic = 0; ic < icg; ++ic) {
                    for (size_t ky = 0; ky < p->kernel_h; ++ky) {
                        for (size_t kx = 0; kx < p->kernel_w; ++kx, ++inner) {
                            for (size_t col = 0; col < count; ++col) {
                                size_t pos = first + col;
                                int64_t iy = (int64_t)(pos / ow * p->stride_h + ky) - p->pad_h;
                                int64_t ix = (int64_t)(pos % ow * p->stride_w + kx) - p->pad_w;
                                float value = 0.0f;
                                if (iy >= 0 && ix >= 0 && iy < p->in_h && ix < p->in_w)
                                    value = input[((batch * p->in_channels + group * icg + ic) * p->in_h + (size_t)iy) * p->in_w + (size_t)ix];
                                conv_scratch[inner * CONV_COLUMNS + col] = value;
                            }
                        }
                    }
                }
                model_profile_end(PROFILE_INPUT_IM2COL, start);
                if (matmul(amu, weights, conv_scratch, dest + first, ocg, count, k, k, CONV_COLUMNS, area)) return -1;
                add_bias(dest + first, biases, ocg, count, area);
            }
        }
    }
    return 0;
}

int accel_model_node(const ModelNode *node, const ModelTensor *tensors,
                     unsigned char *arena, const unsigned char *constants) {
    const ModelTensor *a[4] = {0};
    const void *x[4] = {0};
    for (size_t i = 0; i < node->inputs; ++i) {
        a[i] = tensors + node->in[i];
        x[i] = (a[i]->constant ? constants : arena) + a[i]->offset;
    }
    if (node->out[0] < 0) goto scalar;
    const ModelTensor *z = tensors + node->out[0];
    if (z->constant && node->op != OP_COPY) return -1;
    void *y = z->constant ? (void *)(constants + z->offset) : arena + z->offset;
    float *out = y;
    const float *in = x[0];
    const int32_t *p = node->p;
    switch (node->op) {
        case OP_CONV: {
            Conv2D c = {a[0]->shape[0], a[0]->shape[1], a[0]->shape[2], a[0]->shape[3],
                        a[1]->shape[0], a[1]->shape[2], a[1]->shape[3], p[0], p[1], p[2], p[3], p[4]};
            int amu = c.groups == 1;
            return done(convolution(&c, in, x[1], x[2], out, amu), amu);
        }
        case OP_BMM:
            for (size_t b = 0; b < a[0]->shape[0]; ++b) {
                size_t m = a[0]->shape[1], k = a[0]->shape[2], n = a[1]->shape[2];
                rvv_matmul_ordered(in + b * m * k, (const float *)x[1] + b * k * n,
                                   out + b * m * n, m, n, k, k, n, n);
            }
            return done(0, 0);
        case OP_SILU:
        case OP_SIGMOID:
            rvv_nonlinear(in, out, z->count, node->op == OP_SILU);
            return done(0, 0);
        case OP_SOFTMAX:
            if ((uint32_t)p[0] + 1 != a[0]->rank) return -1;
            rvv_softmax(in, out, a[0]->count / a[0]->shape[p[0]], a[0]->shape[p[0]]);
            return done(0, 0);
        case OP_COPY:
            if (z->bytes != a[0]->bytes) return -1;
            if (y == x[0]) {
                ++accel_alias_nodes;
                return 0;
            }
            rvv_copy_bytes(y, x[0], z->bytes);
            return done(0, 0);
        case OP_SLICE: {
            if (a[0]->dtype != 1) goto scalar;
            size_t axis = p[0];
            return done(slice_fp32_rvv(in, out, product(a[0], 0, axis), a[0]->shape[axis],
                                       product(a[0], axis + 1, a[0]->rank), p[1], z->shape[axis]), 0);
        }
        case OP_CONCAT: {
            const float *inputs[4]; size_t widths[4];
            for (size_t i = 0; i < node->inputs; ++i) { inputs[i] = x[i]; widths[i] = a[i]->shape[p[0]]; }
            return done(concat_fp32_rvv(inputs, out, product(z, 0, p[0]), product(z, p[0] + 1, z->rank),
                                        node->inputs, widths), 0);
        }
        case OP_POOL: {
            Pool2D pool = {a[0]->shape[0], a[0]->shape[1], a[0]->shape[2], a[0]->shape[3],
                           p[0], p[1], p[2], p[3], p[4], p[5]};
            return done(maxpool_fp32_rvv(&pool, in, out), 0);
        }
        case OP_NEAREST: {
            Shape4 shape = {a[0]->shape[0], a[0]->shape[1], a[0]->shape[2], a[0]->shape[3]};
            return done(nearest_fp32_rvv(&shape, z->shape[2] / shape.h, z->shape[3] / shape.w, in, out), 0);
        }
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
            for (size_t i = 0; i < z->count;) {
                size_t vl = __riscv_vsetvl_e32m1(z->count - i < 64 ? z->count - i : 64);
                uint32_t left[64], right[64];
                for (size_t lane = 0; lane < vl; ++lane) {
                    left[lane] = (uint32_t)(broadcast(a[0], z, i + lane) * sizeof(float));
                    right[lane] = (uint32_t)(broadcast(a[1], z, i + lane) * sizeof(float));
                }
                vfloat32m1_t l = __riscv_vluxei32_v_f32m1(in, __riscv_vle32_v_u32m1(left, vl), vl);
                vfloat32m1_t r = __riscv_vluxei32_v_f32m1(x[1], __riscv_vle32_v_u32m1(right, vl), vl);
                vfloat32m1_t value;
                if (node->op == OP_ADD) value = __riscv_vfadd_vv_f32m1(l, r, vl);
                else if (node->op == OP_SUB) value = __riscv_vfsub_vv_f32m1(l, r, vl);
                else value = __riscv_vfmul_vv_f32m1(l, r, vl);
                __riscv_vse32_v_f32m1(out + i, value, vl);
                i += vl;
            }
            return done(0, 0);
        case OP_PERMUTE:
        case OP_EXPAND:
        case OP_GATHER: {
            if (z->dtype != 1) goto scalar;
            size_t strides[4], stride = 1;
            for (size_t d = a[0]->rank; d > 0; --d) { strides[d - 1] = stride; stride *= a[0]->shape[d - 1]; }
            for (size_t i = 0; i < z->count;) {
                size_t vl = __riscv_vsetvl_e32m1(z->count - i < 64 ? z->count - i : 64);
                uint32_t indices[64];
                for (size_t lane = 0; lane < vl; ++lane) {
                    size_t at = i + lane, offset = 0;
                    if (node->op == OP_EXPAND) offset = broadcast(a[0], z, at);
                    else {
                        int64_t selected = 0;
                        if (node->op == OP_GATHER) {
                            selected = ((const int64_t *)x[1])[at];
                            if (selected < 0 || (uint64_t)selected >= a[0]->shape[p[0]]) return -1;
                        }
                        for (size_t d = z->rank; d > 0; --d) {
                            size_t coordinate = at % z->shape[d - 1]; at /= z->shape[d - 1];
                            if (node->op == OP_PERMUTE) offset += coordinate * strides[p[d - 1]];
                            else offset += (d - 1 == (size_t)p[0] ? (size_t)selected : coordinate) * strides[d - 1];
                        }
                    }
                    indices[lane] = (uint32_t)(offset * sizeof(float));
                }
                __riscv_vse32_v_f32m1(out + i, __riscv_vluxei32_v_f32m1(in, __riscv_vle32_v_u32m1(indices, vl), vl), vl);
                i += vl;
            }
            return done(0, 0);
        }
        default: break;
    }
scalar:
    ++accel_scalar_nodes;
    return scalar_model_node(node, tensors, arena, constants);
}
