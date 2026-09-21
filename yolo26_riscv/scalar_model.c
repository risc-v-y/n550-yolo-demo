#include "scalar_model.h"
#include "basic.h"
#include "conv2d.h"
#include "matmul.h"

static size_t product(const ModelTensor *t, size_t begin, size_t end) {
    size_t value = 1;
    for (size_t i = begin; i < end; ++i) value *= t->shape[i];
    return value;
}

static void copy_value(void *output, size_t oi, const void *input, size_t ii, uint32_t dtype) {
    if (dtype == 1) ((float *)output)[oi] = ((const float *)input)[ii];
    else ((int64_t *)output)[oi] = ((const int64_t *)input)[ii];
}

static size_t broadcast_offset(const ModelTensor *from, const ModelTensor *to, size_t index) {
    size_t offset = 0, stride = 1;
    for (size_t d = to->rank; d > 0; --d) {
        size_t coordinate = index % to->shape[d - 1];
        index /= to->shape[d - 1];
        int axis = (int)d - 1 - ((int)to->rank - (int)from->rank);
        if (axis >= 0) {
            if (from->shape[axis] != 1) offset += coordinate * stride;
            stride *= from->shape[axis];
        }
    }
    return offset;
}

/* Same range reduction as our existing SiLU. Only nonpositive arguments occur
 * in stable sigmoid/softmax. This is an approximation whose end-to-end effect
 * is measured; it is not claimed to be a correctly rounded exp implementation.
 */
float scalar_model_exp_neg(float x) {
    if (x <= -80.0f) return 0.0f;
    if (x == 0.0f) return 1.0f;
    int32_t n = (int32_t)(x * 1.4426950408889634f - 0.5f);
    float r = x - (float)n * 0.693145751953125f;
    r -= (float)n * 0.000001428606765330187f;
    float p = 1.0f / 720.0f;
    p = 1.0f / 120.0f + r * p;
    p = 1.0f / 24.0f + r * p;
    p = 1.0f / 6.0f + r * p;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;
    union { uint32_t bits; float value; } scale = {.bits = (uint32_t)(n + 127) << 23};
    return scale.value * p;
}

int scalar_model_node(const ModelNode *node, const ModelTensor *tensors,
                      unsigned char *arena, const unsigned char *constants) {
    const ModelTensor *a[4] = {0}, *z[2] = {0};
    const void *x[4] = {0};
    void *y[2] = {0};
    float topk_values[300];
    ModelTensor topk_value_shape;
    if (!node->inputs || node->inputs > 4 || !node->outputs || node->outputs > 2) return -1;
    for (size_t i = 0; i < node->inputs; ++i) {
        a[i] = tensors + node->in[i];
        x[i] = (a[i]->constant ? constants : arena) + a[i]->offset;
    }
    for (size_t i = 0; i < node->outputs; ++i) {
        if (node->out[i] >= 0) {
            z[i] = tensors + node->out[i];
            if (z[i]->constant && node->op != OP_COPY) return -1;
            y[i] = z[i]->constant ? (void *)(constants + z[i]->offset) : arena + z[i]->offset;
        }
    }
    /* The first detection Top-K consumes only indices. Keep its temporary
     * scores locally while respecting the pruned graph output contract. */
    if (!z[0] && node->op == OP_TOPK && z[1] && z[1]->count <= 300) {
        topk_value_shape = *z[1];
        topk_value_shape.dtype = 1;
        topk_value_shape.bytes = topk_value_shape.count * sizeof(float);
        z[0] = &topk_value_shape;
        y[0] = topk_values;
    }
    if (!z[0]) return -1;
    float *out = y[0];
    const float *in = x[0];
    const int32_t *p = node->p;
    switch (node->op) {
        case OP_CONV: {
            if (node->inputs != 3 || a[0]->rank != 4 || a[1]->rank != 4) return -1;
            Conv2D c = {a[0]->shape[0], a[0]->shape[1], a[0]->shape[2], a[0]->shape[3],
                        a[1]->shape[0], a[1]->shape[2], a[1]->shape[3], p[0], p[1], p[2], p[3], p[4]};
            /* Pointwise convolution is a dense product in our NCHW layout. */
            if (c.kernel_h == 1 && c.kernel_w == 1 && c.stride_h == 1 && c.stride_w == 1 &&
                c.pad_h == 0 && c.pad_w == 0 && c.groups == 1) {
                size_t area = (size_t)c.in_h * c.in_w;
                for (size_t n = 0; n < c.batch; ++n) {
                    float *dest = out + n * c.out_channels * area;
                    matmul_fp32_scalar(x[1], in + n * c.in_channels * area, dest,
                                       c.out_channels, area, c.in_channels, c.in_channels, area, area);
                    for (size_t oc = 0; oc < c.out_channels; ++oc)
                        for (size_t i = 0; i < area; ++i) dest[oc * area + i] += ((const float *)x[2])[oc];
                }
                return 0;
            }
            return conv2d_fp32_scalar(&c, in, x[1], x[2], out);
        }
        case OP_SILU:
            for (size_t i = 0; i < z[0]->count; ++i) out[i] = silu_fp32_value(in[i]);
            return 0;
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
            if (a[0]->dtype != 1 || a[1]->dtype != 1 || z[0]->dtype != 1) return -1;
            for (size_t i = 0; i < z[0]->count; ++i) {
                float left = in[broadcast_offset(a[0], z[0], i)];
                float right = ((const float *)x[1])[broadcast_offset(a[1], z[0], i)];
                out[i] = node->op == OP_ADD ? left + right : node->op == OP_SUB ? left - right : left * right;
            }
            return 0;
        case OP_COPY:
            if (a[0]->count != z[0]->count || a[0]->dtype != z[0]->dtype) return -1;
            if (y[0] != x[0])
                for (size_t i = 0; i < z[0]->count; ++i) copy_value(y[0], i, x[0], i, a[0]->dtype);
            return 0;
        case OP_SLICE: {
            size_t axis = p[0], outer = product(a[0], 0, axis), inner = product(a[0], axis + 1, a[0]->rank);
            size_t width = z[0]->shape[axis];
            if (a[0]->dtype == 1)
                return slice_fp32_scalar(in, out, outer, a[0]->shape[axis], inner, p[1], width);
            for (size_t row = 0; row < outer; ++row)
                for (size_t i = 0; i < width * inner; ++i)
                    copy_value(y[0], row * width * inner + i, x[0], (row * a[0]->shape[axis] + p[1]) * inner + i, 2);
            return 0;
        }
        case OP_CONCAT: {
            size_t axis = p[0], outer = product(z[0], 0, axis), inner = product(z[0], axis + 1, z[0]->rank);
            const float *inputs[4];
            size_t widths[4];
            for (size_t i = 0; i < node->inputs; ++i) {
                if (a[i]->dtype != 1) return -1;
                inputs[i] = x[i]; widths[i] = a[i]->shape[axis];
            }
            return concat_fp32_scalar(inputs, out, outer, inner, node->inputs, widths);
        }
        case OP_POOL: {
            Pool2D pool = {a[0]->shape[0], a[0]->shape[1], a[0]->shape[2], a[0]->shape[3],
                           p[0], p[1], p[2], p[3], p[4], p[5]};
            if (z[1]) return -1; /* This model never consumes pooling indices. */
            return maxpool_fp32_scalar(&pool, in, out);
        }
        case OP_NEAREST: {
            Shape4 shape = {a[0]->shape[0], a[0]->shape[1], a[0]->shape[2], a[0]->shape[3]};
            if (z[0]->shape[2] % shape.h || z[0]->shape[3] % shape.w) return -1;
            return nearest_fp32_scalar(&shape, z[0]->shape[2] / shape.h, z[0]->shape[3] / shape.w, in, out);
        }
        case OP_PERMUTE: {
            size_t strides[4], stride = 1;
            for (size_t d = a[0]->rank; d > 0; --d) { strides[d - 1] = stride; stride *= a[0]->shape[d - 1]; }
            for (size_t i = 0; i < z[0]->count; ++i) {
                size_t rest = i, offset = 0;
                for (size_t d = z[0]->rank; d > 0; --d) {
                    offset += (rest % z[0]->shape[d - 1]) * strides[p[d - 1]];
                    rest /= z[0]->shape[d - 1];
                }
                copy_value(y[0], i, x[0], offset, a[0]->dtype);
            }
            return 0;
        }
        case OP_EXPAND:
            for (size_t i = 0; i < z[0]->count; ++i)
                copy_value(y[0], i, x[0], broadcast_offset(a[0], z[0], i), a[0]->dtype);
            return 0;
        case OP_BMM: {
            if (a[0]->rank != 3 || a[1]->rank != 3) return -1;
            size_t batches = a[0]->shape[0], m = a[0]->shape[1], k = a[0]->shape[2], n = a[1]->shape[2];
            if (a[1]->shape[0] != batches || a[1]->shape[1] != k) return -1;
            for (size_t b = 0; b < batches; ++b)
                matmul_fp32_scalar(in + b * m * k, (const float *)x[1] + b * k * n, out + b * m * n, m, n, k, k, n, n);
            return 0;
        }
        case OP_SOFTMAX: {
            if ((uint32_t)p[0] + 1 != a[0]->rank) return -1;
            size_t width = a[0]->shape[p[0]], rows = a[0]->count / width;
            for (size_t row = 0; row < rows; ++row) {
                float best = in[row * width], sum = 0.0f;
                for (size_t j = 1; j < width; ++j) if (in[row * width + j] > best) best = in[row * width + j];
                for (size_t j = 0; j < width; ++j) {
                    out[row * width + j] = scalar_model_exp_neg(in[row * width + j] - best);
                    sum += out[row * width + j];
                }
                for (size_t j = 0; j < width; ++j) out[row * width + j] /= sum;
            }
            return 0;
        }
        case OP_SIGMOID:
            for (size_t i = 0; i < z[0]->count; ++i) {
                float e = scalar_model_exp_neg(in[i] > 0.0f ? -in[i] : in[i]);
                out[i] = in[i] >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
            }
            return 0;
        case OP_MAX: {
            if ((uint32_t)p[0] + 1 != a[0]->rank) return -1;
            size_t width = a[0]->shape[p[0]], rows = a[0]->count / width;
            for (size_t row = 0; row < rows; ++row) {
                size_t at = 0;
                for (size_t j = 1; j < width; ++j) if (in[row * width + j] > in[row * width + at]) at = j;
                out[row] = in[row * width + at];
                if (y[1]) ((int64_t *)y[1])[row] = at;
            }
            return 0;
        }
        case OP_TOPK: {
            if ((uint32_t)p[0] + 1 != a[0]->rank || !y[1]) return -1;
            size_t width = a[0]->shape[p[0]], k = p[1], rows = a[0]->count / width;
            if (!k || k > width) return -1;
            for (size_t row = 0; row < rows; ++row) {
                float *values = out + row * k;
                int64_t *indices = (int64_t *)y[1] + row * k;
                size_t used = 0;
                for (size_t j = 0; j < width; ++j) {
                    float value = in[row * width + j];
                    if (used == k && value <= values[k - 1]) continue;
                    size_t at = used < k ? used++ : k - 1;
                    while (at && value > values[at - 1]) {
                        values[at] = values[at - 1]; indices[at] = indices[at - 1]; --at;
                    }
                    values[at] = value; indices[at] = j;
                }
            }
            return 0;
        }
        case OP_GATHER: {
            size_t axis = p[0], strides[4], stride = 1;
            for (size_t d = a[0]->rank; d > 0; --d) { strides[d - 1] = stride; stride *= a[0]->shape[d - 1]; }
            if (a[1]->dtype != 2 || a[1]->count != z[0]->count) return -1;
            for (size_t i = 0; i < z[0]->count; ++i) {
                int64_t selected = ((const int64_t *)x[1])[i];
                if (selected < 0 || (uint64_t)selected >= a[0]->shape[axis]) return -1;
                size_t rest = i, offset = 0;
                for (size_t d = z[0]->rank; d > 0; --d) {
                    size_t coordinate = rest % z[0]->shape[d - 1]; rest /= z[0]->shape[d - 1];
                    offset += (d - 1 == axis ? (size_t)selected : coordinate) * strides[d - 1];
                }
                copy_value(y[0], i, x[0], offset, a[0]->dtype);
            }
            return 0;
        }
        case OP_REMAINDER:
        case OP_FLOORDIV:
            if (p[0] <= 0 || a[0]->dtype != 2 || z[0]->dtype != 2) return -1;
            for (size_t i = 0; i < z[0]->count; ++i) {
                int64_t value = ((const int64_t *)x[0])[i];
                if (value < 0) return -1; /* Detection indices are nonnegative. */
                ((int64_t *)y[0])[i] = node->op == OP_REMAINDER ? value % p[0] : value / p[0];
            }
            return 0;
        case OP_CAST:
            if (a[0]->dtype != 2 || z[0]->dtype != 1) return -1;
            for (size_t i = 0; i < z[0]->count; ++i) out[i] = (float)((const int64_t *)x[0])[i];
            return 0;
    }
    return -1;
}
