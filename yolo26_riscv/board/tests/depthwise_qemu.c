/* Exact old-im2col/scalar/new-kernel comparison, never physical firmware. */
#include "../../rvv_model_ops.h"
#include "../../scalar_model.h"
#include "../memory.h"
#include "../cache.h"
#include "model_generated.h"
#include <stdint.h>

enum { MAX_VALUES = 80 * 52 * 52, GUARD = 16 };
typedef struct {
    uint32_t before[GUARD];
    float data[MAX_VALUES];
    uint32_t after[GUARD];
} Buffer;
static Buffer input_clp BOARD_CLP_DATA, output_clp BOARD_CLP_DATA;
static Buffer input_ram __attribute__((aligned(64))), output_ram __attribute__((aligned(64)));
static float expected[MAX_VALUES], weights[256 * 9], biases[256], columns[9 * 32];
extern uint64_t rvv_depthwise3x3_calls;
static unsigned cases;

static uintptr_t host_call(uintptr_t op, uintptr_t arg) {
    register uintptr_t a0 __asm__("a0") = op;
    register uintptr_t a1 __asm__("a1") = arg;
    __asm__ volatile(".option push\n.option norvc\n.balign 16\nslli zero,zero,31\n"
                     "ebreak\nsrai zero,zero,7\n.option pop"
                     : "+r"(a0), "+r"(a1) :: "memory");
    return a0;
}
static void print(const char *s) { host_call(4, (uintptr_t)s); }
static void hex(uint64_t n) {
    char s[19] = "0x0000000000000000";
    for (unsigned i = 0; i < 16; ++i) s[17-i] = "0123456789abcdef"[(n >> (4*i)) & 15];
    print(s);
}
__attribute__((noreturn)) void platform_exit(uintptr_t code) {
    uintptr_t args[2] = {0x20026, code};
    host_call(0x20, (uintptr_t)args);
    for (;;) __asm__ volatile("wfi");
}
void platform_trap(uintptr_t cause, uintptr_t pc, uintptr_t value, const uintptr_t *regs) {
    (void)regs;
    print("DW TRAP cause="); hex(cause); print(" pc="); hex(pc); print(" value="); hex(value);
    platform_exit(99);
}
static void require(int good, unsigned stage, size_t index, uint32_t got, uint32_t want) {
    if (good) return;
    print("DW FAIL case="); hex(cases); print(" stage="); hex(stage);
    print(" index="); hex(index); print(" got="); hex(got); print(" expected="); hex(want); print("\n");
    platform_exit(stage);
}
static uint32_t bits(float f) { union { float f; uint32_t u; } v = {.f=f}; return v.u; }
static float value(size_t index, unsigned frame) {
    static const uint32_t pattern[] = {
        0, 0x80000000, 0x3f800000, 0xbf800000, 0x3eaaaaab, 0xbeaaaaab,
        1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
        0x4b800000, 0xcb800000, 0x3f800001, 0xbf7fffff,
        0x33800000, 0xb3800000, 0x3f000000, 0xbf000000};
    union { float f; uint32_t u; } v = {.u=pattern[(index * 13 + frame * 7) % 20]};
    return v.f;
}
static uint64_t fingerprint(const float *p, size_t count) {
    uint64_t hash = 1;
    for (size_t i = 0; i < count; ++i) hash = (hash ^ bits(p[i])) * UINT64_C(1099511628211);
    return hash;
}
static void old_path(const Conv2D *p, const float *input) {
    size_t area = p->in_h * p->in_w;
    for (size_t b = 0; b < p->batch; ++b) for (size_t c = 0; c < p->in_channels; ++c) {
        const float *plane = input + (b * p->in_channels + c) * area;
        float *out = expected + (b * p->in_channels + c) * area;
        for (size_t first = 0; first < area; first += 32) {
            size_t count = area-first < 32 ? area-first : 32;
            for (size_t ky = 0; ky < 3; ++ky) for (size_t kx = 0; kx < 3; ++kx)
                for (size_t col = 0; col < count; ++col) {
                    size_t pos = first+col, y = pos/p->in_w+ky-1, x = pos%p->in_w+kx-1;
                    columns[(ky*3+kx)*32+col] = y < p->in_h && x < p->in_w ? plane[y*p->in_w+x] : 0.0f;
                }
            rvv_matmul_ordered(weights+c*9, columns, out+first, 1, count, 9, 9, 32, area);
        }
        for (size_t pos = 0; pos < area; ++pos) {
            volatile float sum = 0.0f;
            for (size_t ky = 0; ky < 3; ++ky) for (size_t kx = 0; kx < 3; ++kx) {
                size_t y = pos/p->in_w+ky-1, x = pos%p->in_w+kx-1;
                float input_value = y < p->in_h && x < p->in_w ? plane[y*p->in_w+x] : 0.0f;
                volatile float product = input_value * weights[c*9+ky*3+kx];
                sum = sum + product;
            }
            require(bits(out[pos]) == bits(sum), 2, pos, bits(out[pos]), bits(sum));
            volatile float result = out[pos] + biases[c];
            out[pos] = result;
        }
    }
}
static void test_case(const Conv2D *p, unsigned frame) {
    ++cases;
    size_t n = p->batch * p->in_channels * p->in_h * p->in_w;
    require(n <= MAX_VALUES && p->in_channels <= 256, 1, 0, 0, 0);
    for (size_t i = 0; i < n; ++i) input_ram.data[i] = input_clp.data[i] = value(i, frame);
    for (size_t i = 0; i < p->in_channels * 9; ++i) weights[i] = value(i*3+1, frame+1);
    for (size_t i = 0; i < p->in_channels; ++i) biases[i] = value(i, frame+2);
    uint64_t hash = fingerprint(input_ram.data, n);
    old_path(p, input_ram.data);
    Buffer *inputs[] = {&input_ram, &input_clp}, *outputs[] = {&output_ram, &output_clp};
    for (unsigned mode = 0; mode < 2; ++mode) {
        Buffer *in = inputs[mode], *out = outputs[mode];
        for (unsigned i = 0; i < GUARD; ++i) in->before[i] = in->after[i] = out->before[i] = out->after[i] = 0xdeadbeef;
        for (size_t i = 0; i < MAX_VALUES; ++i) out->data[i] = -1234.0f;
        uint64_t before = rvv_depthwise3x3_calls;
        require(!rvv_depthwise3x3(p, in->data, weights, biases, out->data), 3, 0, 0, 0);
        require(rvv_depthwise3x3_calls == before+1, 4, 0, 0, 0);
        for (size_t i = 0; i < n; ++i)
            require(bits(out->data[i]) == bits(expected[i]), 5, i, bits(out->data[i]), bits(expected[i]));
        require(fingerprint(in->data, n) == hash, 6, 0, 0, 0);
        for (size_t i = n; i < MAX_VALUES; ++i)
            require(bits(out->data[i]) == bits(-1234.0f), 7, i, bits(out->data[i]), bits(-1234.0f));
        for (unsigned i = 0; i < GUARD; ++i)
            require(in->before[i] == 0xdeadbeef && in->after[i] == 0xdeadbeef &&
                    out->before[i] == 0xdeadbeef && out->after[i] == 0xdeadbeef, 8, i, 0, 0);
    }
}
int main(void) {
    unsigned shapes = 0;
    for (unsigned i = 0; i < MODEL_NODE_COUNT; ++i) {
        const ModelNode *node = model_nodes+i;
        if (node->op != OP_CONV || node->p[4] == 1) continue;
        const ModelTensor *in = model_tensors+node->in[0], *w = model_tensors+node->in[1];
        Conv2D p = {in->shape[0], in->shape[1], in->shape[2], in->shape[3], w->shape[0],
                    w->shape[2], w->shape[3], node->p[0], node->p[1], node->p[2], node->p[3], node->p[4]};
        test_case(&p, 0); test_case(&p, 1); ++shapes;
    }
    require(shapes == 8, 9, shapes, shapes, 8);
    static const size_t edges[][2] = {{1,1},{1,17},{17,1},{2,2},{3,3},{5,33},{3,35},{5,65}};
    for (unsigned frm = 0; frm < 5; ++frm) {
        __asm__ volatile("csrw frm,%0" :: "r"(frm) : "memory");
        for (unsigned i = 0; i < sizeof(edges)/sizeof(edges[0]); ++i) {
            Conv2D p = {2,3,edges[i][0],edges[i][1],3,3,3,1,1,1,1,3};
            test_case(&p, frm);
        }
    }
    __asm__ volatile("csrw fcsr,zero" ::: "memory");
    Conv2D p = {1,3,3,3,3,3,3,1,1,1,1,3};
    uint64_t before = rvv_depthwise3x3_calls;
    for (unsigned which = 0; which < 8; ++which) {
        Conv2D bad = p;
        const float *in = input_ram.data, *w = weights, *bias = biases;
        float *out = output_ram.data;
        if (which == 0) bad.stride_h = 2;
        if (which == 1) bad.groups = 1;
        if (which == 2) bad.pad_w = 0;
        if (which == 3) bad.batch = bad.in_h = bad.in_w = UINT32_MAX;
        if (which == 4) in = 0;
        if (which == 5) bias = 0;
        if (which == 6) out = input_ram.data;
        if (which == 7) out = weights;
        uint64_t hash = fingerprint(output_ram.data, 27);
        require(rvv_depthwise3x3(&bad, in, w, bias, out) == -1, 10, which, 0, 0);
        require(fingerprint(output_ram.data, 27) == hash && rvv_depthwise3x3_calls == before, 11, which, 0, 0);
    }
    print("DEPTHWISE PASS: 8 model shapes x2, 8 edges x5 frm, cached/CLP, exact old/scalar/new, guards\n");
    platform_exit(0);
}
