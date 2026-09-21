#include "conv2d.h"

float silu_fp32_value(float x) {
    if (x != x) return x;
    if (x >= 80.0f) return x;
    if (x <= -80.0f) return -0.0f;
    float negative_abs = x > 0.0f ? -x : x;
    /* exp(z) = 2^n * exp(r), |r| <= ln(2)/2. n is nonpositive here.
     * Split ln(2) minimizes cancellation; degree-6 Taylor on this short range.
     */
    int32_t n = (int32_t)(negative_abs * 1.4426950408889634f - 0.5f);
    float r = negative_abs - (float)n * 0.693145751953125f;
    r -= (float)n * 0.000001428606765330187f;
    float polynomial = 1.0f / 720.0f;
    polynomial = 1.0f / 120.0f + r * polynomial;
    polynomial = 1.0f / 24.0f + r * polynomial;
    polynomial = 1.0f / 6.0f + r * polynomial;
    polynomial = 0.5f + r * polynomial;
    polynomial = 1.0f + r * polynomial;
    polynomial = 1.0f + r * polynomial;
    union { uint32_t bits; float value; } scale = {.bits = (uint32_t)(n + 127) << 23};
    float e = scale.value * polynomial;
    return x >= 0.0f ? x / (1.0f + e) : x * e / (1.0f + e);
}

void silu_fp32(float *values, size_t count) {
    for (size_t index = 0; index < count; ++index) values[index] = silu_fp32_value(values[index]);
}
