#include "basic.h"

/* Bound offsets to PTRDIFF_MAX so RVV byte strides also fit ptrdiff_t. */
static size_t product(size_t a, size_t b) {
    return !a || !b || a > (size_t)PTRDIFF_MAX / sizeof(float) / b ? 0 : a * b;
}

int basic_pool_shape(const Pool2D *p, size_t *oh, size_t *ow) {
    if (!p || !oh || !ow || !p->n || !p->c || !p->h || !p->w || !p->kh || !p->kw ||
        !p->sh || !p->sw || p->ph > p->kh / 2 || p->pw > p->kw / 2) return -1;
    uint64_t padded_h = (uint64_t)p->h + 2ull * p->ph, padded_w = (uint64_t)p->w + 2ull * p->pw;
    if (padded_h < p->kh || padded_w < p->kw) return -1;
    uint64_t h = (padded_h - p->kh) / p->sh + 1, w = (padded_w - p->kw) / p->sw + 1;
    if (h > SIZE_MAX || w > SIZE_MAX) return -1;
    size_t planes = product(p->n, p->c);
    if (!product(planes, product(p->h, p->w)) || !product(planes, product((size_t)h, (size_t)w))) return -1;
    *oh = (size_t)h; *ow = (size_t)w;
    return 0;
}

int basic_nearest_shape(const Shape4 *p, size_t sh, size_t sw, size_t *oh, size_t *ow) {
    if (!p || !oh || !ow) return -1;
    size_t planes = product(p->n, p->c), h = product(p->h, sh), w = product(p->w, sw);
    if (!product(planes, product(p->h, p->w)) || !product(planes, product(h, w))) return -1;
    *oh = h; *ow = w;
    return 0;
}

size_t basic_join_shape(size_t outer, size_t inner, size_t parts, const size_t *widths, size_t *axis) {
    if (!parts || !widths || !axis) return 0;
    size_t sum = 0;
    for (size_t part = 0; part < parts; ++part) {
        if (widths[part] > SIZE_MAX - sum) return 0;
        sum += widths[part];
    }
    size_t count = product(product(outer, sum), inner);
    if (count) *axis = sum;
    return count;
}

int basic_slice_shape(size_t outer, size_t axis, size_t inner, size_t start, size_t width) {
    if (!product(product(outer, axis), inner) || start > axis || width > axis - start) return -1;
    return 0;
}

int maxpool_fp32_scalar(const Pool2D *p, const float *input, float *output) {
    size_t oh, ow;
    if (!input || !output || basic_pool_shape(p, &oh, &ow)) return -1;
    const union { uint32_t bits; float value; } negative_inf = {.bits = 0xff800000u};
    for (size_t plane = 0; plane < (size_t)p->n * p->c; ++plane) {
        for (size_t y = 0; y < oh; ++y) {
            for (size_t x = 0; x < ow; ++x) {
                float best = negative_inf.value;
                for (size_t ky = 0; ky < p->kh; ++ky) {
                    int64_t iy = (int64_t)(y * p->sh + ky) - p->ph;
                    if (iy < 0 || iy >= p->h) continue;
                    for (size_t kx = 0; kx < p->kw; ++kx) {
                        int64_t ix = (int64_t)(x * p->sw + kx) - p->pw;
                        if (ix < 0 || ix >= p->w) continue;
                        float value = input[plane * p->h * p->w + (size_t)iy * p->w + (size_t)ix];
                        if (value > best) best = value;
                    }
                }
                output[plane * oh * ow + y * ow + x] = best;
            }
        }
    }
    return 0;
}

int nearest_fp32_scalar(const Shape4 *p, size_t sh, size_t sw, const float *input, float *output) {
    size_t oh, ow;
    if (!input || !output || basic_nearest_shape(p, sh, sw, &oh, &ow)) return -1;
    for (size_t plane = 0; plane < (size_t)p->n * p->c; ++plane)
        for (size_t y = 0; y < oh; ++y)
            for (size_t x = 0; x < ow; ++x)
                output[plane * oh * ow + y * ow + x] = input[plane * p->h * p->w + y / sh * p->w + x / sw];
    return 0;
}

int add_fp32_scalar(const float *a, const float *b, float *output, size_t count) {
    if (count > (size_t)PTRDIFF_MAX / sizeof(float) || (count && (!a || !b || !output))) return -1;
    for (size_t i = 0; i < count; ++i) output[i] = a[i] + b[i];
    return 0;
}

int concat_fp32_scalar(const float *const *inputs, float *output, size_t outer, size_t inner,
                       size_t parts, const size_t *widths) {
    size_t axis;
    if (!inputs || !output || !basic_join_shape(outer, inner, parts, widths, &axis)) return -1;
    for (size_t part = 0; part < parts; ++part) if (widths[part] && !inputs[part]) return -1;
    for (size_t row = 0; row < outer; ++row) {
        size_t start = 0;
        for (size_t part = 0; part < parts; ++part) {
            size_t count = widths[part] * inner;
            for (size_t i = 0; i < count; ++i) output[row * axis * inner + start + i] = inputs[part][row * count + i];
            start += count;
        }
    }
    return 0;
}

int slice_fp32_scalar(const float *input, float *output, size_t outer, size_t axis,
                      size_t inner, size_t start, size_t width) {
    if (basic_slice_shape(outer, axis, inner, start, width) || (width && (!input || !output))) return -1;
    for (size_t row = 0; row < outer; ++row)
        for (size_t i = 0; i < width * inner; ++i)
            output[row * width * inner + i] = input[(row * axis + start) * inner + i];
    return 0;
}
