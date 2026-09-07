// ac/transformer.h -- the pieces of audiocraft's StreamingTransformer that both target
// models need.
//
// MelodyFlow's DiT and MusicGen's LM are the same `StreamingTransformer` with different
// flags: pre-norm LayerNorm with bias, no bias on any projection, a fused [3*dim, dim] qkv
// weight, GELU feed-forward, optional cross-attention. They differ in whether attention is
// causal, whether position comes from RoPE or a sinusoidal input embedding, and whether
// there are U-ViT skips and an additive timestep.
//
// What lives here is the shared mechanism. Block composition lives with the model, because
// that is what actually differs -- see src/mf/dit.h.
//
// Activations are ggml [dim, seq] (channel fastest), matching sa3.cpp's DiT convention, so
// per-channel parameters broadcast without a reshape.
#pragma once

#include "ggml.h"
#include "gguf_model.h"
#include "nn.h"

#include <cmath>
#include <string>

namespace ac {

// LayerNorm with a learned scale and bias, which is what `create_norm_fn('layer_norm')`
// builds (eps 1e-5, hardcoded in audiocraft/modules/transformer.py).
inline ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                               ggml_tensor* weight, ggml_tensor* bias, float eps) {
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, weight);
    return ggml_add(ctx, y, bias);
}

inline ggml_tensor* layer_norm(ggml_context* ctx, const GgufModel& W, const std::string& p,
                               ggml_tensor* x, float eps) {
    return layer_norm(ctx, x, W.get(p + ".weight"), W.get(p + ".bias"), eps);
}

// Rotary embedding using the frequencies stored in the checkpoint rather than a recomputed
// closed form.
//
// This matters: MelodyFlow's `frequencies` buffer has been round-tripped through bfloat16,
// so it is not the analytic 1 / base ** (2i/d) -- index 1 is 0.75, not 0.74989421 -- and
// torch rotates with the stored values. By position 749 that is 0.08 radians of drift.
//
// ggml computes `theta_i = (pos * freq_base ** (-2i/n_dims)) / freq_factors[i]`. Passing
// freq_base = 1 makes the first factor exactly 1 for every i, so the stored frequencies go
// in verbatim as their reciprocals (the converter writes `dit.rope_freq_factors`) with no
// dependence on how ggml accumulates its own power series.
//
// `x` is [head_dim, n_head, seq]; `pos` is I32 [seq]. Pairs are interleaved -- (0,1),
// (2,3), ... -- which is GGML_ROPE_TYPE_NORMAL and matches audiocraft's
// `view_as_complex(x.reshape(..., -1, 2))`.
inline ggml_tensor* rope_stored(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pos,
                                ggml_tensor* freq_factors, int n_dims) {
    return ggml_rope_ext(ctx, x, pos, freq_factors, n_dims, GGML_ROPE_TYPE_NORMAL,
                         /*n_ctx_orig=*/0, /*freq_base=*/1.0f, /*freq_scale=*/1.0f,
                         /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                         /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
}

// `add_zero_attn`: prepend one all-zero key and value to the key sequence.
//
// audiocraft does this with `F.pad(k, (0, 0, 1, 0))` on a [batch, heads, time, dim]
// tensor, i.e. one extra timestep of zeros at the front of the keys, for both self- and
// cross-attention. It is not decoration. In the unconditional branch of classifier-free
// guidance the text mask is all zeros, so `log(mask)` is -inf for every real key and the
// softmax would be undefined; the zero key is the only thing left to attend to, and since
// its value is zero the cross-attention contributes exactly nothing. Drop it and the null
// branch is NaN.
//
// `t` is [head_dim, seq, heads]; returns [head_dim, seq + 1, heads].
inline ggml_tensor* prepend_zero_key(ggml_context* ctx, ggml_tensor* t) {
    ggml_tensor* head = ggml_view_3d(ctx, t, t->ne[0], 1, t->ne[2],
                                     t->nb[1], t->nb[2], 0);
    ggml_tensor* zeros = ggml_scale(ctx, ggml_cont(ctx, head), 0.0f);
    return ggml_concat(ctx, zeros, t, 1);
}

// Split a fused [3*dim, seq] projection into its q, k and v thirds. `i` selects one.
inline ggml_tensor* qkv_slice(ggml_context* ctx, ggml_tensor* qkv, int dim, int i) {
    return ggml_view_2d(ctx, qkv, dim, qkv->ne[1], qkv->nb[1], (size_t)i * dim * sizeof(float));
}

// [dim, seq] -> [head_dim, n_head, seq].
//
// This is the layout ggml_rope_ext wants (it reads ne[2] as the sequence and indexes `pos`
// with it), which is why splitting heads and moving to the attention layout are two steps
// rather than one: rope happens in between.
inline ggml_tensor* to_heads(ggml_context* ctx, ggml_tensor* x, int head_dim, int n_head) {
    return ggml_reshape_3d(ctx, ggml_cont(ctx, x), head_dim, n_head, x->ne[1]);
}

// [head_dim, n_head, seq] -> attention layout [head_dim, seq, n_head].
inline ggml_tensor* to_attention(ggml_context* ctx, ggml_tensor* h) {
    return ggml_cont(ctx, ggml_permute(ctx, h, 0, 2, 1, 3));
}

// The inverse: [head_dim, seq, n_head] -> [dim, seq].
inline ggml_tensor* from_heads(ggml_context* ctx, ggml_tensor* o, int dim) {
    ggml_tensor* h = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));   // [head_dim, n_head, seq]
    return ggml_reshape_2d(ctx, h, dim, h->ne[2]);
}

// Multi-head attention over tensors already in [head_dim, seq, n_head] layout, with the
// optional zero key. `mask` is additive [n_keys, n_queries] or null.
inline ggml_tensor* attention(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k,
                              ggml_tensor* v, ggml_tensor* mask, bool add_zero_attn) {
    if (add_zero_attn) {
        k = prepend_zero_key(ctx, k);
        v = prepend_zero_key(ctx, v);
    }
    const float scale = 1.0f / std::sqrt((float)q->ne[0]);
    return nn::sdpa(ctx, q, k, v, mask, scale);
}

// The feed-forward block of nn.TransformerEncoderLayer: linear -> activation -> linear,
// with no biases in either target model.
//
// audiocraft's `activation='gelu'` reaches nn.TransformerEncoderLayer as the string
// "gelu", which torch maps to F.gelu with approximate='none' -- the erf form. ggml_gelu is
// the tanh approximation, so this deliberately uses ggml_gelu_erf.
inline ggml_tensor* feed_forward(ggml_context* ctx, const GgufModel& W, const std::string& p,
                                 ggml_tensor* x) {
    ggml_tensor* h = ggml_mul_mat(ctx, W.get(p + "0.weight"), x);
    h = ggml_gelu_erf(ctx, h);
    return ggml_mul_mat(ctx, W.get(p + "2.weight"), h);
}

// Sinusoidal timestep features, `TimestepEmbedding.timestep_embedding` in
// audiocraft/models/flow.py: concat(cos(t * freqs), sin(t * freqs)) with
// freqs = exp(-log(max_period) * arange(half) / half). Computed host-side because it is
// one vector per call and depends on a scalar the graph does not otherwise need.
inline std::vector<float> timestep_features(float t, int dim, float max_period = 10000.0f) {
    const int half = dim / 2;
    std::vector<float> out((size_t)dim, 0.0f);
    for (int i = 0; i < half; ++i) {
        const float freq = std::exp(-std::log(max_period) * (float)i / (float)half);
        const float arg = t * freq;
        out[(size_t)i] = std::cos(arg);
        out[(size_t)half + i] = std::sin(arg);
    }
    return out;
}

} // namespace ac
