// mg/lm.h -- MusicGen's language model (audiocraft models/lm.py::LMModel).
//
// A causal pre-norm transformer over `n_q` interleaved EnCodec codebooks. Each sequence
// step sums `n_q` embedding lookups, adds a sinusoidal positional embedding, and runs
// self-attention (causal, KV-cached) and cross-attention onto projected T5-base states.
// `n_q` linear heads then read the last position.
//
// It is the same `StreamingTransformer` as MelodyFlow's DiT with different flags -- so the
// block mechanism comes from ac/transformer.h and only the composition lives here. What
// differs: causal with a cache instead of full-sequence, a sinusoidal position added to the
// input instead of RoPE, no U-ViT skips, no timestep, and no `add_zero_attn`.
//
// Two details that are silently load-bearing:
//
//   * the sinusoidal formula divides the exponent by `half_dim - 1`, not `half_dim`. Off by
//     one from the usual definition, and off by one from the timestep embedding in the same
//     codebase (mf/dit.h). `ac.lm.pos_denominator` carries it so the reader cannot guess.
//   * cross-attention gets **no mask**. `T5Conditioner.forward` zeroes padded positions
//     after `output_proj` (bias included) and `ConditionFuser` passes no mask, so padding is
//     attended over as zero vectors. The unconditional branch is an all-zero context of the
//     same length, and since the attention projections have no biases its cross-attention
//     output is then exactly zero -- so the whole block can be skipped. Note *where* the
//     zeroing happens: after `output_proj`, bias included. Feeding zeros to `lm.cond_proj`
//     instead leaves its bias behind and is a different model. See `lm_forward`.
//
// Activations are ggml [dim, seq].
#pragma once

#include "ac/transformer.h"
#include "ggml.h"
#include "gguf_model.h"
#include "mg/kv_cache.h"
#include "mg/pattern.h"

#include <cmath>
#include <string>
#include <vector>

namespace ac {

struct LmConfig {
    int dim = 1024;
    int layers = 24;
    int heads = 16;
    int head_dim = 64;
    int ff_dim = 4096;
    int cond_dim = 768;      // T5-base hidden size, before lm.cond_proj
    int n_q = 4;
    int card = 2048;
    int special_token_id = 2048;
    int pos_denominator = 511;
    std::vector<int> delays{0, 1, 2, 3};
    float max_period = 10000.0f;
    float norm_eps = 1.0e-5f;
    float cfg_coef = 3.0f;
    int sample_rate = 32000;
    float max_duration = 30.0f;

    static LmConfig from(const GgufModel& m) {
        if (m.string("ac.architecture") != "audiocraft")
            throw gguf_error("not an audiocraft GGUF");
        if (m.u32("ac.format_version") != 1)
            throw gguf_error("unsupported audiocraft GGUF version");
        if (m.string("ac.lm.family") != "musicgen")
            throw gguf_error("not a MusicGen LM GGUF");
        LmConfig c;
        c.dim              = (int)m.u32("ac.lm.dim");
        c.layers           = (int)m.u32("ac.lm.layers");
        c.heads            = (int)m.u32("ac.lm.heads");
        c.head_dim         = (int)m.u32("ac.lm.head_dim");
        c.ff_dim           = (int)m.u32("ac.lm.ff_dim");
        c.cond_dim         = (int)m.u32("ac.lm.cond_dim");
        c.n_q              = (int)m.u32("ac.lm.n_q");
        c.card             = (int)m.u32("ac.lm.card");
        c.special_token_id = (int)m.u32("ac.lm.special_token_id");
        c.pos_denominator  = (int)m.u32("ac.lm.pos_denominator");
        c.max_period       = m.f32("ac.lm.max_period");
        c.norm_eps         = m.f32("ac.lm.norm_eps");
        c.cfg_coef         = m.f32("ac.lm.cfg_coef");
        c.sample_rate      = (int)m.u32("ac.lm.sample_rate");
        c.max_duration     = m.f32("ac.lm.max_duration");
        c.delays.assign((size_t)c.n_q, 0);
        for (int i = 0; i < c.n_q; ++i) c.delays[(size_t)i] = i;
        if (c.dim != c.heads * c.head_dim)
            throw gguf_error("LM dim does not equal heads * head_dim");
        if (c.special_token_id != c.card)
            throw gguf_error("special_token_id should be card: it is the extra embedding row");
        return c;
    }

    DelayPattern pattern(int timesteps) const { return DelayPattern(delays, timesteps); }
};

// `create_sin_embedding` (audiocraft modules/transformer.py), host-side.
//
// phase = position / max_period ** (i / (half_dim - 1)); the embedding is
// cat(cos(phase), sin(phase)). Note the denominator: `half_dim - 1`, so the last frequency
// is exactly 1 / max_period rather than approaching it.
//
// Returns [dim, n] in ggml order for positions `first .. first + n - 1`.
inline std::vector<float> sin_positions(int first, int n, const LmConfig& c) {
    const int half = c.dim / 2;
    const double denom = (double)c.pos_denominator;
    std::vector<float> out((size_t)c.dim * n);
    for (int t = 0; t < n; ++t) {
        float* dst = out.data() + (size_t)t * c.dim;
        const double pos = (double)(first + t);
        for (int i = 0; i < half; ++i) {
            const double phase = pos / std::pow((double)c.max_period, (double)i / denom);
            dst[i] = (float)std::cos(phase);
            dst[half + i] = (float)std::sin(phase);
        }
    }
    return out;
}

// Activations here are [dim, n_tokens, n_seq]: the trailing axis is the classifier-free
// guidance batch, 2 when both streams run in one forward and 1 without guidance. The shared
// helpers in ac/transformer.h are 2D because MelodyFlow's DiT has no such axis, so the head
// reshapes are spelled out again rather than generalized -- the DiT is validated and this is
// four lines.
//
// [dim, n_tokens, n_seq] -> attention layout [head_dim, n_tokens, n_head, n_seq].
inline ggml_tensor* lm_to_attention(ggml_context* ctx, ggml_tensor* x, int head_dim,
                                    int n_head) {
    ggml_tensor* h = ggml_reshape_4d(ctx, ggml_cont(ctx, x), head_dim, n_head,
                                     x->ne[1], x->ne[2]);
    return ggml_cont(ctx, ggml_permute(ctx, h, 0, 2, 1, 3));
}

// The inverse: [head_dim, n_tokens, n_head, n_seq] -> [dim, n_tokens, n_seq].
inline ggml_tensor* lm_from_heads(ggml_context* ctx, ggml_tensor* o, int dim) {
    ggml_tensor* h = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
    return ggml_reshape_3d(ctx, h, dim, h->ne[2], h->ne[3]);
}

// One third of a fused [3*dim, ...] projection, keeping the batch axis.
inline ggml_tensor* lm_qkv_slice(ggml_context* ctx, ggml_tensor* qkv, int dim, int i) {
    return ggml_view_3d(ctx, qkv, dim, qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2],
                        (size_t)i * dim * ggml_element_size(qkv));
}

// One transformer layer. `x` is [dim, n_tokens, n_seq]; `context` is the projected T5
// states [dim, n_ctx, n_seq] or null; `self_mask` is an additive [n_keys, n_tokens] causal
// mask or null, broadcast across heads and streams (a single query with a full cache needs
// none).
//
// The cache write is expanded into the graph by the caller, which is why the k/v copies are
// returned rather than just chained: ggml only executes what the graph reaches, and the
// attention reads a *view* of the cache rather than the copy's result.
inline ggml_tensor* lm_layer(ggml_context* ctx, const GgufModel& W, const std::string& p,
                             ggml_tensor* x, ggml_tensor* context, ggml_tensor* self_mask,
                             const KvCache& cache, int layer, int n_tokens,
                             std::vector<ggml_tensor*>& cache_writes, const LmConfig& c) {
    const int dim = c.dim;

    // --- self-attention: causal, over the whole history ---
    ggml_tensor* h = layer_norm(ctx, W, p + "norm1", x, c.norm_eps);
    ggml_tensor* qkv = ggml_mul_mat(ctx, W.get(p + "self.qkv.weight"), h);
    ggml_tensor* q = lm_to_attention(ctx, lm_qkv_slice(ctx, qkv, dim, 0), c.head_dim, c.heads);
    ggml_tensor* k = lm_to_attention(ctx, lm_qkv_slice(ctx, qkv, dim, 1), c.head_dim, c.heads);
    ggml_tensor* v = lm_to_attention(ctx, lm_qkv_slice(ctx, qkv, dim, 2), c.head_dim, c.heads);
    cache_writes.push_back(ggml_cpy(ctx, k, cache.k_slot(ctx, layer, n_tokens)));
    // V goes in transposed, which is the layout the second matmul wants back out.
    cache_writes.push_back(ggml_cpy(ctx, ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)),
                                    cache.v_slot(ctx, layer, n_tokens)));

    const float scale = 1.0f / std::sqrt((float)c.head_dim);
    ggml_tensor* o = nn::sdpa_vt(ctx, q, cache.k_history(ctx, layer, n_tokens),
                                 cache.v_history(ctx, layer, n_tokens), self_mask, scale);
    o = ggml_mul_mat(ctx, W.get(p + "self.out.weight"), lm_from_heads(ctx, o, dim));
    x = ggml_add(ctx, x, o);

    // --- cross-attention onto the text, when there is any ---
    //
    // Skipped entirely for the unconditional branch. That is not an approximation: its
    // context is all zeros, neither in_proj nor out_proj has a bias, so every key and value
    // is zero, the softmax is uniform over zeros, and the block's contribution to the
    // residual is exactly 0.
    if (context) {
        ggml_tensor* hc = layer_norm(ctx, W, p + "norm_cross", x, c.norm_eps);
        ggml_tensor* cq = ggml_mul_mat(ctx, W.get(p + "cross.q.weight"), hc);
        ggml_tensor* ckv = ggml_mul_mat(ctx, W.get(p + "cross.kv.weight"), context);
        ggml_tensor* q2 = lm_to_attention(ctx, cq, c.head_dim, c.heads);
        ggml_tensor* k2 = lm_to_attention(ctx, lm_qkv_slice(ctx, ckv, dim, 0),
                                          c.head_dim, c.heads);
        ggml_tensor* v2 = lm_to_attention(ctx, lm_qkv_slice(ctx, ckv, dim, 1),
                                          c.head_dim, c.heads);
        // No mask: audiocraft passes none, so T5 padding is attended over as zero vectors.
        // No zero-attention key either, and no cache: the context is the same every step.
        const float cross_scale = 1.0f / std::sqrt((float)c.head_dim);
        ggml_tensor* kq2 = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k2, q2), nullptr,
                                             cross_scale, 0.0f);
        ggml_tensor* oc = ggml_mul_mat(
            ctx, ggml_cont(ctx, ggml_permute(ctx, v2, 1, 0, 2, 3)), kq2);
        oc = ggml_mul_mat(ctx, W.get(p + "cross.out.weight"), lm_from_heads(ctx, oc, dim));
        x = ggml_add(ctx, x, oc);
    }

    // --- feed-forward ---
    ggml_tensor* f = layer_norm(ctx, W, p + "norm2", x, c.norm_eps);
    return ggml_add(ctx, x, feed_forward(ctx, W, p + "ff.", f));
}

// The logits for the *last* position, [card, n_q, n_seq].
//
// `_sample_next_token` takes `logits[..., -1]`, so the heads only ever need the final
// column. Slicing before `out_norm` turns the prefill's output projection from
// n_tokens x n_q x [card, dim] matmuls into n_q of them -- on a 6 s prompt that is 300
// times less work in the heaviest matrix in the model.
//
// Classifier-free guidance runs as one forward of two sequences rather than two forwards.
// audiocraft does the same (`sequence = torch.cat([sequence, sequence], dim=0)`), and note
// what that concatenation says: **the tokens are identical in both streams**. Only the
// cross-attention context differs. So the embedding lookup happens once and broadcasts, and
// what is duplicated is the position tensor -- which the add below promotes for free.
//
// `ids` is I32 [n_tokens, n_q] (codebook-major); `pos_emb` is [dim, n_tokens, n_seq].
//
// `context` is the cross-attention source **already past `lm.cond_proj`**, [dim, n_ctx,
// n_seq], or null to skip cross-attention entirely. Projected rather than raw because that
// is where audiocraft's null stream is zeroed: `output_proj` has a bias, so a projection of
// zero is not a projected zero. See mg/pipeline.cpp.
inline ggml_tensor* lm_forward(ggml_context* ctx, const GgufModel& W, ggml_tensor* ids,
                               ggml_tensor* pos_emb, ggml_tensor* context,
                               ggml_tensor* self_mask, const KvCache& cache, int n_tokens,
                               std::vector<ggml_tensor*>& cache_writes, const LmConfig& c) {
    // One embedding table per codebook, summed. Each row of `ids` is one codebook's tokens.
    ggml_tensor* x = nullptr;
    for (int k = 0; k < c.n_q; ++k) {
        ggml_tensor* row = ggml_view_1d(ctx, ids, n_tokens,
                                        (size_t)k * n_tokens * ggml_element_size(ids));
        ggml_tensor* e = ggml_get_rows(ctx, W.get("lm.emb." + std::to_string(k) + ".weight"),
                                       row);
        x = x ? ggml_add(ctx, x, e) : e;
    }
    // [dim, n_tokens, n_seq] + [dim, n_tokens] broadcasts the embedding across the guidance
    // streams. The position tensor is the wider operand, so it has to come first.
    x = ggml_add(ctx, pos_emb, x);

    for (int i = 0; i < c.layers; ++i) {
        x = lm_layer(ctx, W, "lm." + std::to_string(i) + ".", x, context, self_mask, cache,
                     i, n_tokens, cache_writes, c);
    }

    // Keep only the last position, then normalize and read the heads.
    x = ggml_cont(ctx, ggml_view_3d(ctx, x, c.dim, 1, x->ne[2], x->nb[1], x->nb[2],
                                    (size_t)(n_tokens - 1) * x->nb[1]));
    x = layer_norm(ctx, W, "lm.out_norm", x, c.norm_eps);

    ggml_tensor* logits = nullptr;
    for (int k = 0; k < c.n_q; ++k) {
        ggml_tensor* head = ggml_mul_mat(ctx, W.get("lm.head." + std::to_string(k) + ".weight"),
                                         x);                          // [card, 1, n_seq]
        logits = logits ? ggml_concat(ctx, logits, head, 1) : head;
    }
    return logits;                                                    // [card, n_q, n_seq]
}

// `lm.cond_proj` on its own: T5 states [cond_dim, n_ctx] -> the model width [dim, n_ctx].
//
// Split out of the forward because the result never changes across a generation, and
// because the null stream needs a *projected* zero rather than a projection of zero.
inline ggml_tensor* lm_project_context(ggml_context* ctx, const GgufModel& W,
                                       ggml_tensor* t5_hidden) {
    return ggml_add(ctx, ggml_mul_mat(ctx, W.get("lm.cond_proj.weight"), t5_hidden),
                    W.get("lm.cond_proj.bias"));
}

// A safe graph size for one forward.
//
// A conditioned layer costs about 69 nodes -- 34 for self-attention including the two cache
// writes and the views around them, 28 for cross-attention, 7 for the feed-forward -- so 24
// layers land near 1660 and the first budget of `layers * 64` tripped ggml's assert on the
// very first forward. ggml counts views as nodes, which is easy to forget when estimating.
inline size_t lm_graph_nodes(const LmConfig& c) {
    return (size_t)c.layers * 96 + (size_t)c.n_q * 16 + 256;
}

} // namespace ac
