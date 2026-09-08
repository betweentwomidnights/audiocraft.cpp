// mf/dit.h -- MelodyFlow's flow-matching DiT (audiocraft models/flow.py::FlowModel).
//
// A pre-norm transformer over 128-channel audio latents: RoPE self-attention,
// cross-attention onto projected T5-base states, GELU feed-forward, U-ViT skip connections
// across the stack, and a timestep embedding that is *added* to the normed input before
// each attention block. Not adaLN -- there is no modulation anywhere in this model.
//
// Three details are load-bearing and none of them announce themselves if you get them
// wrong; all three are explained where they happen below:
//   * `add_zero_attn` prepends a zero key/value in both attention blocks (ac/transformer.h)
//   * the rotary frequencies come from the checkpoint, not a closed form (ac/transformer.h)
//   * the skip connections pair layers through `idx % n_skip`, which is not `idx - 13`
//
// Activations are ggml [dim, seq].
#pragma once

#include "ac/transformer.h"
#include "ggml.h"
#include "gguf_model.h"

#include <cmath>
#include <string>
#include <vector>

namespace ac {

struct DitConfig {
    int dim = 1536;
    int layers = 24;
    int heads = 24;
    int head_dim = 64;
    int ff_dim = 6144;
    int latent_dim = 128;
    int cond_dim = 768;      // T5-base hidden size, before dit.cond_proj
    int time_dim = 256;
    int n_skip = 11;
    float rope_base = 10000.0f;
    float norm_eps = 1.0e-5f;
    float cfg_coef = 4.0f;
    float max_duration = 30.0f;
    bool add_zero_attn = true;

    static DitConfig from(const GgufModel& m) {
        if (m.string("ac.architecture") != "audiocraft")
            throw gguf_error("not an audiocraft GGUF");
        if (m.u32("ac.format_version") != 1)
            throw gguf_error("unsupported audiocraft GGUF version");
        if (m.string("ac.dit.family") != "melodyflow")
            throw gguf_error("not a MelodyFlow DiT GGUF");
        DitConfig c;
        c.dim           = (int)m.u32("ac.dit.dim");
        c.layers        = (int)m.u32("ac.dit.layers");
        c.heads         = (int)m.u32("ac.dit.heads");
        c.head_dim      = (int)m.u32("ac.dit.head_dim");
        c.ff_dim        = (int)m.u32("ac.dit.ff_dim");
        c.latent_dim    = (int)m.u32("ac.dit.latent_dim");
        c.cond_dim      = (int)m.u32("ac.dit.cond_dim");
        c.time_dim      = (int)m.u32("ac.dit.time_dim");
        c.n_skip        = (int)m.u32("ac.dit.n_skip");
        c.rope_base     = m.f32("ac.dit.rope_base");
        c.norm_eps      = m.f32("ac.dit.norm_eps");
        c.cfg_coef      = m.f32("ac.dit.cfg_coef");
        c.max_duration  = m.f32("ac.dit.max_duration");
        c.add_zero_attn = m.boolean("ac.dit.add_zero_attn");
        if (c.dim != c.heads * c.head_dim)
            throw gguf_error("DiT dim does not equal heads * head_dim");
        if (c.n_skip != (c.layers - 1) / 2)
            throw gguf_error("DiT skip count does not match (layers - 1) / 2");
        return c;
    }
};

// Which skip projection layer `idx` uses, and whether it uses one at all.
//
// audiocraft's loop is
//     if skip_connections and idx > len(layers) / 2:   x = cat(x, states.pop())
//                                                      x = skip_projections[idx % len(skip_projections)](x)
//     x = layer(x)
//     if skip_connections and idx < len(layers) / 2 - 1: states.append(x)
//
// With 24 layers that pushes the outputs of layers 0..10 and pops them at layers 13..23,
// LIFO -- so layer 13 concatenates layer 10's output and layer 23 concatenates layer 0's.
// The projection index is `idx % 11`, which for 13..23 runs 2,3,...,10,0,1: it is *not*
// aligned with the pairing, and it repeats projections 0 and 1. That looks like a bug in
// the original, but the weights were trained through it, so it is the contract.
struct SkipPlan {
    bool consumes = false;   // this layer pops a saved activation
    bool produces = false;   // this layer's output is saved
    int projection = 0;      // index into dit.skip.* when consuming
};

inline SkipPlan dit_skip_plan(int idx, const DitConfig& c) {
    SkipPlan plan;
    const double half = (double)c.layers / 2.0;
    plan.consumes = c.n_skip > 0 && (double)idx > half;
    plan.produces = c.n_skip > 0 && (double)idx < half - 1.0;
    if (plan.consumes) plan.projection = idx % c.n_skip;
    return plan;
}

// One DiT block. `x` is [dim, seq]; `context` is the projected T5 states [dim, ctx];
// `t_emb` is [dim] broadcast over the sequence; `pos` is I32 [seq]; `cross_mask` is an
// additive [ctx + 1, seq] mask or null (the +1 is the zero key).
inline ggml_tensor* dit_block(ggml_context* ctx, const GgufModel& W, const std::string& p,
                              ggml_tensor* x, ggml_tensor* context, ggml_tensor* t_emb,
                              ggml_tensor* pos, ggml_tensor* freq_factors,
                              ggml_tensor* cross_mask, const DitConfig& c) {
    const int dim = c.dim;
    const int64_t seq = x->ne[1];

    // --- self-attention: pre-norm, plus the timestep, RoPE on q and k ---
    ggml_tensor* h = ggml_add(ctx, layer_norm(ctx, W, p + "norm1", x, c.norm_eps), t_emb);
    ggml_tensor* qkv = ggml_mul_mat(ctx, W.get(p + "self.qkv.weight"), h);   // [3*dim, seq]
    ggml_tensor* q = to_heads(ctx, qkv_slice(ctx, qkv, dim, 0), c.head_dim, c.heads);
    ggml_tensor* k = to_heads(ctx, qkv_slice(ctx, qkv, dim, 1), c.head_dim, c.heads);
    ggml_tensor* v = to_heads(ctx, qkv_slice(ctx, qkv, dim, 2), c.head_dim, c.heads);
    // Rotate in [head_dim, n_head, seq], then move to the attention layout. The zero key is
    // prepended *after* the rotation, as in audiocraft -- rotating zeros would be a no-op
    // anyway, but the ordering is what makes it position-free.
    q = rope_stored(ctx, q, pos, freq_factors, c.head_dim);
    k = rope_stored(ctx, k, pos, freq_factors, c.head_dim);
    ggml_tensor* o = attention(ctx, to_attention(ctx, q), to_attention(ctx, k),
                               to_attention(ctx, v), nullptr, c.add_zero_attn);
    o = ggml_mul_mat(ctx, W.get(p + "self.out.weight"), from_heads(ctx, o, dim));
    x = ggml_add(ctx, x, o);

    // --- cross-attention: pre-norm, plus the timestep again; no rotary here ---
    ggml_tensor* hc = ggml_add(ctx, layer_norm(ctx, W, p + "norm_cross", x, c.norm_eps), t_emb);
    ggml_tensor* cq = ggml_mul_mat(ctx, W.get(p + "cross.q.weight"), hc);         // [dim, seq]
    ggml_tensor* ckv = ggml_mul_mat(ctx, W.get(p + "cross.kv.weight"), context);  // [2*dim, ctx]
    ggml_tensor* q2 = to_attention(ctx, to_heads(ctx, cq, c.head_dim, c.heads));
    ggml_tensor* k2 = to_attention(ctx, to_heads(ctx, qkv_slice(ctx, ckv, dim, 0),
                                                 c.head_dim, c.heads));
    ggml_tensor* v2 = to_attention(ctx, to_heads(ctx, qkv_slice(ctx, ckv, dim, 1),
                                                 c.head_dim, c.heads));
    ggml_tensor* oc = attention(ctx, q2, k2, v2, cross_mask, c.add_zero_attn);
    oc = ggml_mul_mat(ctx, W.get(p + "cross.out.weight"), from_heads(ctx, oc, dim));
    x = ggml_add(ctx, x, oc);

    // --- feed-forward: pre-norm, and note the timestep is NOT added here ---
    ggml_tensor* f = layer_norm(ctx, W, p + "norm2", x, c.norm_eps);
    (void)seq;
    return ggml_add(ctx, x, feed_forward(ctx, W, p + "ff.", f));
}

// The full velocity prediction.
//
// `latent` is [seq, latent_dim] -- the layout the codec produces -- and the returned
// velocity has the same shape. `t5_hidden` is [cond_dim, ctx] straight out of the text
// encoder, projected here by dit.cond_proj. `cross_mask` is additive [ctx + 1, seq], or
// null when every text token is valid.
inline ggml_tensor* dit_forward(ggml_context* ctx, const GgufModel& W, ggml_tensor* latent,
                                ggml_tensor* rescale, ggml_tensor* t_feat,
                                ggml_tensor* t5_hidden, ggml_tensor* pos,
                                ggml_tensor* cross_mask, const DitConfig& c) {
    // FlowModel.forward rescales the input so its standard deviation stays 1 whatever the
    // flow step. `rescale` is an input rather than a baked-in constant so that a whole solve
    // -- 125 forwards for terry -- runs on one graph and one allocation.
    //
    // It carries one value, repeated `latent_dim` times: ggml broadcasts a second operand
    // over the higher dimensions but expects ne0 to match, so a genuine 1-element scalar
    // multiplies only the first channel and silently leaves the rest untouched.
    ggml_tensor* x = ggml_cont(ctx, ggml_transpose(ctx, latent));   // [latent_dim, seq]
    x = ggml_mul_mat(ctx, W.get("dit.in_proj.weight"), ggml_mul(ctx, x, rescale));

    // TimestepEmbedding: sinusoidal features -> Linear -> SiLU -> Linear, no biases.
    ggml_tensor* t_emb = ggml_mul_mat(ctx, W.get("dit.time_embed.0.weight"), t_feat);
    t_emb = ggml_silu(ctx, t_emb);
    t_emb = ggml_mul_mat(ctx, W.get("dit.time_embed.2.weight"), t_emb);

    // The T5 states are projected to the model width once, before any layer sees them.
    ggml_tensor* context = ggml_add(ctx,
        ggml_mul_mat(ctx, W.get("dit.cond_proj.weight"), t5_hidden),
        W.get("dit.cond_proj.bias"));

    ggml_tensor* freq_factors = W.get("dit.rope_freq_factors");

    std::vector<ggml_tensor*> saved;
    for (int idx = 0; idx < c.layers; ++idx) {
        const SkipPlan plan = dit_skip_plan(idx, c);
        if (plan.consumes) {
            if (saved.empty()) throw gguf_error("DiT skip stack underflow");
            ggml_tensor* other = saved.back();
            saved.pop_back();
            x = ggml_concat(ctx, x, other, 0);                       // [2*dim, seq]
            x = ggml_mul_mat(ctx, W.get("dit.skip." + std::to_string(plan.projection) +
                                        ".weight"), x);
        }
        x = dit_block(ctx, W, "dit." + std::to_string(idx) + ".", x, context, t_emb, pos,
                      freq_factors, cross_mask, c);
        if (plan.produces) saved.push_back(x);
    }

    x = layer_norm(ctx, W, "dit.out_norm", x, c.norm_eps);
    x = ggml_mul_mat(ctx, W.get("dit.out_proj.weight"), x);          // [latent_dim, seq]
    // FlowModel scales the output by sqrt(2) so the DiT's output std stays 1.
    x = ggml_scale(ctx, x, std::sqrt(2.0f));
    return ggml_cont(ctx, ggml_transpose(ctx, x));                   // [seq, latent_dim]
}

// The value `rescale` carries: x = z / sqrt(t^2 + (1 - t)^2), flow.py:277. It goes into a
// [latent_dim] tensor -- see dit_forward.
inline float dit_input_rescale(float t) {
    return 1.0f / std::sqrt(t * t + (1.0f - t) * (1.0f - t));
}

// A safe graph size for one velocity prediction.
inline size_t dit_graph_nodes(const DitConfig& c) {
    return (size_t)c.layers * 96 + 256;
}

} // namespace ac
