// ac/conditioner.h -- the T5-base text conditioner both models share.
//
// MelodyFlow and MusicGen use the same encoder, the same tokenizer and the same
// `output_proj` shape; only the width they project to differs, and that projection lives in
// each model's own checkpoint. So what is shared is exactly this: run T5 over one prompt
// and hand back its last hidden state.
//
// The unconditional branch of classifier-free guidance does not need one of these.
// `ClassifierFreeGuidanceDropout` removes the text and `T5Conditioner.forward` zeroes both
// the embeddings and the mask -- note the zeroing happens *after* `output_proj`, bias
// included, so what reaches the model is a projected zero and not a projection of zero.
// Each model then makes the null branch contribute nothing: MelodyFlow by masking every key
// to -inf, MusicGen because zero keys and values through bias-free attention give a zero
// output. Both are documented where they happen.
#pragma once

#include "ac/t5.h"
#include "ac/tokenizer.h"
#include "gguf_model.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ac {

// T5-base's last hidden state for one prompt, [cond_dim, tokens] in ggml order.
struct TextCondition {
    std::vector<float> hidden;
    int tokens = 0;

    // True when the description was the empty string, which is not the same as a short one.
    // `T5Conditioner.tokenize` collects empty entries in `empty_idx` and zeroes their
    // attention mask; `forward` then multiplies the *projected* embeddings by that mask. So
    // what reaches the model is an all-zero context -- identical to the null branch -- and
    // not T5's encoding of an empty prompt, which is a real vector for the EOS token.
    //
    // The test is exact equality with "". `normalize_text` is false in both checkpoints and
    // `word_dropout` only applies while training, so a prompt of nothing but whitespace is a
    // genuine description and stays one.
    bool empty = false;
};

// `T5Conditioner.tokenize`'s `empty_idx` rule, transcribed. A description is empty when it
// is None or exactly "" -- and nothing else. Do not be tempted to trim: `normalize_text` is
// false in both checkpoints, so " " is a description the model was trained to attend to, and
// treating it as empty would silently drop guidance for it.
inline bool description_is_empty(const std::string& prompt) { return prompt.empty(); }

// Runs T5-base over one prompt and releases the encoder before returning. `device` may be
// empty for the default backend.
//
// The encoder is loaded and dropped inside this call on purpose: it is 0.4 GB at F32 and
// the models that consume its output want the whole card to themselves.
inline TextCondition encode_prompt(const std::string& t5_path, const std::string& prompt,
                                   const std::string& device) {
    UnigramTokenizer tokenizer = UnigramTokenizer::load(t5_path.c_str());
    GgufModel t5 = load_gguf(t5_path.c_str(),
                             make_backend(0, device.empty() ? nullptr : device.c_str()));
    const T5EncoderConfig c = T5EncoderConfig::from(t5);
    const std::vector<int32_t> ids = tokenizer.encode(prompt, (int)t5.u32("ac.t5.max_length"));
    const int seq = (int)ids.size();

    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    struct Release {
        ggml_context** ctx;
        ggml_gallocr_t* alloc;
        ~Release() {
            if (*alloc) ggml_gallocr_free(*alloc);
            if (*ctx) ggml_free(*ctx);
        }
    } release{&ctx, &alloc};

    ggml_init_params ip = {(size_t)256 * 1024 * 1024, nullptr, true};
    ctx = ggml_init(ip);
    if (!ctx) throw std::runtime_error("failed to create the T5 graph context");
    ggml_tensor* ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    ggml_tensor* mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
    ggml_tensor* rel_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)seq * seq);
    for (ggml_tensor* t : {ids_t, mask_t, rel_t}) ggml_set_input(t);
    ggml_tensor* hidden = ggml_cont(ctx, t5_encode(ctx, t5, ids_t, mask_t, rel_t, c));
    ggml_set_output(hidden);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, hidden);
    alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(t5.backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph))
        throw std::runtime_error("failed to allocate the T5 graph");

    const std::vector<int32_t> rel =
        t5_relative_position_buckets(seq, c.relative_buckets, c.relative_max_distance);
    // The prompt is tokenized on its own, so there is no padding to mask.
    const std::vector<float> mask((size_t)seq * seq, 0.0f);
    ggml_backend_tensor_set(ids_t, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));
    ggml_backend_tensor_set(rel_t, rel.data(), 0, rel.size() * sizeof(int32_t));
    std::string error;
    if (!graph_compute_checked(t5.backend, graph, "T5 encoder", error))
        throw std::runtime_error(error);

    TextCondition out;
    out.tokens = seq;
    out.empty = description_is_empty(prompt);
    out.hidden.resize((size_t)c.dim * seq);
    ggml_backend_tensor_get(hidden, out.hidden.data(), 0, out.hidden.size() * sizeof(float));
    return out;
}

} // namespace ac
