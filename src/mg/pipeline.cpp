#include "mg/pipeline.h"

#include "mg/sampling.h"
#include "rng.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ac {

namespace {

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// The decoder: one KV cache, one graph, `n_seq` streams through it.
//
// Classifier-free guidance needs two forward passes over the same weights with the same
// tokens, differing only in the cross-attention context. audiocraft runs them as one batch
// of two -- "it is about x2 faster than doing 2 forward passes", says the comment in
// `LMModel.generate` -- and that is what `n_seq` is here. The cache carries a matching axis,
// so the two histories stay separate while sharing every kernel launch.
//
// The graph is rebuilt for every forward. It has to be: the cache views encode how much
// history there is, and ggml bakes a view's offset and extent in at build time. Building
// costs a fraction of a millisecond against milliseconds of compute, and the alternative --
// attending over the full capacity with a mask and keeping one static graph -- doubles the
// attention work to save it.
//
// The context memory and the allocator are reused across steps, so only the graph objects
// are rebuilt, not the buffers behind them.
class Decoder {
public:
    Decoder(const GgufModel& lm, const LmConfig& c, int capacity, int n_seq,
            const TextCondition& cond, bool cross_attention)
        : lm_(lm), c_(c), n_seq_(n_seq), cross_(cross_attention),
          cache_(lm.backend, c.layers, c.head_dim, c.heads, capacity, n_seq) {
        arena_.resize(ggml_tensor_overhead() * lm_graph_nodes(c) * 2 +
                      ggml_graph_overhead_custom(lm_graph_nodes(c), false) + (1u << 20));
        alloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(lm.backend));
        if (!alloc_) throw std::runtime_error("failed to create the LM graph allocator");
        if (cross_) project_context(cond);
    }
    ~Decoder() {
        if (alloc_) ggml_gallocr_free(alloc_);
    }
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    int forwards() const { return forwards_; }
    size_t cache_bytes() const { return cache_.bytes(); }

    // Feed `n_tokens` sequence steps and return the logits for the last one,
    // [card, n_q, n_seq] -- stream 0 conditional, stream 1 (when present) null.
    const std::vector<float>& forward(const int32_t* ids, int n_tokens) {
        ggml_init_params ip = {arena_.size(), arena_.data(), true};
        ggml_context* ctx = ggml_init(ip);
        if (!ctx) throw std::runtime_error("failed to create the LM graph context");
        struct Release {
            ggml_context* ctx;
            ~Release() { ggml_free(ctx); }
        } release{ctx};

        ggml_tensor* ids_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_tokens, c_.n_q);
        ggml_tensor* pos_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c_.dim, n_tokens, n_seq_);
        ggml_tensor* ctx_t = nullptr;
        ggml_tensor* mask_t = nullptr;
        ggml_set_input(ids_t);
        ggml_set_input(pos_t);
        if (cross_) {
            ctx_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c_.dim, n_ctx_, n_seq_);
            ggml_set_input(ctx_t);
        }
        // A single query against a full cache sees every key, so only the prefill needs a
        // mask. `ac.lm` is causal and nothing else here ever feeds more than one token with
        // history already present.
        if (n_tokens > 1) {
            if (cache_.used() != 0)
                throw std::runtime_error("multi-token forwards are only supported as a prefill");
            mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
            ggml_set_input(mask_t);
        }

        std::vector<ggml_tensor*> writes;
        ggml_tensor* logits = lm_forward(ctx, lm_, ids_t, pos_t, ctx_t, mask_t, cache_,
                                         n_tokens, writes, c_);
        ggml_set_output(logits);
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, lm_graph_nodes(c_), false);
        // The cache writes are side effects: nothing downstream reads their results, so
        // without expanding them explicitly ggml would prune them and the history would
        // never be written.
        for (ggml_tensor* w : writes) ggml_build_forward_expand(graph, w);
        ggml_build_forward_expand(graph, logits);
        if (!ggml_gallocr_alloc_graph(alloc_, graph))
            throw std::runtime_error("failed to allocate the LM graph");

        // Every input goes up on every execution. See docs/MELODYFLOW_EDIT.md: the graph
        // arena reclaims an input's memory as soon as its last consumer has run.
        ggml_backend_tensor_set(ids_t, ids, 0, (size_t)n_tokens * c_.n_q * sizeof(int32_t));
        // The streams share their tokens and therefore their positions -- guidance
        // duplicates the sequence and changes only the context -- so this is one block
        // repeated.
        const std::vector<float> pos = sin_positions(cache_.used(), n_tokens, c_);
        for (int seq = 0; seq < n_seq_; ++seq)
            ggml_backend_tensor_set(pos_t, pos.data(),
                                    (size_t)seq * pos.size() * sizeof(float),
                                    pos.size() * sizeof(float));
        if (ctx_t)
            ggml_backend_tensor_set(ctx_t, context_.data(), 0,
                                    context_.size() * sizeof(float));
        if (mask_t) {
            std::vector<float> mask((size_t)n_tokens * n_tokens, 0.0f);
            for (int qi = 0; qi < n_tokens; ++qi)
                for (int kj = qi + 1; kj < n_tokens; ++kj)
                    mask[(size_t)qi * n_tokens + kj] = -INFINITY;
            ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));
        }

        std::string error;
        if (!graph_compute_checked(lm_.backend, graph, "MusicGen LM", error))
            throw std::runtime_error(error);

        out_.resize((size_t)c_.card * c_.n_q * n_seq_);
        ggml_backend_tensor_get(logits, out_.data(), 0, out_.size() * sizeof(float));
        cache_.advance(n_tokens);
        ++forwards_;
        return out_;
    }

private:
    // `lm.cond_proj` applied once, building the [dim, n_ctx, n_seq] source the layers read.
    //
    // Stream 0 gets the projected text; stream 1 is left at **zero**. The distinction
    // matters: `T5Conditioner.forward` zeroes the null branch *after* `output_proj`, and
    // that projection has a bias, so projecting a zero would leave the bias behind and be a
    // different model. Getting it backwards once cost an afternoon (docs/MUSICGEN_LM.md).
    //
    // Doing it here rather than inside the forward also means it runs once per generation
    // rather than twelve hundred times.
    void project_context(const TextCondition& cond) {
        if (cond.tokens <= 0) throw std::runtime_error("the conditioning has no tokens");
        if (cond.hidden.size() != (size_t)c_.cond_dim * cond.tokens)
            throw std::runtime_error("conditioning does not hold cond_dim * tokens values");
        n_ctx_ = cond.tokens;

        std::vector<uint8_t> arena(ggml_tensor_overhead() * 32 +
                                   ggml_graph_overhead() + (1u << 16));
        ggml_init_params ip = {arena.size(), arena.data(), true};
        ggml_context* ctx = ggml_init(ip);
        if (!ctx) throw std::runtime_error("failed to create the projection context");
        struct Release { ggml_context* ctx; ~Release() { ggml_free(ctx); } } release{ctx};

        ggml_tensor* t5 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c_.cond_dim, n_ctx_);
        ggml_set_input(t5);
        ggml_tensor* projected = ggml_cont(ctx, lm_project_context(ctx, lm_, t5));
        ggml_set_output(projected);
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, projected);

        ggml_gallocr_t alloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(lm_.backend));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
            if (alloc) ggml_gallocr_free(alloc);
            throw std::runtime_error("failed to allocate the context projection");
        }
        ggml_backend_tensor_set(t5, cond.hidden.data(), 0, cond.hidden.size() * sizeof(float));
        std::string error;
        const bool ok = graph_compute_checked(lm_.backend, graph, "cond_proj", error);
        context_.assign((size_t)c_.dim * n_ctx_ * n_seq_, 0.0f);
        if (ok) {
            ggml_backend_tensor_get(projected, context_.data(), 0,
                                    (size_t)c_.dim * n_ctx_ * sizeof(float));
        }
        ggml_gallocr_free(alloc);
        if (!ok) throw std::runtime_error(error);
    }

    const GgufModel& lm_;
    LmConfig c_;
    int n_seq_ = 1;
    bool cross_ = true;
    int n_ctx_ = 0;
    KvCache cache_;
    ggml_gallocr_t alloc_ = nullptr;
    std::vector<uint8_t> arena_;
    std::vector<float> context_;      // [dim, n_ctx, n_seq]; stream 1 stays zero
    std::vector<float> out_;
    int forwards_ = 0;
};

} // namespace

std::vector<int32_t> mg_generate(const GgufModel& lm, const LmConfig& c,
                                 const TextCondition& cond,
                                 const std::vector<int32_t>& prompt_codes, int prompt_len,
                                 const GenerateParams& params, const ProgressFn& progress,
                                 GenerateReport& report) {
    const int K = c.n_q, T = params.max_gen_len;
    if (T <= 0) throw std::runtime_error("max_gen_len must be positive");
    if (prompt_len < 0 || prompt_len >= T)
        throw std::runtime_error("the prompt must be shorter than the generation");
    if (prompt_len > 0 && (int)prompt_codes.size() != K * prompt_len)
        throw std::runtime_error("prompt codes do not hold n_q * prompt_len values");
    if (cond.tokens <= 0) throw std::runtime_error("the conditioning has no tokens");
    if (cond.hidden.size() != (size_t)c.cond_dim * cond.tokens)
        throw std::runtime_error("conditioning does not hold cond_dim * tokens values");

    const DelayPattern pattern = c.pattern(T);
    const int S = pattern.steps();
    const int start = pattern.first_step_with_timestep(prompt_len);

    // -1 marks a timestep the loop has still to produce; the prompt is written up front.
    std::vector<int32_t> codes((size_t)K * T, -1);
    for (int q = 0; q < K; ++q)
        for (int t = 0; t < prompt_len; ++t)
            codes[(size_t)q * T + t] = prompt_codes[(size_t)q * prompt_len + t];

    std::vector<int32_t> sequence =
        build_pattern_sequence(pattern, codes.data(), (int32_t)c.special_token_id);
    const std::vector<uint8_t> mask = pattern_mask(pattern);

    const bool guided = params.cfg_coef != 0.0f;
    // Guidance is two streams in one forward. Without it there is one stream, and its
    // cross-attention can be skipped outright: the null source is all zeros and the
    // attention projections have no biases, so those blocks contribute exactly nothing.
    const int n_seq = guided ? 2 : 1;
    Decoder decoder(lm, c, S, n_seq, cond, /*cross_attention=*/guided);

    Rng rng(params.seed);
    std::vector<float> scratch;
    std::vector<float> combined((size_t)c.card * K);
    std::vector<int32_t> feed((size_t)K * S);   // the widest a single forward can be

    report.prefill_tokens = start;
    // The loop's first iteration is the prefill; the rest are single-token decodes.
    report.decode_steps = S - start - 1;
    int previous = 0;
    double t0 = now_s();

    for (int offset = start; offset < S; ++offset) {
        const int n_tokens = offset - previous;
        for (int q = 0; q < K; ++q)
            std::memcpy(feed.data() + (size_t)q * n_tokens,
                        sequence.data() + (size_t)q * S + previous,
                        (size_t)n_tokens * sizeof(int32_t));

        const std::vector<float>& out = decoder.forward(feed.data(), n_tokens);
        const float* logits = out.data();
        if (guided) {
            // uncond + (cond - uncond) * cfg_coef, exactly as `_sample_next_token` writes
            // it. Stream 0 is the conditional half of the batch, stream 1 the null one.
            const float* cond_logits = out.data();
            const float* null_logits = out.data() + (size_t)c.card * K;
            for (size_t i = 0; i < combined.size(); ++i)
                combined[i] =
                    null_logits[i] + (cond_logits[i] - null_logits[i]) * params.cfg_coef;
            logits = combined.data();
        }

        if (offset == start)
            report.first_logits.assign(logits, logits + (size_t)c.card * K);

        for (int q = 0; q < K; ++q) {
            const float* row = logits + (size_t)q * c.card;
            int token;
            if (!mask[(size_t)q * S + offset]) {
                // This position carries no timestep, so it must read back as the special
                // token; the model never emits it and would otherwise fill the tail with
                // real codes that revert to nothing.
                token = c.special_token_id;
            } else if (params.use_sampling) {
                token = sample_top_k(row, c.card, params.top_k, params.temperature, rng, scratch);
            } else {
                token = argmax_token(row, c.card);
            }
            // Only unknown positions are written: the prompt and the special tokens stay.
            int32_t& slot = sequence[(size_t)q * S + offset];
            if (slot == -1) slot = token;
        }

        previous = offset;
        if (offset == start) {
            report.prefill_seconds = now_s() - t0;
            t0 = now_s();
        }
        if (progress) progress(1 + offset - start, S - start);
    }
    report.decode_seconds = now_s() - t0;
    report.forwards = decoder.forwards();
    report.cache_bytes = decoder.cache_bytes();

    std::vector<int32_t> out =
        revert_pattern_sequence(pattern, sequence.data(), (int32_t)c.special_token_id);
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] >= 0 && out[i] < c.card) continue;
        throw std::runtime_error("the generated sequence has a position the pattern never filled");
    }
    return out;
}

} // namespace ac
