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

// One decoding stream: a KV cache plus the graph that feeds it.
//
// The graph is rebuilt for every forward. It has to be: the cache views encode how much
// history there is, and ggml bakes a view's offset and extent in at build time. Building
// costs about 0.3 ms for this stack against several milliseconds of compute, and the
// alternative -- attending over the full capacity with a mask and keeping one static graph
// -- doubles the attention work to save that. Measured before choosing; see
// docs/MUSICGEN_LM.md.
//
// The context memory and the allocator are reused across steps, so only the graph objects
// are rebuilt, not the buffers behind them.
class Stream {
public:
    // `context_width` is 0 to skip cross-attention, `cond_dim` for raw T5 states, or `dim`
    // for a source that is already past `lm.cond_proj`.
    Stream(const GgufModel& lm, const LmConfig& c, int capacity, int context_width)
        : lm_(lm), c_(c), context_width_(context_width),
          cache_(lm.backend, c.layers, c.head_dim, c.heads, capacity) {
        // Room for the graph's tensor structs and the graph itself, reused every step.
        arena_.resize(ggml_tensor_overhead() * lm_graph_nodes(c) * 2 +
                      ggml_graph_overhead_custom(lm_graph_nodes(c), false) + (1u << 20));
        alloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(lm.backend));
        if (!alloc_) throw std::runtime_error("failed to create the LM graph allocator");
    }
    ~Stream() {
        if (alloc_) ggml_gallocr_free(alloc_);
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    int used() const { return cache_.used(); }

    // Feed `n_tokens` sequence steps and return the logits for the last one, [card, n_q].
    const std::vector<float>& forward(const int32_t* ids, int n_tokens,
                                      const std::vector<float>& context, int n_ctx) {
        ggml_init_params ip = {arena_.size(), arena_.data(), true};
        ggml_context* ctx = ggml_init(ip);
        if (!ctx) throw std::runtime_error("failed to create the LM graph context");
        struct Release {
            ggml_context* ctx;
            ~Release() { ggml_free(ctx); }
        } release{ctx};

        ggml_tensor* ids_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_tokens, c_.n_q);
        ggml_tensor* pos_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c_.dim, n_tokens);
        ggml_tensor* ctx_t = nullptr;
        ggml_tensor* mask_t = nullptr;
        ggml_set_input(ids_t);
        ggml_set_input(pos_t);
        if (context_width_ > 0) {
            ctx_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, context_width_, n_ctx);
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
        ggml_tensor* logits = lm_forward(ctx, lm_, ids_t, pos_t, ctx_t,
                                         context_width_ == c_.dim, mask_t, cache_,
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
        const std::vector<float> pos = sin_positions(cache_.used(), n_tokens, c_);
        ggml_backend_tensor_set(pos_t, pos.data(), 0, pos.size() * sizeof(float));
        if (ctx_t)
            ggml_backend_tensor_set(ctx_t, context.data(), 0, context.size() * sizeof(float));
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

        out_.resize((size_t)c_.card * c_.n_q);
        ggml_backend_tensor_get(logits, out_.data(), 0, out_.size() * sizeof(float));
        cache_.advance(n_tokens);
        return out_;
    }

private:
    const GgufModel& lm_;
    LmConfig c_;
    int context_width_ = 0;
    KvCache cache_;
    ggml_gallocr_t alloc_ = nullptr;
    std::vector<uint8_t> arena_;
    std::vector<float> out_;
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
    Stream conditioned(lm, c, S, /*context_width=*/c.cond_dim);
    std::unique_ptr<Stream> unconditioned;
    // The null branch's cross-attention source is all zeros -- `T5Conditioner.forward`
    // zeroes it *after* `output_proj` -- so with no biases in the attention projections the
    // block contributes exactly nothing and can be skipped. `--uncond-cross` builds the
    // blocks anyway and feeds them a projected zero, which is how that claim gets checked
    // rather than asserted. Feeding zeros to `lm.cond_proj` instead would leave its bias
    // behind and is a different model: that mistake cost an afternoon's confusion.
    std::vector<float> zero_context;
    if (guided) {
        const int width = params.uncond_cross ? c.dim : 0;
        unconditioned.reset(new Stream(lm, c, S, width));
        if (params.uncond_cross) zero_context.assign((size_t)c.dim * cond.tokens, 0.0f);
    }

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

        const std::vector<float>& cond_logits =
            conditioned.forward(feed.data(), n_tokens, cond.hidden, cond.tokens);
        ++report.forwards;
        const float* logits = cond_logits.data();
        if (guided) {
            const std::vector<float>& null_logits =
                unconditioned->forward(feed.data(), n_tokens, zero_context, cond.tokens);
            ++report.forwards;
            // uncond + (cond - uncond) * cfg_coef, exactly as `_sample_next_token` writes it.
            for (size_t i = 0; i < combined.size(); ++i)
                combined[i] = null_logits[i] + (cond_logits[i] - null_logits[i]) * params.cfg_coef;
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

    std::vector<int32_t> out =
        revert_pattern_sequence(pattern, sequence.data(), (int32_t)c.special_token_id);
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] >= 0 && out[i] < c.card) continue;
        throw std::runtime_error("the generated sequence has a position the pattern never filled");
    }
    return out;
}

} // namespace ac
