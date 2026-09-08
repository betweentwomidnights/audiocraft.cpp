// mg/kv_cache.h -- the key/value history for the decoding streams.
//
// MusicGen decodes ~1200 steps for a 30 s continuation, and re-attending the whole prefix
// each time would be quadratic. The cache holds every layer's keys and values in a backend
// buffer of its own -- deliberately *not* the graph arena, which `ggml_gallocr` recycles
// between executions (see docs/MELODYFLOW_EDIT.md) -- so a graph can write the current
// step's k/v into it and read the whole history back.
//
// Layouts differ between K and V on purpose:
//
//   K  [head_dim, capacity, n_head, n_seq]   the layout attention wants for `mul_mat(k, q)`
//   V  [capacity, head_dim, n_head, n_seq]   already transposed
//
// V is stored transposed because the second matmul needs it that way. Storing it the
// obvious way costs a `ggml_cont(ggml_permute(...))` over the *entire history* on every
// decode step -- at a full 30 s context that is 148 MB of copy per step across the stack,
// for nothing.
//
// `n_seq` is the classifier-free guidance batch: 2 to run the conditional and null streams
// in one forward, 1 without guidance. It is the last dimension so that a stream's history is
// contiguous, which is what lets every view below be a plain 4D slice.
#pragma once

#include "ggml.h"
#include "gguf_model.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace ac {

class KvCache {
public:
    KvCache(ggml_backend_t backend, int layers, int head_dim, int n_head, int capacity,
            int n_seq)
        : layers_(layers), head_dim_(head_dim), n_head_(n_head), capacity_(capacity),
          n_seq_(n_seq) {
        if (layers <= 0 || head_dim <= 0 || n_head <= 0 || capacity <= 0 || n_seq <= 0)
            throw std::runtime_error("the kv cache needs positive dimensions");
        // Two tensors per layer plus ggml's own per-tensor overhead.
        ggml_init_params ip = {ggml_tensor_overhead() * (size_t)(2 * layers + 8), nullptr, true};
        ctx_ = ggml_init(ip);
        if (!ctx_) throw std::runtime_error("failed to create the kv cache context");
        k_.resize((size_t)layers);
        v_.resize((size_t)layers);
        for (int i = 0; i < layers; ++i) {
            k_[(size_t)i] = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, head_dim, capacity,
                                               n_head, n_seq);
            v_[(size_t)i] = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, capacity, head_dim,
                                               n_head, n_seq);
            ggml_format_name(k_[(size_t)i], "kv.k.%d", i);
            ggml_format_name(v_[(size_t)i], "kv.v.%d", i);
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
        if (!buffer_) throw std::runtime_error("failed to allocate the kv cache buffer");
    }

    ~KvCache() {
        if (buffer_) ggml_backend_buffer_free(buffer_);
        if (ctx_) ggml_free(ctx_);
    }
    KvCache(const KvCache&) = delete;
    KvCache& operator=(const KvCache&) = delete;

    ggml_tensor* k(int layer) const { return k_[(size_t)layer]; }
    ggml_tensor* v(int layer) const { return v_[(size_t)layer]; }

    int capacity() const { return capacity_; }
    int sequences() const { return n_seq_; }
    int used() const { return used_; }
    void reset() { used_ = 0; }
    void advance(int n) {
        if (used_ + n > capacity_)
            throw std::runtime_error("kv cache overflow: the sequence is longer than its capacity");
        used_ += n;
    }

    size_t bytes() const {
        return (size_t)layers_ * 2 * head_dim_ * capacity_ * n_head_ * n_seq_ * sizeof(float);
    }

    // Where this step's keys go: [head_dim, n_tokens, n_head, n_seq] at column `used`.
    ggml_tensor* k_slot(ggml_context* ctx, int layer, int n_tokens) const {
        ggml_tensor* c = k_[(size_t)layer];
        return ggml_view_4d(ctx, c, head_dim_, n_tokens, n_head_, n_seq_,
                            c->nb[1], c->nb[2], c->nb[3], (size_t)used_ * c->nb[1]);
    }
    // And its values, transposed: [n_tokens, head_dim, n_head, n_seq] at row `used`.
    ggml_tensor* v_slot(ggml_context* ctx, int layer, int n_tokens) const {
        ggml_tensor* c = v_[(size_t)layer];
        return ggml_view_4d(ctx, c, n_tokens, head_dim_, n_head_, n_seq_,
                            c->nb[1], c->nb[2], c->nb[3],
                            (size_t)used_ * ggml_element_size(c));
    }

    // The whole history including this step: [head_dim, used + n_tokens, n_head, n_seq].
    ggml_tensor* k_history(ggml_context* ctx, int layer, int n_tokens) const {
        ggml_tensor* c = k_[(size_t)layer];
        return ggml_view_4d(ctx, c, head_dim_, used_ + n_tokens, n_head_, n_seq_,
                            c->nb[1], c->nb[2], c->nb[3], 0);
    }
    // [used + n_tokens, head_dim, n_head, n_seq], the layout nn::sdpa_vt consumes directly.
    ggml_tensor* v_history(ggml_context* ctx, int layer, int n_tokens) const {
        ggml_tensor* c = v_[(size_t)layer];
        return ggml_view_4d(ctx, c, used_ + n_tokens, head_dim_, n_head_, n_seq_,
                            c->nb[1], c->nb[2], c->nb[3], 0);
    }

private:
    int layers_ = 0, head_dim_ = 0, n_head_ = 0, capacity_ = 0, n_seq_ = 1, used_ = 0;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::vector<ggml_tensor*> k_, v_;
};

} // namespace ac
