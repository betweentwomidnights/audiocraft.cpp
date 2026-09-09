// kv_attention_test -- run the decode step's attention on every hostable backend and
// require them all to agree with the CPU.
//
// This exists because of a bug that every other check in the repo walked straight past.
// MusicGen's greedy tokens were exact against torch on CUDA when guidance was on (n_seq = 2)
// and wrong from the second generated token when it was off (n_seq = 1) -- and the audio was
// audibly worse, not merely different: token entropy 8.38 against the CPU's 7.2, the same to
// within 0.04 across every seed, and no rhythmic structure left in the result.
//
// The shapes below are exactly what `lm_layer` builds for one decode step: a single query
// against a KV cache whose history is a non-contiguous 4D view of a much larger buffer. That
// combination -- a one-column query, a strided history, and a batch axis of 1 -- is the thing
// to keep pinned, because it is the shape the whole autoregressive loop runs in and it is not
// what a full-sequence forward exercises.
#include "mg/kv_cache.h"
#include "nn.h"
#include "test_backend.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int failures = 0;

// One decode step: write this token's k/v into the cache at `used`, then attend over the
// whole history. Returns the attention output, [head_dim, 1, n_head, n_seq].
std::vector<float> decode_step(ggml_backend_t backend, int n_seq, int head_dim, int n_head,
                               int capacity, int used, const std::vector<float>& k_fill,
                               const std::vector<float>& q_in, const std::vector<float>& kv_in) {
    ac::KvCache cache(backend, /*layers=*/1, head_dim, n_head, capacity, n_seq);

    // Prime the history the way a prefill would have, one backend upload per stream so the
    // starting state is identical everywhere and cannot itself be the thing under test.
    ggml_tensor* kt = cache.k(0);
    ggml_tensor* vt = cache.v(0);
    ggml_backend_tensor_set(kt, k_fill.data(), 0, k_fill.size() * sizeof(float));
    ggml_backend_tensor_set(vt, k_fill.data(), 0, k_fill.size() * sizeof(float));
    cache.advance(used);

    std::vector<uint8_t> arena(ggml_tensor_overhead() * 256 + ggml_graph_overhead() + (1u << 20));
    ggml_init_params ip = {arena.size(), arena.data(), true};
    ggml_context* ctx = ggml_init(ip);
    struct Release { ggml_context* c; ~Release() { ggml_free(c); } } release{ctx};

    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, 1, n_head, n_seq);
    ggml_tensor* kv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, 1, n_head, n_seq);
    ggml_set_input(q);
    ggml_set_input(kv);

    ggml_tensor* k_write = ggml_cpy(ctx, kv, cache.k_slot(ctx, 0, 1));
    ggml_tensor* v_write = ggml_cpy(ctx, ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3)),
                                    cache.v_slot(ctx, 0, 1));
    ggml_tensor* o = ac::nn::sdpa_vt(ctx, q, cache.k_history(ctx, 0, 1),
                                     cache.v_history(ctx, 0, 1), nullptr,
                                     1.0f / std::sqrt((float)head_dim));
    ggml_set_output(o);

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, k_write);
    ggml_build_forward_expand(graph, v_write);
    ggml_build_forward_expand(graph, o);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
        std::fprintf(stderr, "FAIL: could not allocate the decode graph\n");
        std::abort();
    }
    ggml_backend_tensor_set(q, q_in.data(), 0, q_in.size() * sizeof(float));
    ggml_backend_tensor_set(kv, kv_in.data(), 0, kv_in.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: decode graph compute failed\n");
        std::abort();
    }
    std::vector<float> out((size_t)ggml_nelements(o));
    ggml_backend_tensor_get(o, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    return out;
}

void check_backends_agree(int n_seq) {
    const int head_dim = 64, n_head = 16, capacity = 1503, used = 301;

    std::mt19937 rng(1234 + n_seq);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    auto fill = [&](size_t n) {
        std::vector<float> v(n);
        for (float& x : v) x = dist(rng);
        return v;
    };
    // The whole cache buffer, so the untouched tail past `used` is real data rather than
    // whatever the allocator left behind -- a kernel that reads past the history view would
    // otherwise be caught only by luck.
    const std::vector<float> k_fill = fill((size_t)head_dim * capacity * n_head * n_seq);
    const std::vector<float> q_in = fill((size_t)head_dim * n_head * n_seq);
    const std::vector<float> kv_in = fill((size_t)head_dim * n_head * n_seq);

    const std::vector<float> want = decode_step(ac_test_cpu_backend(), n_seq, head_dim, n_head,
                                                capacity, used, k_fill, q_in, kv_in);
    for (const AcTestBackend& b : ac_test_backends()) {
        const std::vector<float> got = decode_step(b.backend, n_seq, head_dim, n_head,
                                                   capacity, used, k_fill, q_in, kv_in);
        if (got.size() != want.size()) {
            std::fprintf(stderr, "FAIL n_seq=%d %s: %zu values, want %zu\n",
                         n_seq, b.name, got.size(), want.size());
            ++failures;
            continue;
        }
        double worst = 0.0;
        for (size_t i = 0; i < want.size(); ++i)
            worst = std::max(worst, (double)std::fabs(got[i] - want[i]));
        // Attention over 302 keys in f32; backends reduce in different orders, so this is a
        // tolerance on rounding, not on correctness. The bug it was written for produced
        // errors four orders of magnitude past it.
        const double tol = 2e-4;
        std::printf("  n_seq=%d %-10s max|diff| vs CPU %.3e %s\n",
                    n_seq, b.name, worst, worst <= tol ? "ok" : "FAIL");
        if (worst > tol) ++failures;
    }
}

} // namespace

int main() {
    // n_seq 2 is the guided batch and was always correct; n_seq 1 is the unguided path that
    // regressed. Both are pinned so a fix for one cannot quietly break the other.
    check_backends_agree(1);
    check_backends_agree(2);
    if (failures) {
        std::fprintf(stderr, "kv_attention_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("kv_attention_test: ok\n");
    return 0;
}
