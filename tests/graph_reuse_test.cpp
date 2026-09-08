// graph_reuse_test -- pin the contract for running one graph many times.
//
// A flow solve runs the same DiT graph 125 times, and MusicGen's LM will run its graph
// ~1500 times. Building and allocating once and re-executing is the whole point, and it
// hides a trap that cost most of a Phase 3 afternoon:
//
//   `ggml_gallocr` hands out ONE arena for the whole graph and reuses memory as soon as a
//   value is dead. An input tensor uploaded once, before the first execution, is therefore
//   only guaranteed to hold its value for that execution: from the second one on, a later
//   node's scratch may be sitting on top of it. Nothing errors. The first forward is
//   right, every one after it is quietly wrong, and a solve looks like a subtly divergent
//   model rather than a memory bug.
//
// `ggml_gallocr_free_node` exempts GGML_TENSOR_FLAG_OUTPUT from reuse and nothing else, so
// whether a given input survives depends entirely on where in the graph it dies and what
// the allocator does with the block afterwards. On MelodyFlow's DiT it does not survive:
// reading `dit.cond_proj`'s input back after one execution shows it fully overwritten. On
// the small graph below it happens to.
//
// So the contract this asserts is the safe one -- re-upload every input before every
// execution -- and the canary underneath only *reports* what happened to an input that was
// not re-uploaded. Printed, never asserted: it is ggml's behaviour rather than ours, it is
// graph-dependent, and a future ggml that preserves inputs must not fail the suite.
#include "gguf_model.h"
#include "test_backend.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check_close(float got, float want, float tol, const char* what, int index) {
    if (std::fabs(got - want) <= tol) return;
    std::fprintf(stderr, "FAIL %s[%d]: got %.9g, want %.9g\n", what, index, got, want);
    ++failures;
}

// A graph shaped like the ones a solve reuses: an input consumed by the *first* node, a
// long chain after it, and an input consumed by the last node.
//
// The early one is the canary. `ggml_gallocr_free_node` exempts GGML_TENSOR_FLAG_OUTPUT
// from reuse and nothing else, so once `gain`'s only consumer has run, its block goes back
// on the free list and the chain behind it is free to land there. `bias` survives simply
// because it is still live when the graph ends -- which is why this hazard looks random:
// whether an input keeps its value depends on where in the graph it happens to die.
struct Reusable {
    static constexpr int64_t kIn = 64, kOut = 32, kChain = 48;

    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_tensor *x = nullptr, *w = nullptr, *gain = nullptr, *bias = nullptr, *y = nullptr;
    ggml_backend_buffer_t weights = nullptr;

    explicit Reusable(ggml_backend_t b) : backend(b) {
        ggml_init_params ip = {(size_t)16 * 1024 * 1024, nullptr, true};
        ctx = ggml_init(ip);
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kIn, kOut);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kIn, 1);
        gain = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kOut, 1);
        bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kOut, 1);
        for (ggml_tensor* t : {x, gain, bias}) ggml_set_input(t);

        // The weight lives in its own buffer, the way a loaded model's does, so only the
        // three graph inputs are exposed to the arena.
        weights = ggml_backend_alloc_ctx_tensors(ctx, backend);

        ggml_tensor* h = ggml_mul(ctx, ggml_mul_mat(ctx, w, x), gain);   // gain dies here
        for (int i = 0; i < kChain; ++i) h = ggml_add(ctx, h, ggml_scale(ctx, h, 0.001f));
        y = ggml_add(ctx, h, bias);                                      // bias lives to the end
        ggml_set_output(y);
        graph = ggml_new_graph_custom(ctx, 4 * kChain + 64, false);
        ggml_build_forward_expand(graph, y);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
            std::fprintf(stderr, "FAIL: could not allocate the reusable graph\n");
            std::abort();
        }
    }
    ~Reusable() {
        if (alloc) ggml_gallocr_free(alloc);
        if (weights) ggml_backend_buffer_free(weights);
        if (ctx) ggml_free(ctx);
    }

    void run(std::vector<float>& out) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL: graph compute failed\n");
            std::abort();
        }
        out.resize((size_t)kOut);
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    }
};

std::vector<float> host_forward(const std::vector<float>& w, const std::vector<float>& x,
                                const std::vector<float>& gain,
                                const std::vector<float>& bias) {
    std::vector<float> out((size_t)Reusable::kOut, 0.0f);
    for (int64_t o = 0; o < Reusable::kOut; ++o) {
        double acc = 0.0;
        for (int64_t i = 0; i < Reusable::kIn; ++i)
            acc += (double)w[(size_t)o * Reusable::kIn + i] * x[(size_t)i];
        double h = acc * (double)gain[(size_t)o];
        for (int i = 0; i < Reusable::kChain; ++i) h = (float)(h + h * 0.001);
        out[(size_t)o] = (float)h + bias[(size_t)o];
    }
    return out;
}

float wobble(int i, int salt) { return std::sin(0.37f * (float)i + 0.11f * (float)salt); }

} // namespace

int main() {
    const int64_t kIn = Reusable::kIn, kOut = Reusable::kOut;
    std::vector<float> w((size_t)kIn * kOut), gain((size_t)kOut), bias((size_t)kOut);
    for (size_t i = 0; i < w.size(); ++i) w[i] = 0.05f * wobble((int)i, 1);
    for (size_t i = 0; i < gain.size(); ++i) gain[i] = 0.5f + 0.25f * wobble((int)i, 2);
    for (size_t i = 0; i < bias.size(); ++i) bias[i] = 0.1f * wobble((int)i, 3);

    for (const AcTestBackend& b : ac_test_backends()) {
        Reusable r(b.backend);
        ggml_backend_tensor_set(r.w, w.data(), 0, w.size() * sizeof(float));

        // --- the contract: re-upload every input, every execution --------------------
        std::vector<float> got;
        for (int run = 0; run < 4; ++run) {
            std::vector<float> x((size_t)kIn);
            for (size_t i = 0; i < x.size(); ++i) x[i] = wobble((int)i, run + 7);
            ggml_backend_tensor_set(r.x, x.data(), 0, x.size() * sizeof(float));
            ggml_backend_tensor_set(r.gain, gain.data(), 0, gain.size() * sizeof(float));
            ggml_backend_tensor_set(r.bias, bias.data(), 0, bias.size() * sizeof(float));
            r.run(got);
            const std::vector<float> want = host_forward(w, x, gain, bias);
            char what[96];
            std::snprintf(what, sizeof(what), "%s reuse run %d", b.name, run);
            for (int64_t o = 0; o < kOut; ++o)
                check_close(got[(size_t)o], want[(size_t)o], 1e-3f, what, (int)o);
        }

        // --- the canary: an input written once, then left alone -----------------------
        //
        // Upload everything, run, then run again touching only `x`. If `gain` survived,
        // the second result is still right. Reported, never asserted: this is ggml's
        // behaviour, not ours, and a future ggml that preserves inputs must not fail here.
        std::vector<float> x0((size_t)kIn), x1((size_t)kIn);
        for (size_t i = 0; i < x0.size(); ++i) { x0[i] = wobble((int)i, 11); x1[i] = wobble((int)i, 13); }
        ggml_backend_tensor_set(r.x, x0.data(), 0, x0.size() * sizeof(float));
        ggml_backend_tensor_set(r.gain, gain.data(), 0, gain.size() * sizeof(float));
        ggml_backend_tensor_set(r.bias, bias.data(), 0, bias.size() * sizeof(float));
        r.run(got);
        ggml_backend_tensor_set(r.x, x1.data(), 0, x1.size() * sizeof(float));
        r.run(got);
        const std::vector<float> want = host_forward(w, x1, gain, bias);
        double worst = 0.0;
        for (int64_t o = 0; o < kOut; ++o)
            worst = std::max(worst, std::fabs((double)got[(size_t)o] - (double)want[(size_t)o]));
        std::printf("  %-12s an input consumed early, left un-refreshed for a second run: "
                    "max abs err %.3e %s\n", b.name, worst,
                    worst <= 1e-3 ? "(survived)" : "(CLOBBERED -- re-upload every input)");
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("graph_reuse_test: ok\n");
    return 0;
}
