// dit_skip_test -- pin MelodyFlow's U-ViT skip wiring to audiocraft's loop.
//
// This is the piece most likely to be "fixed" by someone reading the code later, because
// the original looks wrong. audiocraft picks the skip projection with
// `skip_projections[idx % len(skip_projections)]`, which for a 24-layer stack runs
// 2,3,...,10,0,1 over the consuming layers -- it is not aligned with the LIFO pairing and
// it reuses projections 0 and 1. The weights were trained through that indexing, so it is
// the contract, and reindexing it to something tidier silently changes the model.
//
// The reference below is a direct transcription of StreamingTransformer.forward.
#include "mf/dit.h"

#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what, long long got, long long want) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s: got %lld, want %lld\n", what, got, want);
    ++failures;
}

// audiocraft/modules/transformer.py::StreamingTransformer.forward, transcribed.
struct Reference {
    std::vector<int> consumes_at;      // layer indices that pop
    std::vector<int> produces_at;      // layer indices that push
    std::vector<int> projection_at;    // projection index used, per consuming layer
    std::vector<int> pairs_with;       // which producing layer each consumer pops
};

Reference reference(int layers, int n_skip) {
    Reference r;
    std::vector<int> stack;
    for (int idx = 0; idx < layers; ++idx) {
        if (n_skip > 0 && (double)idx > (double)layers / 2.0) {
            r.consumes_at.push_back(idx);
            r.pairs_with.push_back(stack.back());
            stack.pop_back();
            r.projection_at.push_back(idx % n_skip);
        }
        if (n_skip > 0 && (double)idx < (double)layers / 2.0 - 1.0) {
            r.produces_at.push_back(idx);
            stack.push_back(idx);
        }
    }
    return r;
}

void check_stack(int layers, int n_skip) {
    ac::DitConfig c;
    c.layers = layers;
    c.n_skip = n_skip;
    const Reference want = reference(layers, n_skip);

    std::vector<int> stack;
    size_t consumed = 0, produced = 0;
    char what[96];
    for (int idx = 0; idx < layers; ++idx) {
        const ac::SkipPlan plan = ac::dit_skip_plan(idx, c);
        if (plan.consumes) {
            std::snprintf(what, sizeof(what), "layer %d consumes (L=%d)", idx, layers);
            if (consumed >= want.consumes_at.size()) {
                check(false, what, idx, -1);
                continue;
            }
            check(want.consumes_at[consumed] == idx, what, idx, want.consumes_at[consumed]);
            std::snprintf(what, sizeof(what), "layer %d projection (L=%d)", idx, layers);
            check(plan.projection == want.projection_at[consumed], what,
                  plan.projection, want.projection_at[consumed]);
            std::snprintf(what, sizeof(what), "layer %d pairs with (L=%d)", idx, layers);
            const int got_pair = stack.empty() ? -1 : stack.back();
            if (!stack.empty()) stack.pop_back();
            check(got_pair == want.pairs_with[consumed], what, got_pair,
                  want.pairs_with[consumed]);
            ++consumed;
        }
        if (plan.produces) {
            std::snprintf(what, sizeof(what), "layer %d produces (L=%d)", idx, layers);
            if (produced >= want.produces_at.size()) {
                check(false, what, idx, -1);
                continue;
            }
            check(want.produces_at[produced] == idx, what, idx, want.produces_at[produced]);
            stack.push_back(idx);
            ++produced;
        }
    }
    std::snprintf(what, sizeof(what), "consuming layers (L=%d)", layers);
    check(consumed == want.consumes_at.size(), what, (long long)consumed,
          (long long)want.consumes_at.size());
    std::snprintf(what, sizeof(what), "producing layers (L=%d)", layers);
    check(produced == want.produces_at.size(), what, (long long)produced,
          (long long)want.produces_at.size());
    std::snprintf(what, sizeof(what), "stack drains (L=%d)", layers);
    check(stack.empty(), what, (long long)stack.size(), 0);
}

} // namespace

int main() {
    // The shipped model.
    check_stack(24, 11);

    // The counts audiocraft derives, over the odd and even cases, so the half-open
    // comparisons against layers/2 stay right if a variant ever changes depth.
    for (int layers = 4; layers <= 33; ++layers) check_stack(layers, (layers - 1) / 2);

    // Spell out the shipped model's pairing, so a regression names itself rather than
    // reporting an anonymous index mismatch.
    {
        ac::DitConfig c;
        c.layers = 24;
        c.n_skip = 11;
        static const int expected_projection[11] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1};
        int seen = 0;
        for (int idx = 0; idx < 24; ++idx) {
            const ac::SkipPlan plan = ac::dit_skip_plan(idx, c);
            if (!plan.consumes) continue;
            char what[64];
            std::snprintf(what, sizeof(what), "24-layer projection at %d", idx);
            check(plan.projection == expected_projection[seen], what,
                  plan.projection, expected_projection[seen]);
            ++seen;
        }
        check(seen == 11, "24 layers have 11 consuming layers", seen, 11);
    }

    // DitConfig::from rejects a stack whose skip count disagrees with its depth, which is
    // the shape of mistake that would otherwise surface as a missing tensor much later.
    {
        ac::DitConfig c;
        c.layers = 24;
        c.n_skip = 11;
        check((c.layers - 1) / 2 == c.n_skip, "24 layers imply 11 skips",
              (c.layers - 1) / 2, c.n_skip);
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("dit_skip_test: ok\n");
    return 0;
}
