// mg/sampling.h -- how a step's logits become a token.
//
// gary always sends `use_sampling=True`, `top_p=0.0`, `top_k=250`, `temperature=1.0`, so
// top-k is the only stochastic path that matters and top-p is deliberately absent. Greedy
// decoding is here because it is the only mode in which two implementations can be compared
// token for token: sampling depends on the RNG stream, and ours is not torch's.
//
// `audiocraft.utils.utils.sample_top_k`, transcribed:
//
//     top_k_value, _ = torch.topk(probs, k, dim=-1)
//     min_value_top_k = top_k_value[..., [-1]]
//     probs *= (probs >= min_value_top_k).float()
//     probs.div_(probs.sum(dim=-1, keepdim=True))
//     next_token = multinomial(probs, num_samples=1)
//
// The comparison is `>=`, so a tie at the k-th value keeps every token that matches it --
// which can leave more than k candidates. That is faithful, not an oversight.
#pragma once

#include "rng.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ac {

// argmax, which is what audiocraft does when `use_sampling` is off or the temperature is 0.
// Ties go to the lowest index, matching torch.argmax.
inline int argmax_token(const float* logits, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (logits[i] > logits[best]) best = i;
    return best;
}

// softmax(logits / temperature) into `probs`, computed in the max-subtracted form so a
// large logit cannot overflow.
inline void softmax_into(const float* logits, int n, float temperature,
                         std::vector<float>& probs) {
    probs.resize((size_t)n);
    float peak = logits[0];
    for (int i = 1; i < n; ++i) peak = std::max(peak, logits[i]);
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const double e = std::exp(((double)logits[i] - peak) / temperature);
        probs[(size_t)i] = (float)e;
        total += e;
    }
    const float inv = (float)(1.0 / total);
    for (int i = 0; i < n; ++i) probs[(size_t)i] *= inv;
}

// One draw from a categorical distribution given as unnormalized weights.
//
// Not torch's stream -- `src/rng.h` is mt19937 + Box-Muller and reproducible only within
// audiocraft.cpp -- so a seed from gary will not reproduce gary's take. Greedy decoding is
// what the parity check uses; see docs/MUSICGEN_LM.md.
inline int categorical(const std::vector<float>& weights, double total, Rng& rng) {
    if (!(total > 0.0)) throw std::runtime_error("categorical needs a positive total weight");
    const double target = (double)rng.uniform() * total;
    double acc = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        acc += (double)weights[i];
        if (acc > target) return (int)i;
    }
    return (int)weights.size() - 1;   // only reachable through rounding
}

// Zero everything below the k-th largest probability and renormalize, in place.
//
// `k <= 0` or `k >= n` leaves the distribution alone. The cutoff is found by partial
// selection rather than a full sort, and the comparison is `<` -- so ties at the cutoff all
// survive, matching torch's `probs >= min_value_top_k`.
inline void top_k_filter(std::vector<float>& probs, int k) {
    const int n = (int)probs.size();
    if (k <= 0 || k >= n) return;
    std::vector<float> top(probs.begin(), probs.begin() + k);
    std::make_heap(top.begin(), top.end(), std::greater<float>());
    for (int i = k; i < n; ++i) {
        if (probs[(size_t)i] <= top.front()) continue;
        std::pop_heap(top.begin(), top.end(), std::greater<float>());
        top.back() = probs[(size_t)i];
        std::push_heap(top.begin(), top.end(), std::greater<float>());
    }
    const float cutoff = top.front();
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        if (probs[(size_t)i] < cutoff) probs[(size_t)i] = 0.0f;
        total += (double)probs[(size_t)i];
    }
    const float inv = (float)(1.0 / total);
    for (int i = 0; i < n; ++i) probs[(size_t)i] *= inv;
}

// Top-k sampling with temperature. `k <= 0` samples from the full distribution.
inline int sample_top_k(const float* logits, int n, int k, float temperature, Rng& rng,
                        std::vector<float>& scratch) {
    if (!(temperature > 0.0f)) return argmax_token(logits, n);
    softmax_into(logits, n, temperature, scratch);
    top_k_filter(scratch, k);
    return categorical(scratch, 1.0, rng);
}

} // namespace ac
