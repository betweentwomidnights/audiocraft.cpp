// mg/pipeline.h -- MusicGen's autoregressive loop, assembled.
//
// `LMModel.generate` builds the delay-pattern sequence, seeds it with the prompt, and fills
// the rest one sequence step at a time. The first model call is a whole-prefix prefill; the
// rest are single tokens against a KV cache.
//
// Classifier-free guidance runs two streams over the same weights: one conditioned on the
// text, one on nothing. They share their tokens and differ only in the cross-attention
// context, so both go through one forward as a batch of two -- which is what audiocraft
// does, and what the KV cache's sequence axis is for.
#pragma once

#include "gguf_model.h"
#include "ac/callbacks.h"
#include "ac/conditioner.h"
#include "mg/lm.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ac {

struct GenerateParams {
    int max_gen_len = 1500;        // timesteps, i.e. duration * frame_rate
    bool use_sampling = true;
    float temperature = 1.0f;
    int top_k = 250;
    // 0 turns guidance off, which halves the work and is how a run is compared against a
    // reference that was also generated without it.
    float cfg_coef = 3.0f;
    uint64_t seed = 1234;
};

struct GenerateReport {
    size_t cache_bytes = 0;             // both guidance streams' history
    // The combined logits of the very first prediction, [card, n_q]. The smallest thing two
    // implementations can disagree about, and the only way to tell a rounding difference
    // from a bug once greedy decoding has amplified one into a different song.
    std::vector<float> first_logits;
    int prefill_tokens = 0;
    int decode_steps = 0;
    int forwards = 0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
};

// Runs the LM to `params.max_gen_len` timesteps.
//
// `prompt_codes` is [n_q][prompt_len] EnCodec codes to continue from, empty for a pure
// text-to-music generation. The returned codes are [n_q][max_gen_len] and **include** the
// prompt, which is what `generate_continuation` returns (`remove_prompts=False`).
std::vector<int32_t> mg_generate(const GgufModel& lm, const LmConfig& config,
                                 const TextCondition& cond,
                                 const std::vector<int32_t>& prompt_codes, int prompt_len,
                                 const GenerateParams& params, const ProgressFn& progress,
                                 GenerateReport& report);

} // namespace ac
