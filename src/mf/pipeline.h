// mf/pipeline.h -- MelodyFlow's `edit`, assembled.
//
// `MelodyFlow.edit` is two solves over one DiT: the source latent is integrated backwards
// to `target_flowstep` with no conditioning and no guidance, then forwards again to 1.0
// under the target prompt. `mf/solver.h` owns the arithmetic; this owns the ggml side --
// one graph, built once for a fixed sequence length, re-executed for every forward.
//
// The three models are deliberately *not* held at once. terry's window is 30 s, which is
// 750 latent frames, and at that length the DiT and the codec each want gigabytes of
// compute buffer; loading them in turn keeps an 8 GB card viable. So the caller drives the
// staging (T5, then the codec, then the DiT, then the codec again) and this header only
// covers the part that needs the DiT.
#pragma once

#include "ac/conditioner.h"
#include "gguf_model.h"
#include "mf/dit.h"
#include "mf/solver.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ac {

// The text conditioner is shared with MusicGen and lives in ac/conditioner.h; the null
// branch does not need one, because masking every text key to -inf makes cross-attention
// return the zero value vector whatever the context held. See DitRunner below.

// A DiT wired for repeated velocity predictions at one sequence length and one conditioning.
class DitRunner {
public:
    DitRunner(const GgufModel& dit, const DitConfig& config, const TextCondition& cond,
              int frames);
    ~DitRunner();
    DitRunner(const DitRunner&) = delete;
    DitRunner& operator=(const DitRunner&) = delete;

    // One `FlowModel.forward`. `latent` and `velocity` are both [frames, latent_dim].
    // `conditional` picks the cross-attention mask: all text keys visible, or all of them
    // masked out so only the zero-attention key survives.
    void predict(const float* latent, float t, bool conditional, float* velocity);

    // `FlowModel.forward` with guidance folded in, ready to hand to `flow_solve`.
    // `cfg_coef` of 0 runs the unconditional branch alone, which is what the inversion
    // pass does -- terry's `src_descriptions` is `[""]`, and the conditioner masks an
    // empty string out entirely.
    VelocityFn guided(float cfg_coef);

    int forwards() const { return forwards_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int forwards_ = 0;
};

struct EditReport {
    int inversion_forwards = 0;
    int generation_forwards = 0;
    double inversion_seconds = 0.0;
    double generation_seconds = 0.0;
    // The latent at the pivot, between the two passes. Kept because the two passes have
    // very different numerical characters -- see docs/MELODYFLOW_EDIT.md -- and comparing
    // them separately is the only way to tell which one a divergence came from.
    std::vector<float> intermediate;
};

// The DiT's per-channel latent statistics, read to the host. They live in the DiT
// checkpoint, not the codec's, and the solver works entirely in their units.
struct LatentStats {
    std::vector<float> mean, std;
};

LatentStats mf_latent_stats(const GgufModel& dit, int latent_dim);

// `MelodyFlow.edit`'s solver half: a normalized prompt latent [frames, latent_dim] in, the
// edited latent out, still normalized. `noise` is only consulted by the regularized
// inversion; `progress` is called with the same (elapsed, total) accounting terry reports.
std::vector<float> mf_edit_latent(DitRunner& dit, const std::vector<float>& prompt_latent,
                                  int frames, int latent_dim, const EditParams& params,
                                  const NoiseFn& noise, const ProgressFn& progress,
                                  EditReport& report);

} // namespace ac
