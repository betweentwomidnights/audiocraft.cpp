// mf/solver.h -- MelodyFlow's ODE solver and its latent regularization.
//
// `FlowModel.generate` (audiocraft models/flow.py) integrates the velocity field from a
// source flow step to a target one. Two directions matter:
//
//   generation  source 0 -> target 1, classifier-free guidance on
//   inversion   source 1 -> target t, guidance forced OFF, latents regularized each step
//
// `MelodyFlow.edit` runs an inversion followed by a generation, which is what terry does.
//
// Everything here is host-side arithmetic over latents that are ~96k floats. The expensive
// part -- the DiT forward -- is injected as a callback, so this file has no ggml in it and
// can be tested without a checkpoint.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ac {

// MSVC does not define M_PI without _USE_MATH_DEFINES, and the schedule is the one place
// that needs it.
constexpr double kPi = 3.14159265358979323846;

enum class FlowSolverKind { Euler, Midpoint };

inline FlowSolverKind parse_flow_solver(const std::string& name) {
    if (name == "euler") return FlowSolverKind::Euler;
    if (name == "midpoint") return FlowSolverKind::Midpoint;
    throw std::runtime_error("solver must be 'euler' or 'midpoint', got '" + name + "'");
}

struct FlowParams {
    FlowSolverKind solver = FlowSolverKind::Euler;
    int steps = 25;
    float source_flowstep = 0.0f;
    float target_flowstep = 1.0f;
    bool regularize = false;
    int regularize_iters = 4;
    int keep_last_k_iters = 2;
    float lambda_kl = 0.2f;
    // Sway sampling (https://arxiv.org/pdf/2410.06885); 0 gives a uniform schedule.
    float sway_coefficient = -0.8f;
    float cfg_coef = 4.0f;

    // Guidance is forced off when integrating backwards, to stop the inversion diverging
    // (flow.py: `cfg_coef = 0.0 if target_flowstep < source_flowstep else self.cfg_coef`).
    bool inverting() const { return target_flowstep < source_flowstep; }
    float effective_cfg_coef() const { return inverting() ? 0.0f : cfg_coef; }
};

// The flow steps the solver visits, `steps + 1` of them.
//
// flow.py builds `arange(0, 1+1e-5, 1/steps)` forwards or `arange(1, -1e-5, -1/steps)`
// backwards, applies the sway warp, then maps [0,1] onto [source, target]. The warp is
// applied to the *unmapped* sequence, so it is symmetric in both directions.
inline std::vector<float> flow_schedule(const FlowParams& p) {
    if (p.steps <= 0) throw std::runtime_error("steps must be positive");
    if (p.solver == FlowSolverKind::Midpoint && p.steps % 2 != 0)
        throw std::runtime_error("the midpoint solver needs an even number of steps");

    const bool forward = p.target_flowstep > p.source_flowstep;
    const double increment = 1.0 / (double)p.steps;
    std::vector<float> schedule;
    schedule.reserve((size_t)p.steps + 1);
    // torch.arange over floats stops just past the endpoint thanks to the 1e-5 slack, so
    // the sequence is exactly steps + 1 long; generate it by index to avoid drift.
    for (int i = 0; i <= p.steps; ++i) {
        const double base = forward ? (double)i * increment : 1.0 - (double)i * increment;
        const double swayed = base + (double)p.sway_coefficient *
                                         (std::cos(kPi * 0.5 * base) - 1.0 + base);
        const double span = forward ? (double)p.target_flowstep - (double)p.source_flowstep
                                    : (double)p.source_flowstep - (double)p.target_flowstep;
        const double offset = forward ? (double)p.source_flowstep : (double)p.target_flowstep;
        schedule.push_back((float)(swayed * span + offset));
    }
    return schedule;
}

// How many DiT evaluations one solve costs, which is what dominates the wall clock.
// Per solver step: `regularize_iters` inner iterations, each one forward for the sequence
// plus another for the regularizing sequence once the KL term switches on, and all of that
// doubled when guidance is active.
inline int flow_forward_count(const FlowParams& p) {
    const int iters = p.regularize ? p.regularize_iters : 1;
    const int threshold = p.regularize ? p.regularize_iters - p.keep_last_k_iters : iters;
    int per_step = 0;
    for (int j = 0; j < iters; ++j) per_step += (j >= threshold) ? 2 : 1;
    if (p.effective_cfg_coef() != 0.0f) per_step *= 2;
    return per_step * p.steps;
}

namespace detail {

// One 4x4 (channel x frame) block's mean and unbiased variance over its 16 elements.
struct BlockStats {
    double mean = 0.0;
    double var = 0.0;
};

inline BlockStats block_stats(const float* x, int frames, int c0, int t0, int block) {
    const int n = block * block;
    double sum = 0.0;
    for (int c = 0; c < block; ++c)
        for (int t = 0; t < block; ++t) sum += x[(size_t)(c0 + c) * frames + (t0 + t)];
    const double mean = sum / n;
    double sq = 0.0;
    for (int c = 0; c < block; ++c) {
        for (int t = 0; t < block; ++t) {
            const double d = x[(size_t)(c0 + c) * frames + (t0 + t)] - mean;
            sq += d * d;
        }
    }
    return {mean, sq / (n - 1)};   // torch.var is unbiased by default
}

} // namespace detail

// One gradient-descent step of `noise_regularization` (audiocraft utils/renoise.py).
//
// The reference builds a KL divergence between patch statistics of the velocity and of a
// reference velocity, backpropagates it, clips the gradient and steps. Autograd is not
// needed: the loss is a closed form over per-patch mean and variance, so its gradient is
// too, and the whole thing is a few passes over ~96k floats.
//
// The patchification looks more elaborate than it is. `patchify_tensor` unfolds by 4 along
// three dims and reshapes to [-1, 4, 4, 4], then `latents_kl_divergence` flattens the last
// two -- so every statistic is taken over one 4-channel x 4-frame block of 16 elements, and
// the outer grouping into patches of four blocks cancels out under the final `.sum()`. Both
// the loss and its gradient are therefore per block. Frames past the last whole block of 4
// are not covered by any patch and get no gradient, which is faithful: `unfold` drops them.
//
// `lambda_ac` is 0 wherever this is called, so the auto-correlation branch -- the only part
// that would consume randomness -- never runs.
inline void kl_regularization_step(std::vector<float>& velocity,
                                   const std::vector<float>& reference,
                                   int channels, int frames, float lambda_kl,
                                   int block = 4) {
    constexpr double kEpsilon = 1e-6;
    constexpr double kGradClip = 100.0;
    const int n = block * block;
    const int c_blocks = channels / block;
    const int t_blocks = frames / block;
    if (c_blocks <= 0 || t_blocks <= 0) return;

    std::vector<float> grad(velocity.size(), 0.0f);
    for (int cb = 0; cb < c_blocks; ++cb) {
        for (int tb = 0; tb < t_blocks; ++tb) {
            const int c0 = cb * block, t0 = tb * block;
            const detail::BlockStats a = detail::block_stats(velocity.data(), frames, c0, t0, block);
            const detail::BlockStats b = detail::block_stats(reference.data(), frames, c0, t0, block);

            // kl = log((v1 + e)/(v0 + e)) + (v0 + (m0 - m1)^2)/(v1 + e) - 1, then |kl|.
            const double v0 = a.var + kEpsilon, v1 = b.var + kEpsilon;
            const double diff = a.mean - b.mean;
            const double kl = std::log(v1 / v0) + (a.var + diff * diff) / v1 - 1.0;
            const double sign = kl < 0.0 ? -1.0 : 1.0;
            const double d_dvar = sign * (-1.0 / v0 + 1.0 / v1);
            const double d_dmean = sign * (2.0 * diff / v1);

            for (int c = 0; c < block; ++c) {
                for (int t = 0; t < block; ++t) {
                    const size_t idx = (size_t)(c0 + c) * frames + (t0 + t);
                    // d mean / dx = 1/n; d var / dx = 2 (x - mean) / (n - 1).
                    double g = d_dmean / n +
                               d_dvar * 2.0 * (velocity[idx] - a.mean) / (n - 1);
                    if (g > kGradClip) g = kGradClip;
                    if (g < -kGradClip) g = -kGradClip;
                    grad[idx] = (float)g;
                }
            }
        }
    }
    for (size_t i = 0; i < velocity.size(); ++i) velocity[i] -= lambda_kl * grad[i];
}

// The full `noise_regularization`: `num_reg_steps` gradient steps, recomputing the loss
// each time because the velocity moves.
inline std::vector<float> noise_regularization(std::vector<float> velocity,
                                               const std::vector<float>& reference,
                                               int channels, int frames, float lambda_kl,
                                               int num_reg_steps = 4) {
    if (lambda_kl <= 0.0f) return velocity;
    for (int i = 0; i < num_reg_steps; ++i)
        kl_regularization_step(velocity, reference, channels, frames, lambda_kl);
    return velocity;
}

// --- the solver itself -------------------------------------------------------------------

// Predicts the velocity field at flow step `t` for one [channels, frames] sequence, with
// classifier-free guidance already folded in. Everything expensive lives behind this.
using VelocityFn = std::function<void(const float* sequence, float t, float* velocity)>;

// Fills `n` floats with standard normal noise. Only the regularized (inversion) pass needs
// one. Injecting it rather than owning it is what makes exact parity testable: torch's
// stream cannot be reproduced here, so both sides read the same dump instead.
using NoiseFn = std::function<void(float* dst, size_t n)>;

using ProgressFn = std::function<void(int done, int total)>;

// `FlowModel.generate`, transcribed.
//
// `prompt` is the starting sequence, already normalized by latent_mean/latent_std, in ggml
// order ([frames, channels] -- channel-major, so index `c * frames + t`). It is also the
// anchor the regularizing sequence is drawn towards, which is why it stays live for the
// whole solve rather than just seeding `gen`.
//
// Two things here look like mistakes and are not:
//   * when regularizing, every forward is evaluated at `schedule[idx + 1]`, not at the
//     current flow step;
//   * the step taken at the end of a regularized step uses the *moving average* of the
//     regularized velocities, while the inner iterations each step from `gen` using the
//     latest one.
inline std::vector<float> flow_solve(FlowParams p,
                                     const std::vector<float>& prompt,
                                     int channels, int frames,
                                     const VelocityFn& predict,
                                     const NoiseFn& noise_fn = {},
                                     const ProgressFn& progress = {}) {
    const size_t n = (size_t)channels * frames;
    if (prompt.size() != n)
        throw std::runtime_error("prompt does not hold channels * frames values");
    if (p.regularize && p.solver == FlowSolverKind::Midpoint)
        throw std::runtime_error("latent regularization only works with the euler solver");
    if (p.keep_last_k_iters > p.regularize_iters)
        throw std::runtime_error("keep_last_k_iters cannot exceed regularize_iters");
    if (!p.regularize) { p.regularize_iters = 1; p.keep_last_k_iters = 0; }
    if (p.regularize && !noise_fn)
        throw std::runtime_error("a regularized solve needs a noise source");

    const int threshold = p.regularize_iters - p.keep_last_k_iters;
    // Weight of iteration jdx in the moving average: jdx / (k * threshold + sum(range(k))).
    // That denominator is exactly the sum of the numerators, so the weights sum to 1.
    double denom = 0.0;
    if (p.regularize) {
        denom = (double)p.keep_last_k_iters * threshold +
                (double)p.keep_last_k_iters * (p.keep_last_k_iters - 1) / 2.0;
        // keep_last_k_iters = 0 leaves the average at zero and the solve takes no step at
        // all; (1, 1) makes the only weight 0/0. audiocraft guards neither.
        if (denom == 0.0)
            throw std::runtime_error("regularize needs keep_last_k_iters >= 1 and "
                                     "regularize_iters > keep_last_k_iters");
    }

    const std::vector<float> schedule = flow_schedule(p);
    const int total = p.steps * p.regularize_iters;

    std::vector<float> gen = prompt;
    std::vector<float> next;                       // the midpoint solver's half step
    std::vector<float> avg(p.regularize ? n : 0, 0.0f);
    std::vector<float> input(n), velocity(n), reg_seq(n), reg_vel(n), noise(n);

    for (int idx = 0; idx < p.steps; ++idx) {
        const float t_now = schedule[(size_t)idx];
        const float t_next = schedule[(size_t)idx + 1];
        const float delta_t = t_next - t_now;
        // Regularized steps evaluate the field at the step they are heading for.
        const float t_eval = p.regularize ? t_next : t_now;

        if (p.solver == FlowSolverKind::Midpoint && idx % 2 == 1) {
            if (next.size() != n) throw std::runtime_error("midpoint half step is missing");
            input = next;
        } else {
            input = gen;
        }

        for (int jdx = 0; jdx < p.regularize_iters; ++jdx) {
            const bool compute_kl = jdx >= threshold;
            if (compute_kl) {
                // A fresh sample from the flow path at t_next, anchored on the prompt. The
                // second, tiny noise term is audiocraft's; it keeps the two sequences from
                // ever coinciding exactly, which would make the KL gradient degenerate.
                noise_fn(noise.data(), n);
                for (size_t i = 0; i < n; ++i)
                    reg_seq[i] = (1.0f - t_next) * noise[i] + t_next * prompt[i];
                noise_fn(noise.data(), n);
                for (size_t i = 0; i < n; ++i) reg_seq[i] += 1.0e-5f * noise[i];
            }
            predict(input.data(), t_eval, velocity.data());
            if (compute_kl) {
                predict(reg_seq.data(), t_eval, reg_vel.data());
                velocity = noise_regularization(std::move(velocity), reg_vel, channels, frames,
                                                p.lambda_kl, 4);
                const double w = (double)jdx / denom;
                for (size_t i = 0; i < n; ++i) avg[i] += (float)((double)velocity[i] * w);
            }
            // Every inner iteration re-steps from `gen`, not from the previous iterate.
            for (size_t i = 0; i < n; ++i) input[i] = gen[i] + velocity[i] * delta_t;
            if (progress) progress(1 + idx * p.regularize_iters + jdx, total);
        }

        if (p.regularize) {
            velocity = avg;
            std::fill(avg.begin(), avg.end(), 0.0f);
        }
        if (p.solver == FlowSolverKind::Midpoint) {
            if (idx % 2 == 0) {
                next.resize(n);
                for (size_t i = 0; i < n; ++i) next[i] = gen[i] + velocity[i] * delta_t;
            } else {
                const float span = t_next - schedule[(size_t)idx - 1];
                for (size_t i = 0; i < n; ++i) gen[i] += velocity[i] * span;
            }
        } else {
            for (size_t i = 0; i < n; ++i) gen[i] += velocity[i] * delta_t;
        }
    }
    return gen;
}

// --- edit: an inversion followed by a generation -----------------------------------------

// `MelodyFlow.edit`'s two solves. The source audio's latent is integrated backwards to
// `target_flowstep` with no conditioning and no guidance, then forwards again to 1.0 under
// the target prompt. `target_flowstep` is the pivot: 0 discards the source entirely, 1
// keeps it, and terry's presets sit between 0.05 and 0.2.
struct EditParams {
    FlowSolverKind solver = FlowSolverKind::Euler;
    int steps = 25;
    float target_flowstep = 0.12f;
    bool regularize = true;
    int regularize_iters = 2;
    int keep_last_k_iters = 1;
    float lambda_kl = 0.2f;
    float cfg_coef = 4.0f;
    float sway_coefficient = -0.8f;
};

inline FlowParams edit_inversion_params(const EditParams& e) {
    if (!(e.target_flowstep >= 0.0f && e.target_flowstep < 1.0f))
        throw std::runtime_error("target_flowstep must be in [0, 1)");
    FlowParams p;
    p.solver = e.solver;
    p.steps = e.steps;
    p.source_flowstep = 1.0f;
    p.target_flowstep = e.target_flowstep;
    p.regularize = e.regularize;
    p.regularize_iters = e.regularize_iters;
    p.keep_last_k_iters = e.keep_last_k_iters;
    p.lambda_kl = e.lambda_kl;
    p.sway_coefficient = e.sway_coefficient;
    p.cfg_coef = e.cfg_coef;      // ignored: inverting() forces guidance off
    return p;
}

inline FlowParams edit_generation_params(const EditParams& e) {
    FlowParams p = edit_inversion_params(e);
    p.source_flowstep = e.target_flowstep;
    p.target_flowstep = 1.0f;
    // `edit` pops `regularize` before the second call, so the constructor default applies.
    p.regularize = false;
    return p;
}

// The DiT evaluations one `edit` costs, inversion plus generation. terry's settings give
// 75 + 50 = 125.
inline int edit_forward_count(const EditParams& e) {
    return flow_forward_count(edit_inversion_params(e)) +
           flow_forward_count(edit_generation_params(e));
}

} // namespace ac
