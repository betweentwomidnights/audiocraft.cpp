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

#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
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

} // namespace ac
