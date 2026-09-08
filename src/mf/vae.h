// mf/vae.h -- MelodyFlow's quantizer-free SEANet codec, wired up.
//
// `ac/seanet.h` builds the graphs; this runs them, and adds the three host-side pieces that
// sit between the codec and the flow model:
//
//   * `vae_sample`      -- the posterior sample the encoder's mean||scale output implies
//   * `latent_normalize`/`latent_denormalize` -- the DiT works in units of latent_mean and
//     latent_std, which live in the *DiT* checkpoint, not the codec's
//   * `conform_audio`   -- terry's input handling: resample, force stereo, crop to the
//     model window
//
// Latents are ggml [frames, channels]: channel-major, so channel `c` at frame `t` is at
// `c * frames + t`. That is also torch's ravel order for [C, T], which is why the raw f32
// dumps compare directly.
#pragma once

#include "ac/lstm.h"
#include "ac/seanet.h"
#include "audio_post.h"
#include "gguf_model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ac {

// Called with each named intermediate when a caller wants to bisect a divergence.
using TapFn = std::function<void(const char*, const std::vector<float>&)>;

namespace detail {

struct VaeArena {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ~VaeArena() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }
};

struct VaeTap {
    std::string name;
    ggml_tensor* tensor = nullptr;
};

// Build, allocate and run a one-input graph, returning the output on the host. `build` may
// append to `taps`; each tap is marked as a graph output and reported when `tap` is set.
inline std::vector<float> vae_run_graph(
    const GgufModel& W, const char* what, int64_t in0, int64_t in1,
    const std::vector<float>& input,
    const std::function<ggml_tensor*(ggml_context*, ggml_tensor*, std::vector<VaeTap>&)>& build,
    int64_t& out0, int64_t& out1, const TapFn& tap, size_t nodes, size_t arena_mb = 4096) {
    VaeArena arena;
    ggml_init_params ip = {arena_mb * 1024 * 1024, nullptr, true};
    arena.ctx = ggml_init(ip);
    if (!arena.ctx) throw std::runtime_error("failed to create the graph context");
    ggml_tensor* x = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, in0, in1);
    ggml_set_input(x);
    std::vector<VaeTap> taps;
    ggml_tensor* y = ggml_cont(arena.ctx, build(arena.ctx, x, taps));
    ggml_set_output(y);
    ggml_cgraph* graph = ggml_new_graph_custom(arena.ctx, nodes, false);
    if (tap) {
        for (VaeTap& t : taps) {
            ggml_set_output(t.tensor);
            ggml_build_forward_expand(graph, t.tensor);
        }
    }
    ggml_build_forward_expand(graph, y);
    arena.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    if (!arena.alloc || !ggml_gallocr_alloc_graph(arena.alloc, graph))
        throw std::runtime_error(std::string("failed to allocate the ") + what + " graph");

    if (input.size() != (size_t)in0 * in1)
        throw std::runtime_error(std::string(what) + ": input size does not match its shape");
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    std::string error;
    if (!graph_compute_checked(W.backend, graph, what, error)) throw std::runtime_error(error);

    if (tap) {
        for (const VaeTap& t : taps) {
            std::vector<float> host((size_t)ggml_nelements(t.tensor));
            ggml_backend_tensor_get(t.tensor, host.data(), 0, host.size() * sizeof(float));
            tap(t.name.c_str(), host);
        }
    }
    out0 = y->ne[0];
    out1 = y->ne[1];
    std::vector<float> host((size_t)out0 * out1);
    ggml_backend_tensor_get(y, host.data(), 0, host.size() * sizeof(float));
    return host;
}

} // namespace detail

// Encode planar audio [samples, channels] to the encoder's mean||scale output, which is
// [frames, 2 * latent_dim].
inline std::vector<float> vae_encode(const GgufModel& codec, const SeanetConfig& c,
                                     const std::vector<float>& audio, int n_samples,
                                     int n_channels, const TapFn& tap = {}) {
    if (n_channels != c.channels)
        throw std::runtime_error("audio channel count does not match the codec's");
    if (n_samples % c.hop_length)
        throw std::runtime_error("sample count is not a whole number of codec frames");
    const int frames = n_samples / c.hop_length;
    int64_t o0 = 0, o1 = 0;
    std::vector<float> latent = detail::vae_run_graph(
        codec, "SEANet encoder", n_samples, n_channels, audio,
        [&](ggml_context* ctx, ggml_tensor* x, std::vector<detail::VaeTap>& taps) {
            ggml_tensor* h = seanet_encode_pre(ctx, codec, x, c);
            taps.push_back({"mf_vae_enc_pre", h});
            h = lstm_graph(ctx, codec, "codec.encoder.lstm", h, c.lstm_layers);
            taps.push_back({"mf_vae_enc_lstm", h});
            return seanet_encode_post(ctx, codec, h, c);
        }, o0, o1, tap, seanet_graph_nodes(c, frames));
    if (o0 != frames || o1 != c.encoder_dim)
        throw std::runtime_error("encoder produced an unexpected latent shape");
    return latent;
}

// Decode a latent [frames, latent_dim] to planar audio [samples, channels].
inline std::vector<float> vae_decode(const GgufModel& codec, const SeanetConfig& c,
                                     const std::vector<float>& latent, int frames,
                                     int64_t& out_samples, const TapFn& tap = {}) {
    if (latent.size() != (size_t)frames * c.latent_dim)
        throw std::runtime_error("latent does not hold frames * latent_dim values");
    int64_t o0 = 0, o1 = 0;
    std::vector<float> audio = detail::vae_run_graph(
        codec, "SEANet decoder", frames, c.latent_dim, latent,
        [&](ggml_context* ctx, ggml_tensor* x, std::vector<detail::VaeTap>& taps) {
            ggml_tensor* h = seanet_decode_pre(ctx, codec, x, c);
            taps.push_back({"mf_vae_dec_pre", h});
            h = lstm_graph(ctx, codec, "codec.decoder.lstm", h, c.lstm_layers);
            taps.push_back({"mf_vae_dec_lstm", h});
            return seanet_decode_post(ctx, codec, h, c);
        }, o0, o1, tap, seanet_graph_nodes(c, frames));
    if (o1 != c.channels)
        throw std::runtime_error("decoder produced an unexpected channel count");
    out_samples = o0;
    return audio;
}

// `audiocraft.utils.utils.vae_sample`: randn_like(mean) * (softplus(scale) + 1e-4) + mean.
//
// `noise` supplies frames * latent_dim standard normals. It is a parameter rather than an
// internal RNG because this draw is the first thing terry's `torch.manual_seed` feeds, and
// parity runs need to replay exactly the same numbers.
inline std::vector<float> vae_sample(const std::vector<float>& encoded, int frames,
                                     int latent_dim, const std::vector<float>& noise) {
    const size_t n = (size_t)frames * latent_dim;
    if (encoded.size() != n * 2)
        throw std::runtime_error("encoder output does not hold mean and scale");
    if (noise.size() != n) throw std::runtime_error("noise does not hold frames * latent_dim");
    std::vector<float> z(n);
    for (int ch = 0; ch < latent_dim; ++ch) {
        const float* mean = encoded.data() + (size_t)ch * frames;
        const float* scale = encoded.data() + (size_t)(ch + latent_dim) * frames;
        const float* eps = noise.data() + (size_t)ch * frames;
        float* dst = z.data() + (size_t)ch * frames;
        for (int t = 0; t < frames; ++t) {
            // softplus, in the form that does not overflow for large |scale|.
            const float stdev = std::log1p(std::exp(-std::fabs(scale[t]))) +
                                std::max(scale[t], 0.0f) + 1e-4f;
            dst[t] = eps[t] * stdev + mean[t];
        }
    }
    return z;
}

// (z - latent_mean) / (latent_std + 1e-5), and its inverse. The statistics are per channel
// and live in the DiT checkpoint: the codec knows nothing about them.
inline void latent_normalize(std::vector<float>& z, int frames, int latent_dim,
                             const std::vector<float>& mean, const std::vector<float>& std) {
    if ((int)mean.size() != latent_dim || (int)std.size() != latent_dim)
        throw std::runtime_error("latent statistics do not match latent_dim");
    for (int ch = 0; ch < latent_dim; ++ch) {
        const float inv = 1.0f / (std[(size_t)ch] + 1e-5f);
        float* dst = z.data() + (size_t)ch * frames;
        for (int t = 0; t < frames; ++t) dst[t] = (dst[t] - mean[(size_t)ch]) * inv;
    }
}

inline void latent_denormalize(std::vector<float>& z, int frames, int latent_dim,
                               const std::vector<float>& mean, const std::vector<float>& std) {
    if ((int)mean.size() != latent_dim || (int)std.size() != latent_dim)
        throw std::runtime_error("latent statistics do not match latent_dim");
    for (int ch = 0; ch < latent_dim; ++ch) {
        const float scale = std[(size_t)ch] + 1e-5f;
        float* dst = z.data() + (size_t)ch * frames;
        for (int t = 0; t < frames; ++t) dst[t] = dst[t] * scale + mean[(size_t)ch];
    }
}

// terry's input handling, in one place: resample to the codec's rate, force the codec's
// channel count (mono is duplicated, extra channels dropped), then crop to a whole number
// of frames and at most `max_frames` of them.
//
// `audio` is planar [samples, channels]; `n_samples`, `n_channels` and `rate` are updated.
inline void conform_audio(std::vector<float>& audio, int& n_samples, int& n_channels,
                          int& rate, const SeanetConfig& c, int max_frames) {
    if (rate != c.sample_rate) {
        int resampled = 0;
        audio = resample_planar_linear(audio, n_samples, n_channels, rate, c.sample_rate,
                                       resampled);
        n_samples = resampled;
        rate = c.sample_rate;
    }
    if (n_channels != c.channels) {
        std::vector<float> fixed((size_t)n_samples * c.channels);
        for (int ch = 0; ch < c.channels; ++ch) {
            const int src = n_channels == 1 ? 0 : (ch < n_channels ? ch : n_channels - 1);
            std::copy_n(audio.data() + (size_t)src * n_samples, n_samples,
                        fixed.data() + (size_t)ch * n_samples);
        }
        audio.swap(fixed);
        n_channels = c.channels;
    }
    int keep = n_samples;
    if (max_frames > 0) keep = std::min(keep, max_frames * c.hop_length);
    keep -= keep % c.hop_length;
    if (keep <= 0) throw std::runtime_error("input is shorter than one codec frame");
    if (keep != n_samples) {
        std::vector<float> cropped((size_t)keep * n_channels);
        for (int ch = 0; ch < n_channels; ++ch)
            std::copy_n(audio.data() + (size_t)ch * n_samples, keep,
                        cropped.data() + (size_t)ch * keep);
        audio.swap(cropped);
        n_samples = keep;
    }
}

} // namespace ac
