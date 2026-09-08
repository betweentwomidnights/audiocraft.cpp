// mf/vae.h -- what MelodyFlow's codec does that MusicGen's does not.
//
// The conv stack, the LSTM and the audio conforming are shared and live in `ac/codec.h`.
// What is specific to MelodyFlow is the bottleneck: its codec has no quantizer, so the
// encoder emits `mean||scale` and generation samples the posterior, and the DiT works in
// units of latent statistics that live in the *DiT's* checkpoint rather than the codec's.
#pragma once

#include "ac/codec.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ac {

// `codec_encode` under its MelodyFlow name, so the parity drivers keep reading the way the
// docs describe them.
inline std::vector<float> vae_encode(const GgufModel& codec, const SeanetConfig& c,
                                     const std::vector<float>& audio, int n_samples,
                                     int n_channels, const TapFn& tap = {}) {
    return codec_encode(codec, c, audio, n_samples, n_channels, tap);
}

inline std::vector<float> vae_decode(const GgufModel& codec, const SeanetConfig& c,
                                     const std::vector<float>& latent, int frames,
                                     int64_t& out_samples, const TapFn& tap = {}) {
    return codec_decode(codec, c, latent, frames, out_samples, tap);
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

} // namespace ac
