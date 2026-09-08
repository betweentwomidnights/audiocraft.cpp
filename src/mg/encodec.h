// mg/encodec.h -- the residual vector quantizer at EnCodec's bottleneck.
//
// The conv stack is shared (`ac/codec.h`); what MusicGen adds is a discrete bottleneck.
// `n_q` codebooks of `bins` vectors are applied in series: the first quantizes the latent,
// the second quantizes what the first got wrong, and so on. The LM models those code
// indices, which is the only reason MusicGen is a language model at all.
//
//   encode   for each level: pick the nearest codebook vector to the running residual,
//            record its index, subtract it
//   decode   sum one lookup per level
//
// Both run on the host. Decoding is `n_q` lookups per frame, which is nothing; encoding a
// 6 s prompt is 300 frames x 4 levels x 2048 candidates x 128 dims, about 0.3 GFLOP, and
// happens once per request. Moving it into a graph would buy little and cost the round
// trip.
//
// Latents are ggml [frames, dim] -- channel-major, `c * frames + t`. Codes are [n_q][frames].
#pragma once

#include "gguf_model.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace ac {

// The codebooks read to the host, one [bins, dim] table per level.
//
// Stored in the GGUF as ggml [dim, bins], which is row-major [bins][dim] in memory: each
// codebook vector is contiguous, which is what both the search and the lookup want.
struct RvqCodebooks {
    int n_q = 0, bins = 0, dim = 0;
    std::vector<float> embed;          // [n_q][bins][dim]
    // ||e||^2 per vector, precomputed: the nearest-neighbour search needs it every frame
    // and it never changes.
    std::vector<float> norms;          // [n_q][bins]

    const float* vector(int level, int index) const {
        return embed.data() + ((size_t)level * bins + index) * dim;
    }
};

inline RvqCodebooks rvq_load(const GgufModel& codec, int n_q, int bins, int dim) {
    RvqCodebooks q;
    q.n_q = n_q;
    q.bins = bins;
    q.dim = dim;
    q.embed.resize((size_t)n_q * bins * dim);
    q.norms.resize((size_t)n_q * bins);
    for (int level = 0; level < n_q; ++level) {
        ggml_tensor* t = codec.get("codec.rvq." + std::to_string(level) + ".embed");
        if (t->ne[0] != dim || t->ne[1] != bins || t->type != GGML_TYPE_F32)
            throw std::runtime_error("codebook " + std::to_string(level) +
                                     " is not [dim, bins] f32");
        float* dst = q.embed.data() + (size_t)level * bins * dim;
        ggml_backend_tensor_get(t, dst, 0, (size_t)bins * dim * sizeof(float));
        for (int i = 0; i < bins; ++i) {
            double sum = 0.0;
            const float* v = dst + (size_t)i * dim;
            for (int d = 0; d < dim; ++d) sum += (double)v[d] * v[d];
            q.norms[(size_t)level * bins + i] = (float)sum;
        }
    }
    return q;
}

// Codes -> latent: one lookup per level, summed.
inline std::vector<float> rvq_decode(const RvqCodebooks& q, const std::vector<int32_t>& codes,
                                     int frames) {
    if (codes.size() != (size_t)q.n_q * frames)
        throw std::runtime_error("codes do not hold n_q * frames values");
    std::vector<float> latent((size_t)q.dim * frames, 0.0f);
    for (int level = 0; level < q.n_q; ++level) {
        for (int t = 0; t < frames; ++t) {
            const int32_t index = codes[(size_t)level * frames + t];
            if (index < 0 || index >= q.bins)
                throw std::runtime_error("a code is outside [0, bins)");
            const float* v = q.vector(level, index);
            for (int d = 0; d < q.dim; ++d) latent[(size_t)d * frames + t] += v[d];
        }
    }
    return latent;
}

// Latent -> codes, greedily and level by level.
//
// `EncodecEuclideanCodebook.quantize` writes the distance as
// `-(||x||^2 - 2 x.e^T + ||e||^2)` and takes the max. `||x||^2` is the same for every
// candidate, so the search here maximizes `2 x.e - ||e||^2` instead -- same argmax, one
// fewer pass over the frame. Ties go to the lowest index, matching `torch.max`.
inline std::vector<int32_t> rvq_encode(const RvqCodebooks& q, const std::vector<float>& latent,
                                       int frames) {
    if (latent.size() != (size_t)q.dim * frames)
        throw std::runtime_error("latent does not hold dim * frames values");
    std::vector<int32_t> codes((size_t)q.n_q * frames, 0);
    std::vector<float> residual((size_t)q.dim);

    for (int t = 0; t < frames; ++t) {
        for (int d = 0; d < q.dim; ++d) residual[(size_t)d] = latent[(size_t)d * frames + t];
        for (int level = 0; level < q.n_q; ++level) {
            const float* table = q.vector(level, 0);
            const float* norms = q.norms.data() + (size_t)level * q.bins;
            int best = 0;
            float best_score = -3.4e38f;
            for (int i = 0; i < q.bins; ++i) {
                const float* v = table + (size_t)i * q.dim;
                float dot = 0.0f;
                for (int d = 0; d < q.dim; ++d) dot += residual[(size_t)d] * v[d];
                const float score = 2.0f * dot - norms[i];
                if (score > best_score) { best_score = score; best = i; }
            }
            codes[(size_t)level * frames + t] = best;
            const float* chosen = q.vector(level, best);
            for (int d = 0; d < q.dim; ++d) residual[(size_t)d] -= chosen[d];
        }
    }
    return codes;
}

} // namespace ac
