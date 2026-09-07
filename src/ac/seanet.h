// ac/seanet.h -- the SEANet convolutional codec shared by both target models.
//
// MelodyFlow's quantizer-free VAE (48 kHz stereo, ratios [8,8,6,5], snake) and MusicGen's
// EnCodec 32 kHz (mono, ratios [8,5,4,4], ELU, RVQ) are the same network with different
// settings, so one graph builder serves both; everything variable comes from the GGUF
// metadata via SeanetConfig.
//
// Audio and feature tensors are ggml [T, C] -- time fastest, channels slowest -- matching
// sa3.cpp's Oobleck convention, so channel parameters broadcast as [1, C].
//
// The bottleneck LSTM is unrolled into the same graph by ac/lstm.h, so an encode or a
// decode is one graph on any backend. seanet_encode_pre/_post are still exposed so a
// caller can tap the bottleneck when bisecting a divergence.
//
// Constant folding lives in the converter (tools/convert_seanet.py): weight_norm is fused,
// snake's reciprocal is precomputed, and ConvTranspose1d weights are stored in the col2im
// layout. This file only composes stock ggml operations.
#pragma once

#include "ggml.h"
#include "ac/lstm.h"
#include "gguf_model.h"
#include "nn.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ac {

struct SeanetConfig {
    std::string quantizer;          // "no_quant" (VAE) or "rvq"
    std::string activation;         // "snake" or "ELU"
    int channels = 2;
    int n_filters = 64;
    int n_residual_layers = 1;
    std::vector<int> ratios;        // decoder order; the encoder walks them reversed
    int kernel_size = 7;
    int residual_kernel_size = 3;
    int last_kernel_size = 7;
    int dilation_base = 2;
    int compress = 2;
    int lstm_layers = 2;
    int latent_dim = 128;           // decoder input
    int encoder_dim = 256;          // encoder output (2 * latent_dim for a VAE)
    int sample_rate = 48000;
    int hop_length = 1920;
    int frame_rate = 25;

    bool snake() const { return activation == "snake"; }

    // Channel count entering the bottleneck: n_filters doubled once per ratio.
    int bottleneck_channels() const {
        int mult = 1;
        for (size_t i = 0; i < ratios.size(); ++i) mult *= 2;
        return n_filters * mult;
    }

    static SeanetConfig from(const GgufModel& m) {
        if (m.string("ac.architecture") != "audiocraft")
            throw gguf_error("not an audiocraft GGUF");
        if (m.u32("ac.format_version") != 1)
            throw gguf_error("unsupported audiocraft GGUF version");
        SeanetConfig c;
        c.quantizer            = m.string("ac.codec.quantizer");
        c.activation           = m.string("ac.codec.activation");
        c.channels             = (int)m.u32("ac.codec.channels");
        c.n_filters            = (int)m.u32("ac.codec.n_filters");
        c.n_residual_layers    = (int)m.u32("ac.codec.n_residual_layers");
        c.kernel_size          = (int)m.u32("ac.codec.kernel_size");
        c.residual_kernel_size = (int)m.u32("ac.codec.residual_kernel_size");
        c.last_kernel_size     = (int)m.u32("ac.codec.last_kernel_size");
        c.dilation_base        = (int)m.u32("ac.codec.dilation_base");
        c.compress             = (int)m.u32("ac.codec.compress");
        c.lstm_layers          = (int)m.u32("ac.codec.lstm_layers");
        c.latent_dim           = (int)m.u32("ac.codec.latent_dim");
        c.encoder_dim          = (int)m.u32("ac.codec.encoder_dim");
        c.sample_rate          = (int)m.u32("ac.codec.sample_rate");
        c.hop_length           = (int)m.u32("ac.codec.hop_length");
        c.frame_rate           = (int)m.u32("ac.codec.frame_rate");
        for (int32_t r : m.i32s("ac.codec.ratios")) c.ratios.push_back((int)r);
        if (c.ratios.empty()) throw gguf_error("codec has no ratios");
        int hop = 1;
        for (int r : c.ratios) hop *= r;
        if (hop != c.hop_length)
            throw gguf_error("codec ratio product disagrees with the declared hop length");
        return c;
    }
};

namespace detail {

inline int64_t ceil_div(int64_t a, int b) { return (a + b - 1) / b; }

} // namespace detail

// The padding audiocraft's StreamableConv1d applies, and the length it produces.
//
// Kept as a pure function because this arithmetic is where the subtle bugs live: the
// padding is asymmetric whenever `padding_total` is odd (MelodyFlow's ratio-5 stage), and
// `extra` is the tail `get_extra_padding_for_conv1d` adds so the final window is full.
// audiocraft applies `extra` as part of the same reflect pad, not as zeros.
struct ConvGeometry {
    int pad_left = 0;
    int pad_right = 0;    // excludes `extra`
    int extra = 0;
    int64_t out_frames = 0;
};

inline ConvGeometry seanet_conv_geometry(int64_t frames, int kernel, int stride,
                                         int dilation) {
    ConvGeometry g;
    const int effective = (kernel - 1) * dilation + 1;
    const int padding_total = effective - stride;
    // ideal_length = ceil(T / stride) * stride, so `extra` is what rounds T up.
    g.extra = (int)(detail::ceil_div(frames, stride) * stride - frames);
    g.pad_right = padding_total / 2;
    g.pad_left = padding_total - g.pad_right;
    g.out_frames = detail::ceil_div(frames, stride);
    return g;
}

// The trim audiocraft's StreamableConvTranspose1d applies after an unpadded
// ConvTranspose1d. ggml_col2im_1d crops symmetrically, so we crop by `pad_right` and drop
// the remaining `pad_left - pad_right` (0 or 1) frames from the left with a view.
struct TransposeGeometry {
    int pad_left = 0;
    int pad_right = 0;
    int64_t col2im_frames = 0;   // what col2im returns when cropped by pad_right
    int64_t out_frames = 0;      // what we want: frames * stride
    int64_t drop_left() const { return col2im_frames - out_frames; }
};

inline TransposeGeometry seanet_transpose_geometry(int64_t frames, int kernel, int stride) {
    TransposeGeometry g;
    const int padding_total = kernel - stride;
    g.pad_right = padding_total / 2;
    g.pad_left = padding_total - g.pad_right;
    g.col2im_frames = (frames - 1) * stride + kernel - 2 * (int64_t)g.pad_right;
    g.out_frames = frames * stride;
    return g;
}

// Snake1d: x + (1 / (alpha + 1e-9)) * sin(alpha * x)^2, with alpha per channel.
// The reciprocal is precomputed by the converter.
inline ggml_tensor* seanet_snake(ggml_context* ctx, ggml_tensor* x,
                                 ggml_tensor* alpha, ggml_tensor* alpha_recip) {
    ggml_tensor* a = ggml_reshape_2d(ctx, alpha, 1, alpha->ne[0]);
    ggml_tensor* r = ggml_reshape_2d(ctx, alpha_recip, 1, alpha_recip->ne[0]);
    ggml_tensor* s = ggml_sin(ctx, ggml_mul(ctx, x, a));
    return ggml_add(ctx, x, ggml_mul(ctx, ggml_sqr(ctx, s), r));
}

// The activation between blocks, selected by the checkpoint. `prefix` names the snake
// parameters; for ELU it is unused because ELU has none.
inline ggml_tensor* seanet_act(ggml_context* ctx, const GgufModel& W,
                               const std::string& prefix, ggml_tensor* x,
                               const SeanetConfig& c) {
    if (!c.snake()) return ggml_elu(ctx, x);
    return seanet_snake(ctx, x, W.get(prefix + ".alpha"), W.get(prefix + ".alpha_recip"));
}

// StreamableConv1d, non-causal (`modules/conv.py`). Padding is asymmetric for odd
// `padding_total`, and an extra tail is added so the last window is full; audiocraft
// applies all of it in one reflect pad, so we do too.
inline ggml_tensor* seanet_conv1d(ggml_context* ctx, const GgufModel& W,
                                  const std::string& prefix, ggml_tensor* x,
                                  int stride, int dilation) {
    ggml_tensor* w = W.get(prefix + ".weight");     // ggml [k, in, out]
    const int64_t frames = x->ne[0];
    const ConvGeometry g = seanet_conv_geometry(frames, (int)w->ne[0], stride, dilation);

    if (g.pad_left || g.pad_right + g.extra) {
        // audiocraft zero-extends first when the input is shorter than the padding; no
        // realistic input for either model gets near that, and ggml_pad_reflect_1d would
        // assert, so say so clearly instead.
        const int max_pad = g.pad_left > g.pad_right + g.extra ? g.pad_left
                                                               : g.pad_right + g.extra;
        if (frames <= max_pad)
            throw std::runtime_error(
                "input of " + std::to_string(frames) + " frames is too short for " + prefix +
                " (needs more than " + std::to_string(max_pad) + "); audiocraft's short-input "
                "zero-extension path is not implemented");
        x = ggml_pad_reflect_1d(ctx, ggml_cont(ctx, x), g.pad_left, g.pad_right + g.extra);
    }
    // nn::conv_1d_f32, not ggml_conv_1d: the latter rounds activations through an F16
    // im2col, and sixteen stacked convolutions turn that into a visible parity loss.
    ggml_tensor* y = nn::conv_1d_f32(ctx, w, x, stride, /*pad=*/0, dilation);
    ggml_tensor* b = W.get(prefix + ".bias");
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
}

// StreamableConvTranspose1d, non-causal. torch runs ConvTranspose1d with no padding and
// then trims (padding_left, padding_right); ggml_col2im_1d crops symmetrically, so we crop
// by the smaller side and drop the remaining element (0 or 1) with a view.
inline ggml_tensor* seanet_conv_transpose1d(ggml_context* ctx, const GgufModel& W,
                                            const std::string& prefix, ggml_tensor* x,
                                            int stride, int out_channels) {
    ggml_tensor* w = W.get(prefix + ".weight_col2im");   // ggml [in, kernel * out]
    const int kernel = (int)(w->ne[1] / out_channels);
    if ((int64_t)kernel * out_channels != w->ne[1])
        throw gguf_error(prefix + ".weight_col2im is not a whole number of output channels");
    const TransposeGeometry g = seanet_transpose_geometry(x->ne[0], kernel, stride);

    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));       // [in, T]
    ggml_tensor* columns = ggml_mul_mat(ctx, w, xt);                // [kernel*out, T]
    ggml_tensor* y = ggml_col2im_1d(ctx, columns, stride, out_channels, g.pad_right);

    if (y->ne[0] != g.col2im_frames)
        throw std::runtime_error("unexpected col2im length for " + prefix);
    if (g.drop_left()) {
        // Only an odd padding_total gets here, leaving one extra frame on the left.
        y = ggml_cont(ctx, ggml_view_2d(ctx, y, g.out_frames, y->ne[1], y->nb[1],
                                        (size_t)g.drop_left() * sizeof(float)));
    }
    ggml_tensor* b = W.get(prefix + ".bias");
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
}

// SEANetResnetBlock: act -> conv(k=residual_kernel_size, dilation) -> act -> conv(k=1),
// added to an identity shortcut (true_skip).
inline ggml_tensor* seanet_residual(ggml_context* ctx, const GgufModel& W,
                                    const std::string& prefix, ggml_tensor* x,
                                    int dilation, const SeanetConfig& c) {
    ggml_tensor* h = seanet_act(ctx, W, prefix + ".snake1", x, c);
    h = seanet_conv1d(ctx, W, prefix + ".conv1", h, /*stride=*/1, dilation);
    h = seanet_act(ctx, W, prefix + ".snake2", h, c);
    h = seanet_conv1d(ctx, W, prefix + ".conv2", h, /*stride=*/1, /*dilation=*/1);
    return ggml_add(ctx, x, h);
}

// --- encoder -------------------------------------------------------------------------
// Everything up to the bottleneck. `audio` is [samples, channels]; returns
// [bottleneck_channels, frames], transposed because that is the layout lstm_graph slices
// per timestep. Exposed separately from seanet_encode so a caller can tap the bottleneck.
inline ggml_tensor* seanet_encode_pre(ggml_context* ctx, const GgufModel& W,
                                      ggml_tensor* audio, const SeanetConfig& c) {
    ggml_tensor* x = seanet_conv1d(ctx, W, "codec.encoder.in", audio, 1, 1);
    for (size_t stage = 0; stage < c.ratios.size(); ++stage) {
        const std::string p = "codec.encoder.stage." + std::to_string(stage);
        // The encoder consumes the ratios in reverse of the decoder's order.
        const int ratio = c.ratios[c.ratios.size() - 1 - stage];
        for (int j = 0; j < c.n_residual_layers; ++j) {
            int dilation = 1;
            for (int d = 0; d < j; ++d) dilation *= c.dilation_base;
            x = seanet_residual(ctx, W, p + ".res." + std::to_string(j), x, dilation, c);
        }
        x = seanet_act(ctx, W, p + ".snake", x, c);
        x = seanet_conv1d(ctx, W, p + ".down", x, /*stride=*/ratio, /*dilation=*/1);
    }
    return ggml_cont(ctx, ggml_transpose(ctx, x));   // [channels, frames]
}

// `features` is [bottleneck_channels, frames] after the LSTM. Returns the latent as
// [frames, encoder_dim] -- `mean||scale` for a quantizer-free codec.
inline ggml_tensor* seanet_encode_post(ggml_context* ctx, const GgufModel& W,
                                       ggml_tensor* features, const SeanetConfig& c) {
    ggml_tensor* x = ggml_cont(ctx, ggml_transpose(ctx, features));   // [frames, channels]
    x = seanet_act(ctx, W, "codec.encoder.out_snake", x, c);
    return seanet_conv1d(ctx, W, "codec.encoder.out", x, 1, 1);
}

// --- decoder -------------------------------------------------------------------------
// `latent` is [frames, latent_dim]; returns [bottleneck_channels, frames], again in the
// layout lstm_graph wants.
inline ggml_tensor* seanet_decode_pre(ggml_context* ctx, const GgufModel& W,
                                      ggml_tensor* latent, const SeanetConfig& c) {
    ggml_tensor* x = seanet_conv1d(ctx, W, "codec.decoder.in", latent, 1, 1);
    return ggml_cont(ctx, ggml_transpose(ctx, x));   // [channels, frames]
}

// `features` is [bottleneck_channels, frames]; returns audio as [samples, channels].
inline ggml_tensor* seanet_decode_post(ggml_context* ctx, const GgufModel& W,
                                       ggml_tensor* features, const SeanetConfig& c) {
    ggml_tensor* x = ggml_cont(ctx, ggml_transpose(ctx, features));   // [frames, channels]
    int mult = c.bottleneck_channels() / c.n_filters;
    for (size_t stage = 0; stage < c.ratios.size(); ++stage) {
        const std::string p = "codec.decoder.stage." + std::to_string(stage);
        const int ratio = c.ratios[stage];
        const int out_channels = c.n_filters * mult / 2;
        x = seanet_act(ctx, W, p + ".snake", x, c);
        x = seanet_conv_transpose1d(ctx, W, p + ".up", x, ratio, out_channels);
        for (int j = 0; j < c.n_residual_layers; ++j) {
            int dilation = 1;
            for (int d = 0; d < j; ++d) dilation *= c.dilation_base;
            x = seanet_residual(ctx, W, p + ".res." + std::to_string(j), x, dilation, c);
        }
        mult /= 2;
    }
    x = seanet_act(ctx, W, "codec.decoder.out_snake", x, c);
    return seanet_conv1d(ctx, W, "codec.decoder.out", x, 1, 1);
}

// --- whole codec ---------------------------------------------------------------------
// The halves above exist so a caller can tap the bottleneck; these are what everything
// else should use.

// audio [samples, channels] -> latent [frames, encoder_dim].
inline ggml_tensor* seanet_encode(ggml_context* ctx, const GgufModel& W,
                                  ggml_tensor* audio, const SeanetConfig& c) {
    ggml_tensor* x = seanet_encode_pre(ctx, W, audio, c);
    x = lstm_graph(ctx, W, "codec.encoder.lstm", x, c.lstm_layers);
    return seanet_encode_post(ctx, W, x, c);
}

// latent [frames, latent_dim] -> audio [samples, channels].
inline ggml_tensor* seanet_decode(ggml_context* ctx, const GgufModel& W,
                                  ggml_tensor* latent, const SeanetConfig& c) {
    ggml_tensor* x = seanet_decode_pre(ctx, W, latent, c);
    x = lstm_graph(ctx, W, "codec.decoder.lstm", x, c.lstm_layers);
    return seanet_decode_post(ctx, W, x, c);
}

// A safe graph size for one encode or decode over `frames` latent frames. The unrolled
// LSTM dominates; the convolutional stack is a few hundred nodes whatever the length.
inline size_t seanet_graph_nodes(const SeanetConfig& c, int64_t frames) {
    return lstm_graph_nodes(c.lstm_layers, frames) + 1024;
}

} // namespace ac
