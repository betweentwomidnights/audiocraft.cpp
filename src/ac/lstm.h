// ac/lstm.h -- the 2-layer LSTM that sits in both SEANet codecs' bottleneck.
//
// ggml has no LSTM operation, so this unrolls the recurrence into the graph. The
// alternative -- running it on the host between two half-graphs, the way sa3.cpp stages
// T5 -> DiT -> autoencoder -- was implemented first and measured: a scalar host loop spent
// 6.7 s per direction on MelodyFlow's 30 s window (1024 hidden, 2 layers, 750 steps),
// against 2.9 s for the entire convolutional stack around it. See docs/MELODYFLOW_VAE.md.
//
// Unrolling costs graph size: about 15 nodes per step per layer, so ~22k nodes for a 30 s
// window. In exchange every matmul goes through ggml's threaded, vectorised, backend-
// portable path instead of a hand-rolled GEMV, and the codec stays one graph on any
// backend rather than needing a host round trip in the middle.
//
// Two things make the unroll cheap. The input projection W_ih @ X is not recurrent, so it
// is one matmul over all timesteps per layer -- half the arithmetic, hoisted out of the
// loop. And the converter has already summed torch's two bias vectors, since nn.LSTM adds
// b_ih and b_hh unconditionally on every step.
//
// Matches audiocraft's `StreamableLSTM` (modules/lstm.py): `nn.LSTM(dim, dim, num_layers)`
// over a time-first sequence, plus a residual skip.
#pragma once

#include "ggml.h"
#include "gguf_model.h"

#include <string>

namespace ac {

// Nodes this adds to a graph, so callers can size ggml_new_graph_custom. One step is
// 17 nodes today; the margin is there so a small change to the cell does not turn into a
// GGML_ASSERT(cgraph->n_nodes < cgraph->size) abort deep in a run.
inline size_t lstm_graph_nodes(int layers, int64_t frames) {
    return (size_t)layers * (16 + (size_t)frames * 24) + 16;
}

// One StreamableLSTM. `x` is [hidden, frames] -- channel-fastest, i.e. one contiguous
// vector per timestep, which is what the per-step views below slice. Returns the same
// shape, with the residual skip already applied.
//
// `prefix` is e.g. "codec.encoder.lstm"; layers comes from ac.codec.lstm_layers, and
// layers == 0 is a pass-through.
inline ggml_tensor* lstm_graph(ggml_context* ctx, const GgufModel& W,
                               const std::string& prefix, ggml_tensor* x, int layers) {
    if (layers <= 0) return x;
    const int64_t hidden = x->ne[0];
    const int64_t frames = x->ne[1];
    const int64_t gates = 4 * hidden;

    ggml_tensor* residual = x;
    ggml_tensor* input = x;

    for (int layer = 0; layer < layers; ++layer) {
        const std::string p = prefix + "." + std::to_string(layer) + ".";
        ggml_tensor* w_ih = W.get(p + "w_ih");     // ggml [hidden, 4*hidden]
        ggml_tensor* w_hh = W.get(p + "w_hh");
        ggml_tensor* bias = W.get(p + "bias");     // [4*hidden]

        // The non-recurrent half, for every timestep at once.
        ggml_tensor* projected = ggml_add(ctx, ggml_mul_mat(ctx, w_ih, input), bias);

        // Zero initial hidden and cell state, as nn.LSTM uses when none is supplied.
        ggml_tensor* state_h = ggml_scale(
            ctx, ggml_cont(ctx, ggml_view_1d(ctx, projected, hidden, 0)), 0.0f);
        ggml_tensor* state_c = state_h;

        ggml_tensor* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, frames);
        output = ggml_scale(ctx, output, 0.0f);

        for (int64_t t = 0; t < frames; ++t) {
            ggml_tensor* g = ggml_add(
                ctx,
                ggml_view_1d(ctx, projected, gates, (size_t)t * projected->nb[1]),
                ggml_mul_mat(ctx, w_hh, state_h));

            // torch packs the gates as i, f, g, o.
            auto gate = [&](int index) {
                return ggml_view_1d(ctx, g, hidden, (size_t)index * hidden * sizeof(float));
            };
            ggml_tensor* in_gate     = ggml_sigmoid(ctx, gate(0));
            ggml_tensor* forget_gate = ggml_sigmoid(ctx, gate(1));
            ggml_tensor* cell        = ggml_tanh(ctx, gate(2));
            ggml_tensor* out_gate    = ggml_sigmoid(ctx, gate(3));

            state_c = ggml_add(ctx, ggml_mul(ctx, forget_gate, state_c),
                               ggml_mul(ctx, in_gate, cell));
            state_h = ggml_mul(ctx, out_gate, ggml_tanh(ctx, state_c));

            // In-place: a copying ggml_set_1d would rewrite the whole [hidden, frames]
            // buffer once per step, which is quadratic in the sequence length. Each step
            // writes a disjoint slice and the chain is ordered by its own data dependency.
            output = ggml_set_1d_inplace(ctx, output, state_h,
                                         (size_t)t * hidden * sizeof(float));
        }
        input = output;
    }
    return ggml_add(ctx, input, residual);
}

} // namespace ac
