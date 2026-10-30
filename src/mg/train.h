// Teacher-forced generation training over MusicGen's delayed codebooks.
// Optional prefix masking permits matched continuation-focused ablations.
#pragma once
#include "mg/lm.h"
#include "ac/train_utils.h"
#include "ggml-alloc.h"
#include <filesystem>
#include <random>

namespace ac {

inline ggml_tensor *mg_train_forward(ggml_context *ctx, const GgufModel &W, const LmConfig &c,
                                     ggml_tensor *ids, ggml_tensor *positions, ggml_tensor *context,
                                     ggml_tensor *causal, const LoraMap *adapters) {
    const int steps = (int)ids->ne[0];
    ggml_tensor *x = nullptr;
    for (int q = 0; q < c.n_q; ++q) {
        auto row = ggml_view_1d(ctx, ids, steps, (size_t)q * steps * sizeof(int32_t));
        auto e = ggml_get_rows(ctx, W.get("lm.emb." + std::to_string(q) + ".weight"), row);
        x = x ? ggml_add(ctx, x, e) : e;
    }
    x = ggml_add(ctx, x, positions);
    auto heads = [&](ggml_tensor *t) { return to_attention(ctx, to_heads(ctx, t, c.head_dim, c.heads)); };
    for (int i = 0; i < c.layers; ++i) {
        const auto p = "lm." + std::to_string(i) + ".";
        auto h = train_layer_norm(ctx, W, p + "norm1", x, c.norm_eps);
        auto qkv = lora_linear(ctx, W, p + "self.qkv.weight", h, adapters);
        auto o = attention(ctx, heads(qkv_slice(ctx, qkv, c.dim, 0)), heads(qkv_slice(ctx, qkv, c.dim, 1)),
                           heads(qkv_slice(ctx, qkv, c.dim, 2)), causal, false, true);
        x = ggml_add(ctx, x, lora_linear(ctx, W, p + "self.out.weight", from_heads(ctx, o, c.dim), adapters));
        h = train_layer_norm(ctx, W, p + "norm_cross", x, c.norm_eps);
        auto q = lora_linear(ctx, W, p + "cross.q.weight", h, adapters);
        auto kv = lora_linear(ctx, W, p + "cross.kv.weight", context, adapters);
        o = attention(ctx, heads(q), heads(qkv_slice(ctx, kv, c.dim, 0)), heads(qkv_slice(ctx, kv, c.dim, 1)),
                      nullptr, false, true);
        x = ggml_add(ctx, x,
                     lora_linear(ctx, W, p + "cross.out.weight", from_heads(ctx, o, c.dim), adapters));
        h = train_layer_norm(ctx, W, p + "norm2", x, c.norm_eps);
        h = train_gelu(ctx, lora_linear(ctx, W, p + "ff.0.weight", h, adapters));
        x = ggml_add(ctx, x, lora_linear(ctx, W, p + "ff.2.weight", h, adapters));
    }
    x = train_layer_norm(ctx, W, "lm.out_norm", x, c.norm_eps);
    ggml_tensor *logits = nullptr;
    for (int q = 0; q < c.n_q; ++q) {
        auto y = ggml_mul_mat(ctx, W.get("lm.head." + std::to_string(q) + ".weight"), x);
        logits = logits ? ggml_concat(ctx, logits, y, 1) : y;
    }
    return logits; // [card, sequence steps * codebooks], codebook-major.
}

struct MgTrainGraph {
    ggml_context *ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *selected = nullptr, *ids = nullptr, *positions = nullptr, *context = nullptr,
                *causal = nullptr, *target = nullptr, *loss = nullptr, *logits = nullptr;
    LoraMap adapters;
    std::vector<MfTrainParam> params;
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> constants;
    const GgufModel &model;
    LmConfig config;
    int frames, tokens, prefix_frames, steps, rank;
    float alpha;
    std::string adapter_type;

    MgTrainGraph(const GgufModel &W, const LmConfig &c, int frames_, int tokens_, int prefix_, int rank_,
                 float alpha_, int first_layer, uint32_t seed, std::string type = "dora-rows", bool feedforward = false)
        : model(W), config(c), frames(frames_), tokens(tokens_), prefix_frames(prefix_), rank(rank_),
          alpha(alpha_), adapter_type(std::move(type)) {
        if (frames <= 0 || tokens <= 0 || prefix_frames < 0 || prefix_frames >= frames || rank <= 0 ||
            rank > c.dim || !std::isfinite(alpha) || alpha <= 0 || first_layer < 0 ||
            first_layer >= c.layers || (adapter_type != "lora" && adapter_type != "dora-rows"))
            throw std::runtime_error("invalid MusicGen training configuration");
        steps = c.pattern(frames).steps() - 1;
        try {
            ctx = ggml_init({64 * 1024 * 1024, nullptr, true});
            if (!ctx)
                throw std::runtime_error("training context allocation failed");
            ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, steps, c.n_q);
            positions = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.dim, steps);
            context = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.dim, tokens);
            causal = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, steps, steps);
            target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.card, c.n_q * (frames - prefix_frames));
            selected = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_q * (frames - prefix_frames));
            for (auto t : {ids, positions, context, causal, target, selected})
                ggml_set_input(t);
            std::mt19937 rng(seed);
            std::normal_distribution<float> normal(0, 0.02f);
            for (int layer = first_layer; layer < c.layers; ++layer)
                for (const char *suffix : {"self.qkv", "self.out", "cross.q", "cross.kv", "cross.out", "ff.0", "ff.2"}) {
                    if(!feedforward && std::string(suffix).rfind("ff.",0)==0) continue;
                    const auto name = "lm." + std::to_string(layer) + "." + suffix + ".weight";
                    auto base = W.get(name);
                    auto A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, base->ne[0], rank);
                    auto B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, base->ne[1]);
                    adapters[name] = {A, B, alpha / rank};
                    std::vector<ggml_tensor *> pp{A, B};
                    std::vector<float> norm_sq;
                    if (adapter_type == "dora-rows") {
                        norm_sq = lora_row_norm_sq(base);
                        auto mag = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, base->ne[1]);
                        auto norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, base->ne[1]);
                        ggml_set_input(norm);
                        constants.push_back({norm, norm_sq});
                        adapters[name].magnitude = mag;
                        adapters[name].base_norm_sq = norm;
                        pp.push_back(mag);
                    }
                    for (auto p : pp) {
                        ggml_set_param(p);
                        ggml_set_input(p);
                        MfTrainParam hp;
                        hp.tensor = p;
                        hp.name = name + (p == A ? ".lora_A" : p == B ? ".lora_B" : ".magnitude");
                        ggml_set_name(p, hp.name.c_str());
                        hp.value.resize((size_t)ggml_nelements(p), 0);
                        if (p == A)
                            for (auto &v : hp.value)
                                v = normal(rng);
                        if (p != A && p != B)
                            for (size_t i = 0; i < hp.value.size(); ++i)
                                hp.value[i] = std::sqrt(norm_sq[i]);
                        hp.m.resize(hp.value.size(), 0);
                        hp.v.resize(hp.value.size(), 0);
                        params.push_back(std::move(hp));
                    }
                }
            logits = mg_train_forward(ctx, W, c, ids, positions, context, causal, &adapters);
            ggml_set_output(logits);
            // Gather only real continuation positions: CE backward assumes unit-sum labels.
            loss = ggml_cross_entropy_loss(ctx, ggml_get_rows(ctx, logits, selected), target);
            ggml_set_loss(loss);
            ggml_set_output(loss);
            graph = ggml_new_graph_custom(ctx, 65536, true);
            ggml_build_forward_expand(graph, loss);
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto n = ggml_graph_node(graph, i);
                if (n->op == GGML_OP_MUL_MAT)
                    ggml_mul_mat_set_prec(n, GGML_PREC_F32);
            }
            ggml_build_backward_expand(ctx, graph, nullptr);
            for (auto &p : params) {
                p.grad = ggml_graph_get_grad(graph, p.tensor);
                if (!p.grad)
                    p.grad = ggml_graph_get_grad_acc(graph, p.tensor);
                if (!p.grad)
                    throw std::runtime_error("missing adapter gradient: " + p.name);
                ggml_set_output(p.grad);
            }
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto n = ggml_graph_node(graph, i);
                if (!ggml_backend_dev_supports_op(ggml_backend_get_device(W.backend), n))
                    throw std::runtime_error(std::string("training backend does not support ") +
                                             ggml_op_name(n->op));
            }
            alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
            if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph))
                throw std::runtime_error("training graph does not fit; shorten windows/train fewer layers");
            fprintf(stderr, "[train] %zu adapters, %d nodes, %.1f MiB graph buffer\n", adapters.size(),
                    ggml_graph_n_nodes(graph), ggml_gallocr_get_buffer_size(alloc, 0) / 1048576.0);
        } catch (...) {
            release();
            throw;
        }
    }
    MgTrainGraph(const MgTrainGraph &) = delete;
    MgTrainGraph &operator=(const MgTrainGraph &) = delete;
    ~MgTrainGraph() { release(); }
    void release() {
        if (alloc)
            ggml_gallocr_free(alloc);
        if (ctx)
            ggml_free(ctx);
        alloc = nullptr;
        ctx = nullptr;
    }

    float compute(const std::vector<int32_t> &codes, const std::vector<float> &projected) {
        if (codes.size() != (size_t)frames * config.n_q || projected.size() != (size_t)tokens * config.dim)
            throw std::runtime_error("invalid training sample shape");
        for (auto v : codes)
            if (v < 0 || v >= config.card)
                throw std::runtime_error("invalid codec token");
        for (auto v : projected)
            if (!std::isfinite(v))
                throw std::runtime_error("nonfinite condition");
        const auto pattern = config.pattern(frames);
        auto seq = build_pattern_sequence(pattern, codes.data(), config.special_token_id);
        std::vector<int32_t> input((size_t)steps * config.n_q);
        std::vector<float> onehot((size_t)config.card * (frames - prefix_frames) * config.n_q, 0);
        std::vector<int32_t> selected_rows;
        for (int q = 0; q < config.n_q; ++q)
            for (int s = 0; s < steps; ++s) {
                input[(size_t)q * steps + s] = seq[(size_t)q * (steps + 1) + s];
                int t = pattern.timestep_at(q, s + 1);
                if (t >= prefix_frames) {
                    onehot[selected_rows.size() * config.card + codes[(size_t)q * frames + t]] = 1;
                    selected_rows.push_back(q * steps + s);
                }
            }
        auto upload = [](ggml_tensor *t, const std::vector<float> &v) {
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
        };
        ggml_backend_tensor_set(ids, input.data(), 0, input.size() * sizeof(int32_t));
        ggml_backend_tensor_set(selected, selected_rows.data(), 0, selected_rows.size() * 4);
        upload(target, onehot);
        upload(context, projected);
        upload(positions, sin_positions(0, steps, config));
        std::vector<float> mask((size_t)steps * steps, 0);
        for (int q = 0; q < steps; ++q)
            for (int k = q + 1; k < steps; ++k)
                mask[(size_t)q * steps + k] = -INFINITY;
        upload(causal, mask);
        for (const auto &p : params)
            upload(p.tensor, p.value);
        for (const auto &p : constants)
            upload(p.first, p.second);
        ggml_graph_reset(graph);
        std::string error;
        if (!graph_compute_checked(model.backend, graph, "MusicGen DoRA backward", error))
            throw std::runtime_error(error);
        float result;
        ggml_backend_tensor_get(loss, &result, 0, sizeof(float));
        if (!std::isfinite(result))
            throw std::runtime_error("nonfinite training loss");
        return result;
    }
    void adam_step(int step, float lr, float clip = 1);
    void save(const std::string &path) const;
};
inline void MgTrainGraph::adam_step(int step, float lr, float clip) {
    if (step <= 0 || !std::isfinite(lr) || lr <= 0 || !std::isfinite(clip) || clip <= 0)
        throw std::runtime_error("invalid optimizer settings");
    std::vector<std::vector<float>> gradients;
    double norm_sq = 0;
    for (const auto &p : params) {
        gradients.push_back(train_read_tensor(p.grad));
        for (float g : gradients.back())
            norm_sq += (double)g * g;
    }
    const float scale = (float)std::min(1.0, clip / (std::sqrt(norm_sq) + 1e-12));
    const float bc1 = 1 - std::pow(0.9f, (float)step), bc2 = 1 - std::pow(0.999f, (float)step);
    for (size_t j = 0; j < params.size(); ++j) {
        auto &p = params[j];
        for (size_t i = 0; i < p.value.size(); ++i) {
            const float g = scale * gradients[j][i];
            p.m[i] = 0.9f * p.m[i] + 0.1f * g;
            p.v[i] = 0.999f * p.v[i] + 0.001f * g * g;
            p.value[i] -= lr * (p.m[i] / bc1) / (std::sqrt(p.v[i] / bc2) + 1e-8f);
            if (!std::isfinite(p.value[i]))
                throw std::runtime_error("nonfinite optimizer update");
        }
    }
}

inline void MgTrainGraph::save(const std::string &path) const {
    if (std::filesystem::exists(path))
        throw std::runtime_error("refusing to overwrite adapter: " + path);
    auto g = gguf_init_empty();
    auto descriptors = ggml_init({ggml_tensor_overhead() * (params.size() + 1), nullptr, true});
    if (!descriptors) {
        gguf_free(g);
        throw std::runtime_error("adapter descriptor allocation failed");
    }
    gguf_set_val_str(g, "general.architecture", "ac-musicgen-lora");
    gguf_set_val_u32(g, "ac.lora.version", 1);
    gguf_set_val_u32(g, "ac.lora.rank", (uint32_t)rank);
    gguf_set_val_f32(g, "ac.lora.alpha", alpha);
    gguf_set_val_str(g, "ac.lora.type", adapter_type.c_str());
    gguf_set_val_str(g, "ac.lora.training_activation", "erf-approx-as-7.1.26");
    gguf_set_val_str(g, "ac.lora.training_objective",
                     prefix_frames == 0 ? "generation-cross-entropy" : "continuation-cross-entropy");
    gguf_set_val_u32(g, "ac.lora.training_frames", frames);
    gguf_set_val_u32(g, "ac.lora.prefix_frames", prefix_frames);
    if (model.gguf && gguf_find_key(model.gguf, "general.name") >= 0)
        gguf_set_val_str(g, "ac.lora.base_name", model.string("general.name").c_str());
    for (const auto &p : params) {
        // Use host-only descriptors; the live graph tensors have device buffers.
        auto t = ggml_new_tensor_2d(descriptors, GGML_TYPE_F32, p.tensor->ne[0], p.tensor->ne[1]);
        ggml_set_name(t, p.name.c_str());
        t->data = const_cast<float *>(p.value.data());
        gguf_add_tensor(g, t);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(descriptors);
    if (!ok)
        throw std::runtime_error("failed to save adapter: " + path);
}
} // namespace ac
