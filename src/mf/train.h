// Experimental, batch-one functional LoRA training on cached codec/T5 outputs.
#pragma once
#include "mf/dit.h"
#include "ac/train_utils.h"
#include "mf/vae.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <filesystem>
#include <random>

namespace ac {

// Keep validation uniform/stratified even when training uses logit-normal time.
struct MfTimeSampler {
    std::string kind;
    float mean, stddev;
    MfTimeSampler(std::string kind_ = "uniform", float mean_ = 0, float stddev_ = 1)
        : kind(std::move(kind_)), mean(mean_), stddev(stddev_) {
        if ((kind != "uniform" && kind != "logit-normal") || !std::isfinite(mean) ||
            !std::isfinite(stddev) || stddev <= 0)
            throw std::runtime_error("invalid timestep sampling settings");
    }
    float sample(std::mt19937& rng) const {
        if (kind == "uniform") return std::uniform_real_distribution<float>(0, 1)(rng);
        const float z = std::normal_distribution<float>(mean, stddev)(rng);
        const float e = std::exp(-std::fabs(z));
        return std::clamp(z >= 0 ? 1 / (1 + e) : e / (1 + e), 1e-6f, 1 - 1e-6f);
    }
};

struct MfTrainGraph {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *x = nullptr, *target = nullptr, *cond = nullptr, *pos = nullptr,
                *rescale = nullptr, *tfeat = nullptr, *mask = nullptr, *loss = nullptr;
    LoraMap adapters;
    std::vector<MfTrainParam> params;
    int frames, tokens, rank;
    float alpha;
    const GgufModel& model;
    DitConfig config;
    std::string training_objective, training_time_density;
    std::string adapter_type;
    std::vector<std::pair<ggml_tensor*, std::vector<float>>> constants;

    MfTrainGraph(const GgufModel& W, const DitConfig& c, int frames_, int tokens_,
                 int rank_, float alpha_, int first_layer, uint32_t seed, std::string type = "lora", bool feedforward = false)
        : frames(frames_), tokens(tokens_), rank(rank_), alpha(alpha_), model(W), config(c), adapter_type(std::move(type)) {
        if (frames <= 0 || tokens <= 0 || rank <= 0 || rank > c.dim ||
            !std::isfinite(alpha) || alpha <= 0 || first_layer < 0 || first_layer >= c.layers ||
            (adapter_type != "lora" && adapter_type != "dora-rows"))
            throw std::runtime_error("invalid training dimensions or adapter configuration");
        try {
            ctx = ggml_init({64 * 1024 * 1024, nullptr, true});
            if (!ctx) throw std::runtime_error("training context allocation failed");
            x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frames, c.latent_dim);
            target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frames, c.latent_dim);
            cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.cond_dim, tokens);
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, frames);
            rescale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.latent_dim, 1);
            tfeat = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.time_dim);
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tokens + 1, frames);
            for (auto t : {x, target, cond, pos, rescale, tfeat, mask}) ggml_set_input(t);
            std::mt19937 rng(seed);
            std::normal_distribution<float> normal(0, 0.02f);
            // Attention projections only. Start at a later layer to bound backward memory.
            for (int layer = first_layer; layer < c.layers; ++layer) {
                for (const auto* suffix : {"self.qkv", "self.out", "cross.q", "cross.kv", "cross.out", "ff.0", "ff.2"}) {
                    if(!feedforward && std::string(suffix).rfind("ff.",0)==0) continue;
                    const std::string name = "dit." + std::to_string(layer) + "." + suffix + ".weight";
                    auto base = W.get(name);
                    if (ggml_n_dims(base) != 2 || (base->type != GGML_TYPE_F32 && base->type != GGML_TYPE_F16))
                        throw gguf_error("experimental training requires F16/F32 projections: " + name);
                    auto A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, base->ne[0], rank);
                    auto B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, base->ne[1]);
                    adapters[name] = {A, B, alpha / rank};
                    std::vector<ggml_tensor*> parameters{A, B};
                    std::vector<float> norm_sq;
                    if (adapter_type == "dora-rows") {
                        norm_sq = lora_row_norm_sq(base);
                        auto magnitude = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, base->ne[1]);
                        auto norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, base->ne[1]);
                        ggml_set_input(norm);
                        adapters[name].magnitude = magnitude;
                        adapters[name].base_norm_sq = norm;
                        constants.push_back({norm, norm_sq});
                        parameters.push_back(magnitude);
                    }
                    for (auto p : parameters) {
                        ggml_set_param(p);
                        ggml_set_input(p);
                        MfTrainParam hp;
                        hp.name = name + (p == A ? ".lora_A" : p == B ? ".lora_B" : ".magnitude");
                        ggml_set_name(p, hp.name.c_str());
                        hp.tensor = p;
                        hp.value.resize((size_t)ggml_nelements(p), 0.0f);
                        if (p == A) for (float& value : hp.value) value = normal(rng);
                        if (p != A && p != B) for (size_t i = 0; i < hp.value.size(); ++i) hp.value[i] = std::sqrt(norm_sq[i]);
                        hp.m.resize(hp.value.size(), 0.0f);
                        hp.v.resize(hp.value.size(), 0.0f);
                        params.push_back(std::move(hp));
                    }
                }
            }
            auto velocity = dit_forward(ctx, W, x, rescale, tfeat, cond, pos, mask, c, &adapters, true);
            loss = ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, velocity, target))),
                              1.0f / (float)(frames * c.latent_dim));
            ggml_set_loss(loss);
            ggml_set_output(loss);
            graph = ggml_new_graph_custom(ctx, 65536, true);
            ggml_build_forward_expand(graph, loss);
            // Training needs smooth F32 arithmetic, including frozen F32 matmuls on Vulkan.
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_MUL_MAT) ggml_mul_mat_set_prec(node, GGML_PREC_F32);
            }
            ggml_build_backward_expand(ctx, graph, nullptr);
            for (auto& p : params) {
                p.grad = ggml_graph_get_grad(graph, p.tensor);
                if (!p.grad) p.grad = ggml_graph_get_grad_acc(graph, p.tensor);
                if (!p.grad) throw std::runtime_error("missing adapter gradient: " + p.name);
                ggml_set_output(p.grad);
            }
            // Reject unsupported backend ops before attempting compute (no fallback scheduler here).
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto node = ggml_graph_node(graph, i);
                if (!ggml_backend_dev_supports_op(ggml_backend_get_device(W.backend), node))
                    throw std::runtime_error(std::string("training backend does not support ") + ggml_op_name(node->op));
            }
            alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
            if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph))
                throw std::runtime_error("training graph does not fit; reduce frames or train fewer layers");
            fprintf(stderr, "[train] %zu adapters, %d nodes, %.1f MiB graph buffer\n",
                    adapters.size(), ggml_graph_n_nodes(graph),
                    ggml_gallocr_get_buffer_size(alloc, 0) / 1048576.0);
        } catch (...) { release(); throw; }
    }
    MfTrainGraph(const MfTrainGraph&) = delete;
    MfTrainGraph& operator=(const MfTrainGraph&) = delete;
    ~MfTrainGraph() { release(); }
    void release() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
        alloc = nullptr; ctx = nullptr;
    }

    float compute(const std::vector<float>& clean, const std::vector<float>& noise,
                  const std::vector<float>& hidden, float t, bool unconditional = false, int valid_tokens = -1) {
        const size_t n = (size_t)frames * config.latent_dim;
        if (clean.size() != n || noise.size() != n || hidden.size() != (size_t)tokens * config.cond_dim ||
            !std::isfinite(t) || t < 0 || t > 1)
            throw std::runtime_error("invalid training sample");
        // MelodyFlow's solver runs noise at t=0 to normalized audio at t=1.
        std::vector<float> input(n), velocity(n);
        for (size_t i = 0; i < n; ++i) {
            input[i] = (1 - t) * noise[i] + t * clean[i];
            velocity[i] = clean[i] - noise[i];
        }
        auto upload = [](ggml_tensor* dst, const std::vector<float>& v) {
            ggml_backend_tensor_set(dst, v.data(), 0, v.size() * sizeof(float));
        };
        upload(x, input); upload(target, velocity); upload(cond, hidden);
        upload(tfeat, timestep_features(t, config.time_dim));
        upload(rescale, std::vector<float>((size_t)config.latent_dim, dit_input_rescale(t)));
        std::vector<int32_t> positions((size_t)frames);
        for (int i = 0; i < frames; ++i) positions[(size_t)i] = i;
        ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
        std::vector<float> mask_values((size_t)(tokens + 1) * frames, unconditional ? -INFINITY : 0.0f);
        if (unconditional) for (int i = 0; i < frames; ++i) mask_values[(size_t)i * (tokens + 1)] = 0;
        if (valid_tokens >= 0 && !unconditional) {
            if (valid_tokens > tokens) throw std::runtime_error("invalid valid token count");
            for (int i = 0; i < frames; ++i) for (int k = valid_tokens + 1; k <= tokens; ++k)
                mask_values[(size_t)i * (tokens + 1) + k] = -INFINITY;
        }
        upload(mask, mask_values);
        // gallocr may reuse parameter/input buffers, so restore every input each step.
        for (const auto& p : params) upload(p.tensor, p.value);
        for (const auto& c : constants) upload(c.first, c.second);
        ggml_graph_reset(graph);
        std::string error;
        if (!graph_compute_checked(model.backend, graph, "MelodyFlow LoRA backward", error))
            throw std::runtime_error(error);
        float result;
        ggml_backend_tensor_get(loss, &result, 0, sizeof(float));
        if (!std::isfinite(result)) throw std::runtime_error("nonfinite training loss");
        return result;
    }

    float compute_bridge(const std::vector<float>& clean, const std::vector<float>& pivot,
                         const std::vector<float>& hidden, float t, float start,
                         bool unconditional = false, int valid_tokens = -1) {
        if (!std::isfinite(start) || start < 0 || start >= 1 || !std::isfinite(t) ||
            t < start || t > 1 || pivot.size() != clean.size())
            throw std::runtime_error("invalid edit bridge");
        // Express the straight pivot->target bridge in physical model time.
        // x_t=(1-u)*pivot+u*clean, u=(t-start)/(1-start).
        // dx_t/dt=(clean-pivot)/(1-start). The equivalent t=0 endpoint
        // lets the existing checked forward/backward graph implement this exactly.
        std::vector<float> endpoint(clean.size());
        for (size_t i = 0; i < clean.size(); ++i)
            endpoint[i] = (pivot[i] - start * clean[i]) / (1 - start);
        return compute(clean, endpoint, hidden, t, unconditional, valid_tokens);
    }

    void adam_step(int step, float lr, float clip = 1.0f) {
        if (step <= 0 || !std::isfinite(lr) || lr <= 0 || !std::isfinite(clip) || clip <= 0)
            throw std::runtime_error("invalid optimizer settings");
        std::vector<std::vector<float>> gradients;
        double norm_sq = 0;
        for (const auto& p : params) {
            gradients.push_back(train_read_tensor(p.grad));
            for (float g : gradients.back()) norm_sq += (double)g * g;
        }
        const float scale = (float)std::min(1.0, clip / (std::sqrt(norm_sq) + 1e-12));
        const float bc1 = 1 - std::pow(0.9f, (float)step), bc2 = 1 - std::pow(0.999f, (float)step);
        for (size_t j = 0; j < params.size(); ++j) {
            auto& p = params[j];
            for (size_t i = 0; i < p.value.size(); ++i) {
                const float g = scale * gradients[j][i];
                p.m[i] = 0.9f * p.m[i] + 0.1f * g;
                p.v[i] = 0.999f * p.v[i] + 0.001f * g * g;
                p.value[i] -= lr * (p.m[i] / bc1) / (std::sqrt(p.v[i] / bc2) + 1e-8f);
                if (!std::isfinite(p.value[i])) throw std::runtime_error("nonfinite optimizer update");
            }
        }
    }

    void save(const std::string& path) const {
        if (std::filesystem::exists(path)) throw std::runtime_error("refusing to overwrite adapter: " + path);
        auto g = gguf_init_empty();
        auto descriptors = ggml_init({ggml_tensor_overhead() * (params.size() + 1), nullptr, true});
        if (!descriptors) { gguf_free(g); throw std::runtime_error("adapter descriptor allocation failed"); }
        gguf_set_val_str(g, "general.architecture", "ac-melodyflow-lora");
        gguf_set_val_u32(g, "ac.lora.version", 1);
        gguf_set_val_u32(g, "ac.lora.rank", (uint32_t)rank);
        gguf_set_val_f32(g, "ac.lora.alpha", alpha);
        gguf_set_val_str(g, "ac.lora.type", adapter_type.c_str());
        gguf_set_val_str(g, "ac.lora.training_activation", "erf-approx-as-7.1.26");
        if (!training_objective.empty()) gguf_set_val_str(g, "ac.lora.training_objective", training_objective.c_str());
        if (!training_time_density.empty()) gguf_set_val_str(g, "ac.lora.training_time_density", training_time_density.c_str());
        if (model.gguf && gguf_find_key(model.gguf, "general.name") >= 0)
            gguf_set_val_str(g, "ac.lora.base_name", model.string("general.name").c_str());
        for (const auto& p : params) {
            // Use host-only descriptors; the live graph tensors have device buffers.
            auto t = ggml_new_tensor_2d(descriptors, GGML_TYPE_F32, p.tensor->ne[0], p.tensor->ne[1]);
            ggml_set_name(t, p.name.c_str());
            t->data = const_cast<float*>(p.value.data());
            gguf_add_tensor(g, t);
        }
        const bool ok = gguf_write_to_file(g, path.c_str(), false);
        gguf_free(g);
        ggml_free(descriptors);
        if (!ok) throw std::runtime_error("failed to save adapter: " + path);
    }
};

} // namespace ac
