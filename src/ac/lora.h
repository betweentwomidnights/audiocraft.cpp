#pragma once

#include "gguf_model.h"
#include <cmath>
#include <map>

namespace ac {

// Frozen row norms shared by training and inference. DoRA currently requires
// F16/F32 base projections, matching the experimental training path.
inline std::vector<float> lora_row_norm_sq(ggml_tensor* w) {
    if (ggml_n_dims(w) != 2 || (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16))
        throw gguf_error("DoRA rows requires F16/F32 matrix weights");
    std::vector<float> values((size_t)ggml_nelements(w));
    if (w->type == GGML_TYPE_F32) ggml_backend_tensor_get(w, values.data(), 0, values.size()*4);
    else {
        std::vector<ggml_fp16_t> half(values.size());
        ggml_backend_tensor_get(w, half.data(), 0, half.size()*sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(half.data(), values.data(), (int64_t)values.size());
    }
    std::vector<float> result((size_t)w->ne[1]);
    for (int64_t o = 0; o < w->ne[1]; ++o) {
        double sum = w->ne[0]*1e-12;
        for (int64_t i = 0; i < w->ne[0]; ++i) {
            const float v = values[(size_t)o*w->ne[0]+i];
            if (!std::isfinite(v)) throw gguf_error("nonfinite DoRA base weight");
            sum += (double)v*v;
        }
        result[(size_t)o] = (float)sum;
    }
    return result;
}
struct LoraConstants {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~LoraConstants() { if (buffer) ggml_backend_buffer_free(buffer); if (ctx) ggml_free(ctx); }
};

struct LoraProjection {
    ggml_tensor* A = nullptr; // [in, rank]
    ggml_tensor* B = nullptr; // [rank, out]
    float scale = 1.0f;
    ggml_tensor* magnitude = nullptr;
    ggml_tensor* base_norm_sq = nullptr;
    float strength = 1.0f; // DoRA blends the complete weight change, including magnitude.
    std::shared_ptr<LoraConstants> constants;
};
using LoraMap = std::map<std::string, LoraProjection>;

inline ggml_tensor* lora_linear(ggml_context* ctx, const GgufModel& W,
                                const std::string& name, ggml_tensor* x,
                                const LoraMap* adapters = nullptr) {
    ggml_tensor* y = ggml_mul_mat(ctx, W.get(name), x);
    if (adapters) {
        auto it = adapters->find(name);
        if (it != adapters->end()) {
            const auto& p = it->second;
            if ((p.magnitude && p.strength == 0) || (!p.magnitude && p.scale == 0)) return y;
            auto base_y = y;
            y = ggml_add(ctx, y, ggml_scale(ctx,
                ggml_mul_mat(ctx, p.B, ggml_mul_mat(ctx, p.A, x)), p.scale));
            if (p.magnitude) {
                auto w = W.get(name);
                const int64_t out = w->ne[1];
                // ||W+sBA||² = ||W||² + 2s<W,BA> + s²||BA||².
                // Keep the norm fully differentiable, without a dense adapted matrix.
                auto wta = ggml_mul_mat(ctx, w, p.A);
                auto bt = ggml_cont(ctx, ggml_transpose(ctx, p.B));
                auto t2 = ggml_sum_rows(ctx, ggml_cont(ctx,
                    ggml_transpose(ctx, ggml_mul(ctx, wta, bt))));
                auto ata = ggml_mul_mat(ctx, p.A, p.A);
                auto t3 = ggml_sum_rows(ctx, ggml_mul(ctx, p.B, ggml_mul_mat(ctx, ata, p.B)));
                auto nsq = ggml_add(ctx, ggml_add(ctx, p.base_norm_sq,
                    ggml_scale(ctx, ggml_reshape_1d(ctx, t2, out), 2*p.scale)),
                    ggml_scale(ctx, ggml_reshape_1d(ctx, t3, out), p.scale*p.scale));
                y = ggml_mul(ctx, y, ggml_div(ctx, p.magnitude, ggml_sqrt(ctx, nsq)));
                if (p.strength != 1) y = ggml_add(ctx, base_y,
                    ggml_scale(ctx, ggml_sub(ctx, y, base_y), p.strength));
            }
        }
    }
    return y;
}

// Training-only LayerNorm: centering followed by RMSNorm has an autodiff rule.
inline ggml_tensor* train_layer_norm(ggml_context* ctx, const GgufModel& W,
                                     const std::string& p, ggml_tensor* x, float eps) {
    // MEAN backward assumes a scalar; SUM_ROWS + SCALE supports multiple rows.
    auto mean = ggml_scale(ctx, ggml_sum_rows(ctx, x), 1.0f / (float)x->ne[0]);
    auto centered = ggml_sub(ctx, x, ggml_repeat(ctx, mean, x));
    return ggml_add(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, centered, eps),
                                 W.get(p + ".weight")), W.get(p + ".bias"));
}

// Abramowitz-Stegun 7.1.26 erf approximation (absolute erf error <= 1.5e-7).
// Built from differentiable ggml ops because GELU_ERF has no backward rule.
inline ggml_tensor* train_gelu(ggml_context* ctx, ggml_tensor* x) {
    auto z = ggml_scale(ctx, x, 0.7071067811865475f);
    auto a = ggml_abs(ctx, z);
    auto t = ggml_div(ctx, ggml_scale_bias(ctx, a, 0.0f, 1.0f), ggml_scale_bias(ctx, a, 0.3275911f, 1.0f));
    auto poly = ggml_scale_bias(ctx, t, 1.061405429f, -1.453152027f);
    poly = ggml_scale_bias(ctx, ggml_mul(ctx, poly, t), 1.0f, 1.421413741f);
    poly = ggml_scale_bias(ctx, ggml_mul(ctx, poly, t), 1.0f, -0.284496736f);
    poly = ggml_scale_bias(ctx, ggml_mul(ctx, poly, t), 1.0f, 0.254829592f);
    poly = ggml_mul(ctx, poly, t);
    auto e = ggml_exp(ctx, ggml_neg(ctx, ggml_sqr(ctx, z)));
    auto erf_abs = ggml_scale_bias(ctx, ggml_mul(ctx, poly, e), -1.0f, 1.0f);
    auto erf = ggml_mul(ctx, ggml_sgn(ctx, z), erf_abs);
    return ggml_mul(ctx, x, ggml_scale_bias(ctx, erf, 0.5f, 0.5f));
}

// Validate an adapter before constructing any graph. The GGUF owns its tensors.
inline LoraMap load_lora(const GgufModel& base, const GgufModel& adapter,
                           float strength = 1.0f) {
    const auto arch = adapter.string("general.architecture");
    if ((arch != "ac-melodyflow-lora" && arch != "ac-musicgen-lora" && arch != "ac-lora") ||
        adapter.u32("ac.lora.version") != 1)
        throw gguf_error("unsupported adapter");
    const int rank = (int)adapter.u32("ac.lora.rank");
    const float alpha = adapter.f32("ac.lora.alpha");
    const std::string type = gguf_find_key(adapter.gguf, "ac.lora.type") >= 0
        ? adapter.string("ac.lora.type") : "lora";
    if (type != "lora" && type != "dora-rows") throw gguf_error("unsupported adapter type: " + type);
    const bool dora = type == "dora-rows";
    if (rank <= 0 || !std::isfinite(alpha) || alpha <= 0 || !std::isfinite(strength))
        throw gguf_error("invalid adapter rank, alpha or strength");
    LoraMap out;
    for (const auto& entry : adapter.tensors) {
        const auto& name = entry.first;
        const std::string suffix = ".lora_A";
        if (name.size() <= suffix.size() ||
            name.compare(name.size() - suffix.size(), suffix.size(), suffix)) continue;
        const auto stem = name.substr(0, name.size() - suffix.size());
        auto A = entry.second;
        auto B = adapter.get(stem + ".lora_B");
        auto W = base.get(stem);
        if (A->type != GGML_TYPE_F32 || B->type != GGML_TYPE_F32 ||
            ggml_n_dims(W) != 2 || A->ne[0] != W->ne[0] || A->ne[1] != rank ||
            B->ne[0] != rank || B->ne[1] != W->ne[1] ||
            A->ne[2] != 1 || A->ne[3] != 1 || B->ne[2] != 1 || B->ne[3] != 1)
            throw gguf_error("adapter shape mismatch: " + stem);
        LoraProjection p{A, B, (dora ? 1 : strength) * alpha / rank};
        if (dora) {
            p.magnitude = adapter.get(stem + ".magnitude");
            auto m = p.magnitude;
            if (m->type != GGML_TYPE_F32 || m->ne[0] != W->ne[1] || m->ne[1] != 1 || m->ne[2] != 1 || m->ne[3] != 1)
                throw gguf_error("DoRA magnitude shape mismatch: " + stem);
            const auto nsq = lora_row_norm_sq(W);
            p.constants = std::make_shared<LoraConstants>();
            p.constants->ctx = ggml_init({ggml_tensor_overhead()+256, nullptr, true});
            if (!p.constants->ctx) throw gguf_error("DoRA norm descriptor allocation failed");
            p.base_norm_sq = ggml_new_tensor_1d(p.constants->ctx, GGML_TYPE_F32, W->ne[1]);
            p.constants->buffer = ggml_backend_alloc_ctx_tensors(p.constants->ctx, base.backend);
            if (!p.constants->buffer) throw gguf_error("DoRA norm allocation failed");
            ggml_backend_tensor_set(p.base_norm_sq, nsq.data(), 0, nsq.size()*4);
            p.strength = strength;
        }
        out[stem] = std::move(p);
    }
    if (out.empty() || adapter.tensors.size() != (dora ? 3 : 2) * out.size())
        throw gguf_error("empty or incomplete adapter");
    for (const auto& entry : out) {
        // Reading through the backend also rejects corrupt/nonfinite adapter values.
        std::vector<ggml_tensor*> tensors{entry.second.A, entry.second.B};
        if (entry.second.magnitude) tensors.push_back(entry.second.magnitude);
        for (auto tensor : tensors) {
            std::vector<float> data((size_t)ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
            for (float v : data) if (!std::isfinite(v)) throw gguf_error("nonfinite adapter tensor");
        }
    }
    if (gguf_find_key(adapter.gguf, "ac.lora.base_name") >= 0 &&
        adapter.string("ac.lora.base_name") != base.string("general.name"))
        throw gguf_error("adapter base model name does not match");
    return out;
}

inline LoraMap load_mf_lora(const GgufModel& base, const GgufModel& adapter, float strength = 1) {
    if (adapter.string("general.architecture") != "ac-melodyflow-lora")
        throw gguf_error("not a MelodyFlow adapter");
    auto result = load_lora(base, adapter, strength);
    for (const auto& entry : result) {
        bool supported = false;
        for (const char* suffix : {".self.qkv.weight", ".self.out.weight", ".cross.q.weight",
                                  ".cross.kv.weight", ".cross.out.weight", ".ff.0.weight", ".ff.2.weight"}) {
            const std::string end = suffix;
            supported |= entry.first.size() > end.size() && entry.first.rfind("dit.", 0) == 0 &&
                entry.first.compare(entry.first.size()-end.size(), end.size(), end) == 0;
        }
        if (!supported) throw gguf_error("unsupported MelodyFlow adapter target: " + entry.first);
    }
    return result;
}

inline LoraMap load_mg_lora(const GgufModel& base, const GgufModel& adapter, float strength = 1) {
    if (adapter.string("general.architecture") != "ac-musicgen-lora")
        throw gguf_error("not a MusicGen adapter");
    auto result = load_lora(base, adapter, strength);
    for (const auto& entry : result) {
        bool supported = false;
        for (const char* suffix : {".self.qkv.weight", ".self.out.weight", ".cross.q.weight",
                                  ".cross.kv.weight", ".cross.out.weight", ".ff.0.weight", ".ff.2.weight"}) {
            const std::string end = suffix;
            supported |= entry.first.size() > end.size() && entry.first.rfind("lm.", 0) == 0 &&
                entry.first.compare(entry.first.size()-end.size(), end.size(), end) == 0;
        }
        if (!supported) throw gguf_error("unsupported MusicGen adapter target: " + entry.first);
    }
    return result;
}

} // namespace ac
