// Exercise the complete tiny DiT backward, including skips, RoPE and fused QKV views.
#include "mf/train.h"
#include "mf/pipeline.h"
#include "test_backend.h"
#include <chrono>
#include <iostream>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
ac::GgufModel tiny_model(ggml_backend_t backend, const ac::DitConfig& c) {
    ac::GgufModel W;
    W.backend = backend; W.owns_backend = false;
    W.gguf = gguf_init_empty();
    gguf_set_val_str(W.gguf, "general.name", "Tiny MelodyFlow test model");
    W.ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    auto add = [&](const std::string& name, int in, int out = 0) {
        auto t = out ? ggml_new_tensor_2d(W.ctx, GGML_TYPE_F32, in, out)
                     : ggml_new_tensor_1d(W.ctx, GGML_TYPE_F32, in);
        ggml_set_name(t, name.c_str()); W.tensors[name] = t;
    };
    add("dit.in_proj.weight", c.latent_dim, c.dim);
    add("dit.out_proj.weight", c.dim, c.latent_dim);
    add("dit.time_embed.0.weight", c.time_dim, c.dim);
    add("dit.time_embed.2.weight", c.dim, c.dim);
    add("dit.cond_proj.weight", c.cond_dim, c.dim);
    add("dit.cond_proj.bias", c.dim);
    add("dit.rope_freq_factors", c.head_dim / 2);
    for (int i = 0; i < c.n_skip; ++i) add("dit.skip." + std::to_string(i) + ".weight", c.dim * 2, c.dim);
    for (int i = 0; i < c.layers; ++i) {
        auto p = "dit." + std::to_string(i) + ".";
        for (auto n : {"norm1", "norm_cross", "norm2"}) {
            add(p + n + ".weight", c.dim); add(p + n + ".bias", c.dim);
        }
        add(p + "self.qkv.weight", c.dim, 3 * c.dim);
        add(p + "self.out.weight", c.dim, c.dim);
        add(p + "cross.q.weight", c.dim, c.dim);
        add(p + "cross.kv.weight", c.dim, 2 * c.dim);
        add(p + "cross.out.weight", c.dim, c.dim);
        add(p + "ff.0.weight", c.dim, c.ff_dim);
        add(p + "ff.2.weight", c.ff_dim, c.dim);
    }
    add("dit.out_norm.weight", c.dim); add("dit.out_norm.bias", c.dim);
    W.buf = ggml_backend_alloc_ctx_tensors(W.ctx, backend);
    require(W.buf != nullptr, "tiny weights allocation failed");
    std::mt19937 rng(44);
    std::normal_distribution<float> normal(0, 0.15f);
    for (const auto& item : W.tensors) {
        std::vector<float> data((size_t)ggml_nelements(item.second));
        for (auto& v : data) v = normal(rng);
        if (item.first.find("norm") != std::string::npos) {
            bool bias = item.first.find("bias") != std::string::npos;
            for (auto& v : data) v = bias ? 0 : 1;
        }
        if (item.first == "dit.rope_freq_factors") for (auto& v : data) v = 1.0f;
        ggml_backend_tensor_set(item.second, data.data(), 0, data.size() * sizeof(float));
    }
    return W;
}

void test(ggml_backend_t backend, const std::string& backend_name, const std::string& type) {
    ac::DitConfig c;
    c.dim = 8; c.layers = 4; c.heads = 2; c.head_dim = 4; c.ff_dim = 16;
    c.latent_dim = 4; c.cond_dim = 6; c.time_dim = 8; c.n_skip = 1;
    auto W = tiny_model(backend, c);
    for (int mode = 0; mode < 2; ++mode) {
        auto ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 3);
        ggml_set_input(input); ggml_set_param(input);
        auto q = ac::to_attention(ctx, ac::to_heads(ctx, ac::qkv_slice(ctx, input, 8, 0), 4, 2));
        auto k = ac::to_attention(ctx, ac::to_heads(ctx, ac::qkv_slice(ctx, input, 8, 1), 4, 2));
        auto v = ac::to_attention(ctx, ac::to_heads(ctx, ac::qkv_slice(ctx, input, 8, 2), 4, 2));
        auto value = mode ? ac::attention(ctx, q, k, v, nullptr, true, true)
                         : ggml_add(ctx, q, ggml_add(ctx, k, v));
        auto loss = ggml_sum(ctx, ggml_sqr(ctx, value));
        ggml_set_loss(loss); ggml_set_output(loss);
        auto graph = ggml_new_graph_custom(ctx, 2048, true);
        ggml_build_forward_expand(graph, loss);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            if (ggml_graph_node(graph, i)->op == GGML_OP_MUL_MAT)
                ggml_mul_mat_set_prec(ggml_graph_node(graph, i), GGML_PREC_F32);
        ggml_build_backward_expand(ctx, graph, nullptr);
        auto grad = ggml_graph_get_grad(graph, input); ggml_set_output(grad);
        auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(alloc, graph), "attention allocation failed");
        std::vector<float> data(72);
        for (size_t i = 0; i < data.size(); ++i) data[i] = 0.3f * std::sin((float)i);
        auto calc = [&]() {
            ggml_backend_tensor_set(input, data.data(), 0, data.size() * 4);
            ggml_graph_reset(graph); require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "attention compute failed");
            float r; ggml_backend_tensor_get(loss, &r, 0, 4); return r;
        };
        calc(); auto g = ac::train_read_tensor(grad);
        for (size_t j : {size_t(0), size_t(8), size_t(16), size_t(47)}) {
            float saved = data[j]; data[j] = saved + 0.002f; float plus = calc();
            data[j] = saved - 0.002f; float minus = calc(); data[j] = saved;
            std::cout << "attention " << mode << " " << j << " analytic=" << g[j] << " finite=" << (plus-minus)/0.004f << '\n';
            require(std::fabs(g[j] - (plus-minus)/0.004f) < 0.004f, "attention gradient mismatch");
        }
        ggml_gallocr_free(alloc); ggml_free(ctx);
    }
    {
        auto ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3);
        ggml_set_input(input); ggml_set_param(input);
        auto value = ac::train_layer_norm(ctx, W, "dit.out_norm", input, c.norm_eps);
        value = ac::train_gelu(ctx, value);
        auto loss = ggml_sum(ctx, ggml_sqr(ctx, value));
        ggml_set_loss(loss); ggml_set_output(loss);
        auto graph = ggml_new_graph_custom(ctx, 2048, true);
        ggml_build_forward_expand(graph, loss); ggml_build_backward_expand(ctx, graph, nullptr);
        auto grad = ggml_graph_get_grad(graph, input); ggml_set_output(grad);
        auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(alloc, graph), "primitive allocation failed");
        std::vector<float> data(24);
        for (size_t i = 0; i < data.size(); ++i) data[i] = std::sin((float)i);
        auto calc = [&]() {
            ggml_backend_tensor_set(input, data.data(), 0, data.size() * 4);
            ggml_graph_reset(graph); require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "primitive compute failed");
            float r; ggml_backend_tensor_get(loss, &r, 0, 4); return r;
        };
        calc(); auto g = ac::train_read_tensor(grad);
        for (size_t k : {size_t(0), size_t(7), size_t(15)}) {
            float saved = data[k]; data[k] = saved + 0.002f; float plus = calc();
            data[k] = saved - 0.002f; float minus = calc(); data[k] = saved;
            std::cout << "primitive " << k << " analytic=" << g[k] << " finite=" << (plus-minus)/0.004f << '\n';
            require(std::fabs(g[k] - (plus-minus)/0.004f) < 0.004f, "primitive gradient mismatch");
        }
        ggml_gallocr_free(alloc); ggml_free(ctx);
    }
    auto original = ac::train_read_tensor(W.get("dit.0.self.qkv.weight"));
    ac::MfTrainGraph train(W, c, 3, 2, 2, 2.0f, 0, 17, type);
    std::vector<float> clean(12), noise(12), hidden(12);
    for (int i = 0; i < 12; ++i) { clean[i] = std::sin((float)i); noise[i] = std::cos((float)i); hidden[i] = 0.1f * i; }
    constexpr float t = 0.43f;
    auto compute = [&]() { return train.compute(clean, noise, hidden, t); };

    // Compare the training forward (with its erf approximation) to the inference graph.
    ac::TextCondition condition; condition.hidden = hidden; condition.tokens = 2;
    ac::DitRunner inference(W, c, condition, 3);
    std::vector<float> x(12), velocity(12);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (1 - t) * noise[i] + t * clean[i];
    inference.predict(x.data(), t, true, velocity.data());
    double reference_loss = 0;
    for (size_t i = 0; i < x.size(); ++i) reference_loss += std::pow(velocity[i] - clean[i] + noise[i], 2);
    reference_loss /= x.size();
    float before = compute();
    // Vulkan's stock inference matmuls may round F32 operands to F16. Training requests
    // full F32; allow that known inference rounding here, keeping the CPU check tight.
    const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    const double inference_tolerance = cpu ? 2e-5 : 1e-3;
    std::cout << backend_name << " training/inference loss delta " << before - reference_loss << '\n';
    require(std::fabs(before - reference_loss) < inference_tolerance, "training forward differs from inference");
    require(std::fabs(compute() - before) < 1e-6, "repeated backward changed loss");
    {
        constexpr float start = 0.11f;
        std::vector<float> pivot = noise, bridge_x(12), bridge_v(12);
        const float u = (t - start) / (1 - start);
        for (size_t i = 0; i < bridge_x.size(); ++i) bridge_x[i] = (1-u)*pivot[i] + u*clean[i];
        inference.predict(bridge_x.data(), t, true, bridge_v.data());
        double expected = 0;
        for (size_t i = 0; i < bridge_v.size(); ++i)
            expected += std::pow(bridge_v[i] - (clean[i]-pivot[i])/(1-start), 2);
        require(std::fabs(train.compute_bridge(clean, pivot, hidden, t, start) - expected/bridge_v.size()) < inference_tolerance,
                "edit bridge physical-time interpolation/velocity scaling mismatch");
        bool bad = false;
        try { train.compute_bridge(clean, pivot, hidden, 0.05f, start); } catch (const std::runtime_error&) { bad = true; }
        require(bad, "edit bridge accepted time before pivot");
    }

    // Zero B must give a nonzero B gradient and a zero A gradient.
    auto ga = ac::train_read_tensor(train.params[0].grad);
    auto gb = ac::train_read_tensor(train.params[1].grad);
    require(std::all_of(ga.begin(), ga.end(), [](float v) { return std::fabs(v) < 1e-8; }), "zero-B A gradient is nonzero");
    require(std::any_of(gb.begin(), gb.end(), [](float v) { return std::fabs(v) > 1e-6; }), "zero-B B gradient is zero");

    // Give B a small value so A also has a measurable derivative.
    for (auto& p : train.params) if (p.name.find(".lora_B") != std::string::npos)
        for (float& v : p.value) v = 0.03f;
    for (size_t j : {size_t(0), size_t(1), size_t(2), train.params.size() - 1}) {
        compute();
        auto grad = ac::train_read_tensor(train.params[j].grad);
        size_t k = (size_t)(std::max_element(grad.begin(), grad.end(),
            [](float a, float b) { return std::fabs(a) < std::fabs(b); }) - grad.begin());
        float saved = train.params[j].value[k];
        constexpr float eps = 0.002f;
        train.params[j].value[k] = saved + eps; float plus = compute();
        train.params[j].value[k] = saved - eps; float minus = compute();
        train.params[j].value[k] = saved;
        float fd = (plus - minus) / (2 * eps);
        std::cout << backend_name << " derivative " << j << " analytic=" << grad[k] << " finite=" << fd << '\n';
        require(std::fabs(grad[k] - fd) < 0.0002f + 0.03f * std::fabs(fd), "finite difference gradient mismatch");
    }
    before = compute();
    for (int i = 1; i <= 30; ++i) { compute(); train.adam_step(i, 0.004f); }
    float after = compute();
    require(after < before * 0.9f, "tiny fixed-sample loss did not decrease");
    require(ac::train_read_tensor(W.get("dit.0.self.qkv.weight")) == original, "base weight changed");
    std::cout << backend_name << " loss " << before << " -> " << after << '\n';

    auto path = std::filesystem::temp_directory_path() /
        ("ac-mf-lora-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    struct Remove { std::filesystem::path p; ~Remove() { std::error_code ec; std::filesystem::remove(p, ec); } } remove{path};
    train.save(path.string());
    auto adapter = ac::load_gguf(path.string().c_str(), backend);
    auto map = ac::load_mf_lora(W, adapter);
    require(map.size() == 20, "adapter roundtrip target count mismatch");
    for (const auto& p : train.params)
        require(ac::train_read_tensor(adapter.get(p.name)) == p.value, "adapter roundtrip values changed");
    gguf_set_val_str(adapter.gguf, "general.architecture", "ac-lora");
    require(ac::load_lora(W, adapter).size() == map.size(), "generic adapter loader failed");
    gguf_set_val_str(adapter.gguf, "general.architecture", "ac-melodyflow-lora");
    // Compare functional row scaling with an independently materialized matrix,
    // including interpolation/extrapolation of the complete DoRA weight update.
    if (type == "dora-rows") for (float strength : {0.0f, 0.5f, 1.0f, 1.4f, -0.5f}) {
        auto m = ac::load_mf_lora(W, adapter, strength);
        const auto& p = m.at("dit.0.self.qkv.weight");
        auto w = W.get("dit.0.self.qkv.weight");
        auto a = ac::train_read_tensor(p.A), b = ac::train_read_tensor(p.B), mag = ac::train_read_tensor(p.magnitude);
        std::vector<float> in(24), expected(72);
        for (size_t i = 0; i < in.size(); ++i) in[i] = std::sin((float)i);
        for (int o = 0; o < 24; ++o) {
            double norm = 8e-12; std::vector<double> row(8);
            for (int i = 0; i < 8; ++i) {
                row[i] = original[o*8+i];
                for (int r = 0; r < 2; ++r) row[i] += p.scale*b[o*2+r]*a[r*8+i];
                norm += row[i]*row[i];
            }
            for (int s = 0; s < 3; ++s) for (int i = 0; i < 8; ++i)
                expected[s*24+o] += (float)((original[o*8+i]+strength*(mag[o]*row[i]/std::sqrt(norm)-original[o*8+i]))*in[s*8+i]);
        }
        auto ctx = ggml_init({1024*1024,nullptr,true});
        auto input = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,8,3); ggml_set_input(input);
        auto output = ac::lora_linear(ctx,W,"dit.0.self.qkv.weight",input,&m); ggml_set_output(output);
        auto graph = ggml_new_graph_custom(ctx,256,false); ggml_build_forward_expand(graph,output);
        for (int i=0;i<ggml_graph_n_nodes(graph);++i) if (ggml_graph_node(graph,i)->op==GGML_OP_MUL_MAT)
            ggml_mul_mat_set_prec(ggml_graph_node(graph,i),GGML_PREC_F32);
        auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(alloc,graph),"DoRA projection allocation failed");
        ggml_backend_tensor_set(input,in.data(),0,in.size()*4);
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"DoRA projection compute failed");
        auto got = ac::train_read_tensor(output);
        for (size_t i=0;i<got.size();++i) require(std::fabs(got[i]-expected[i])<2e-5,"functional DoRA differs from materialized weight");
        ggml_gallocr_free(alloc); ggml_free(ctx);
    }
    auto zero_map = ac::load_mf_lora(W, adapter, 0);
    ac::DitRunner zero(W, c, condition, 3, &zero_map);
    std::vector<float> zero_velocity(12);
    zero.predict(x.data(), t, true, zero_velocity.data());
    for (size_t i = 0; i < velocity.size(); ++i)
        require(std::fabs(velocity[i] - zero_velocity[i]) < 1e-6, "zero-strength adapter changed inference");
    ac::DitRunner adapted(W, c, condition, 3, &map);
    adapted.predict(x.data(), t, true, zero_velocity.data());
    double adapted_loss = 0;
    for (size_t i = 0; i < x.size(); ++i) adapted_loss += std::pow(zero_velocity[i] - clean[i] + noise[i], 2);
    require(std::fabs(adapted_loss / x.size() - after) < inference_tolerance, "saved adapter inference differs from training");
    require(std::isfinite(train.compute(clean, noise, hidden, t, true)), "unconditional backward is nonfinite");
    auto single_condition = condition;
    single_condition.tokens = 1; single_condition.hidden.resize(6);
    ac::DitRunner single(W, c, single_condition, 3, &map);
    single.predict(x.data(), t, true, zero_velocity.data());
    double single_loss = 0;
    for (size_t i = 0; i < x.size(); ++i) single_loss += std::pow(zero_velocity[i] - clean[i] + noise[i], 2);
    require(std::fabs(single_loss / x.size() - train.compute(clean, noise, hidden, t, false, 1)) < inference_tolerance,
            "padded conditioning mask changed prediction");
    // Generation-only LoRA must preserve base inversion while changing the final edit.
    ac::EditParams edit; edit.steps = 3; edit.regularize = false; edit.target_flowstep = 0.13f;
    ac::EditReport base_report, both_report, generation_report, zero_report;
    const auto base_edit = ac::mf_edit_latent(inference, x, 3, 4, edit, {}, {}, base_report);
    ac::mf_edit_latent(adapted, x, 3, 4, edit, {}, {}, both_report);
    const auto generation_edit = ac::mf_edit_latent(adapted, x, 3, 4, edit, {}, {}, generation_report, &inference);
    const auto zero_edit = ac::mf_edit_latent(zero, x, 3, 4, edit, {}, {}, zero_report, &inference);
    require(generation_report.intermediate == base_report.intermediate, "generation-only LoRA changed base inversion");
    require(generation_report.inversion_forwards == 3 && generation_report.generation_forwards == 6,
            "split-runner edit forward accounting is incorrect");
    bool inversion_changed = false, generation_changed = false;
    for (size_t i = 0; i < base_edit.size(); ++i) {
        require(std::fabs(base_edit[i] - zero_edit[i]) < 1e-6, "zero-strength generation-only LoRA changed edit");
        inversion_changed |= std::fabs(both_report.intermediate[i] - base_report.intermediate[i]) > 1e-5;
        generation_changed |= std::fabs(generation_edit[i] - base_edit[i]) > 1e-5;
    }
    require(inversion_changed, "both-pass adapter did not affect inversion in fixture");
    require(generation_changed, "generation-only adapter did not affect final edit in fixture");
    bool rejected = false;
    gguf_set_val_u32(adapter.gguf, "ac.lora.rank", 3);
    try { ac::load_mf_lora(W, adapter); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "adapter rank/shape mismatch was accepted");
    gguf_set_val_u32(adapter.gguf, "ac.lora.rank", 2);
    gguf_set_val_str(adapter.gguf, "ac.lora.base_name", "wrong checkpoint");
    rejected = false;
    try { ac::load_mf_lora(W, adapter); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "adapter base name mismatch was accepted");
    gguf_set_val_str(adapter.gguf, "ac.lora.base_name", "Tiny MelodyFlow test model");
    gguf_set_val_str(adapter.gguf, "ac.lora.type", "dora-cols");
    rejected = false;
    try { ac::load_mf_lora(W, adapter); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "unsupported adapter type accepted");
    gguf_set_val_str(adapter.gguf, "ac.lora.type", type.c_str());
    if (type == "dora-rows") {
        auto magnitude = adapter.get("dit.0.self.qkv.weight.magnitude");
        --magnitude->ne[0]; rejected = false;
        try { ac::load_mf_lora(W, adapter); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "malformed magnitude accepted");
        ++magnitude->ne[0];
    }
}
}

int main() {
    try {
        std::mt19937 uniform_rng(73), logit_rng(73);
        ac::MfTimeSampler uniform, logit("logit-normal");
        int uniform_tails = 0, logit_tails = 0;
        double logit_sum = 0;
        for (int i = 0; i < 20000; ++i) {
            const float u = uniform.sample(uniform_rng), l = logit.sample(logit_rng);
            require(l > 0 && l < 1 && std::isfinite(l), "invalid logit-normal timestep");
            uniform_tails += u < 0.1f || u > 0.9f;
            logit_tails += l < 0.1f || l > 0.9f;
            logit_sum += l;
        }
        require(uniform_tails > 3500 && uniform_tails < 4500, "uniform time density changed");
        require(logit_tails > 350 && logit_tails < 800 && std::fabs(logit_sum / 20000 - 0.5) < 0.01,
                "logit-normal density disagrees with standard Gaussian sigmoid");
        bool invalid = false;
        try { ac::MfTimeSampler bad("logit-normal", 0, 0); } catch (const std::runtime_error&) { invalid = true; }
        require(invalid, "invalid time sampler accepted");
        for (const auto& b : ac_test_backends()) for (const auto& type : {"lora", "dora-rows"}) test(b.backend, b.name, type);
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
