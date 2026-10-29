// Export fixed-sample gradients and Adam updates for an independent PyTorch oracle.
#include "mf/train.h"
#include <fstream>

namespace {
std::vector<float> read(const std::filesystem::path& path, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float)) || f.peek() != EOF)
        throw std::runtime_error("invalid input: " + path.string());
    return v;
}
void write(const std::filesystem::path& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float)))
        throw std::runtime_error("cannot write: " + path.string());
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: mf_train_parity_probe fixture-dir output-dir mode device");
        const std::string mode(argv[3]);
        if (mode != "conditional" && mode != "padded" && mode != "null" && mode != "bridge" && mode != "bridge-null")
            throw std::runtime_error("invalid conditioning mode");
        const std::filesystem::path in(argv[1]), out(argv[2]);
        std::filesystem::create_directories(out);
        auto backend = ac::make_backend(1, argv[4]);
        struct Release { ggml_backend_t b; ~Release() { ggml_backend_free(b); } } release{backend};
        auto model = ac::load_gguf((in / "base.gguf").string().c_str(), backend);
        auto config = ac::DitConfig::from(model);
        auto adapter = ac::load_gguf((in / "adapter.gguf").string().c_str(), backend);
        const auto map = ac::load_mf_lora(model, adapter);
        const auto type = gguf_find_key(adapter.gguf, "ac.lora.type") >= 0 ? adapter.string("ac.lora.type") : "lora";
        ac::MfTrainGraph graph(model, config, 7, 3, 2, 2, 0, 17, type);
        if (map.size() != graph.adapters.size()) throw std::runtime_error("adapter target mismatch");
        if (adapter.f32("ac.lora.alpha") != 2) throw std::runtime_error("fixture must use alpha 2");
        for (auto& p : graph.params) {
            auto value = adapter.get(p.name);
            if (!ggml_are_same_shape(value, p.tensor)) throw std::runtime_error("fixture adapter shape mismatch");
            p.value = ac::train_read_tensor(value);
        }
        const auto clean = read(in / "clean.f32", 7 * config.latent_dim);
        const auto noise = read(in / "noise.f32", 7 * config.latent_dim);
        const auto hidden = read(in / "hidden.f32", 3 * config.cond_dim);
        std::ofstream losses(out / "losses.txt");
        losses.precision(10);
        const bool null = mode == "null" || mode == "bridge-null";
        const int tokens = mode == "padded" ? 1 : 3;
        for (int step = 0; step < 3; ++step) {
            losses << (mode == "bridge" || mode == "bridge-null"
                ? graph.compute_bridge(clean, noise, hidden, 0.43f, 0.11f, null, tokens)
                : graph.compute(clean, noise, hidden, 0.43f, null, tokens)) << '\n';
            for (const auto& p : graph.params)
                write(out / ("step" + std::to_string(step) + "-" + p.name + ".grad.f32"), ac::train_read_tensor(p.grad));
            graph.adam_step(step + 1, 0.0001f);
            graph.save((out / ("step" + std::to_string(step + 1) + ".gguf")).string());
        }
        return 0;
    } catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 1; }
}
