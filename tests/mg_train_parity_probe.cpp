// Export fixed-sample gradients and Adam updates for an independent PyTorch oracle.
#include "mg/train.h"
#include "mg/pipeline.h"
#include <fstream>

namespace {
std::vector<float> read(const std::filesystem::path &path, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float)) || f.peek() != EOF)
        throw std::runtime_error("invalid input: " + path.string());
    return v;
}
void write(const std::filesystem::path &path, const std::vector<float> &v) {
    std::ofstream f(path, std::ios::binary);
    if (!f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float)))
        throw std::runtime_error("cannot write: " + path.string());
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 5)
            throw std::runtime_error("usage: mg_train_parity_probe fixture-dir output-dir mode device");
        const std::string mode(argv[3]);
        const std::filesystem::path in(argv[1]), out(argv[2]);
        std::filesystem::create_directories(out);
        auto backend = ac::make_backend(1, argv[4]);
        struct Release {
            ggml_backend_t b;
            ~Release() { ggml_backend_free(b); }
        } release{backend};
        auto model = ac::load_gguf((in / "base.gguf").string().c_str(), backend);
        auto config = ac::LmConfig::from(model);
        auto adapter = ac::load_gguf((in / "adapter.gguf").string().c_str(), backend);
        const auto map = ac::load_mg_lora(model, adapter);
        const auto type =
            gguf_find_key(adapter.gguf, "ac.lora.type") >= 0 ? adapter.string("ac.lora.type") : "lora";
        ac::MgTrainGraph graph(model, config, 7, 3, mode == "noprefix" ? 0 : 3, 2, 2, 0, 17, type);
        if (map.size() != graph.adapters.size())
            throw std::runtime_error("adapter target mismatch");
        if (adapter.f32("ac.lora.alpha") != 2)
            throw std::runtime_error("fixture must use alpha 2");
        for (auto &p : graph.params) {
            auto value = adapter.get(p.name);
            if (!ggml_are_same_shape(value, p.tensor))
                throw std::runtime_error("fixture adapter shape mismatch");
            p.value = ac::train_read_tensor(value);
        }
        std::vector<int32_t> codes(7 * config.n_q);
        std::ifstream codes_file(in / "codes.i32", std::ios::binary);
        if (!codes_file.read(reinterpret_cast<char *>(codes.data()), codes.size() * 4))
            throw std::runtime_error("invalid codes");
        const auto hidden = read(in / "context.f32", 3 * config.dim);
        std::ofstream losses(out / "losses.txt");
        losses.precision(10);
        for (int step = 0; step < 3; ++step) {
            losses << graph.compute(codes, hidden) << '\n';
            for (const auto &p : graph.params)
                write(out / ("step" + std::to_string(step) + "-" + p.name + ".grad.f32"),
                      ac::train_read_tensor(p.grad));
            graph.adam_step(step + 1, 0.0001f);
            graph.save((out / ("step" + std::to_string(step + 1) + ".gguf")).string());
        }
        // The cached serving path must use the trained projections during prefill
        // and decoding, with exactly the same teacher-forced causal logits.
        graph.compute(codes, hidden);
        const auto expected = ac::train_read_tensor(graph.logits);
        auto saved = ac::load_gguf((out / "step3.gguf").string().c_str(), backend);
        auto active = ac::load_mg_lora(model, saved);
        ac::TextCondition cond;
        cond.tokens = 3;
        cond.hidden = hidden;
        cond.empty = false;
        ac::GenerateParams gp;
        gp.max_gen_len = 7;
        gp.use_sampling = false;
        gp.cfg_coef = 1;
        std::vector<int32_t> prompt(3 * config.n_q);
        for (int q = 0; q < config.n_q; ++q)
            for (int t = 0; t < 3; ++t)
                prompt[q * 3 + t] = codes[q * 7 + t];
        ac::GenerateReport report;
        auto generated = ac::mg_generate(model, config, cond, prompt, 3, gp, {}, report, &active);
        for (int q = 0; q < config.n_q; ++q)
            for (int v = 0; v < config.card; ++v)
                if (std::fabs(report.first_logits[q * config.card + v] -
                              expected[((size_t)q * graph.steps + 3) * config.card + v]) >
                    (std::string(argv[4]) == "cpu" ? 3e-5f : 2e-3f))
                    throw std::runtime_error(
                        "cached adapter logits disagree: q=" + std::to_string(q) + " v=" + std::to_string(v) +
                        " infer=" + std::to_string(report.first_logits[q * config.card + v]) + " train=" +
                        std::to_string(expected[((size_t)q * graph.steps + 3) * config.card + v]));
        for (int q = 0; q < config.n_q; ++q)
            for (int t = 0; t < 3; ++t)
                if (generated[q * 7 + t] != prompt[q * 3 + t])
                    throw std::runtime_error("continuation modified prefix codes");
        auto zero = ac::load_mg_lora(model, saved, 0);
        ac::GenerateReport br, zr;
        auto base_codes = ac::mg_generate(model, config, cond, prompt, 3, gp, {}, br);
        auto zero_codes = ac::mg_generate(model, config, cond, prompt, 3, gp, {}, zr, &zero);
        if (base_codes != zero_codes || br.first_logits != zr.first_logits)
            throw std::runtime_error("zero-strength serving changed base output");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
