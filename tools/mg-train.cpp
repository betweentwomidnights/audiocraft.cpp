#include "mg/train.h"
#include "ac/prompt_pool.h"
#include "ac/train_controls.h"
#include <chrono>
#include <set>

namespace {
struct Sample {
    std::string name, source, caption;
    std::vector<int32_t> codes;
    std::vector<float> context;
    int tokens, frames;
};
std::vector<Sample> read_samples(const std::string &dir, const ac::GgufModel &model, const ac::LmConfig &c,
                                 int frames, int &tokens) {
    std::vector<std::filesystem::path> paths;
    for (const auto &e : std::filesystem::directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".gguf")
            paths.push_back(e.path());
    std::sort(paths.begin(), paths.end());
    std::vector<Sample> out;
    auto backend = ac::make_backend(1, "cpu");
    struct Release {
        ggml_backend_t b;
        ~Release() { ggml_backend_free(b); }
    } release{backend};
    for (const auto &p : paths) {
        auto cache = ac::load_gguf(p.string().c_str(), backend);
        if (cache.string("general.architecture") != "ac-musicgen-train-cache" ||
            cache.u32("ac.cache.version") != 1 ||
            cache.string("ac.cache.base_name") != model.string("general.name"))
            throw std::runtime_error("cache/base mismatch: " + p.string());
        auto codes = cache.get("codes"), context = cache.get("context");
        if (codes->type != GGML_TYPE_I32 || codes->ne[0] < frames || codes->ne[1] != c.n_q ||
            codes->ne[2] != 1 || codes->ne[3] != 1 || context->type != GGML_TYPE_F32 ||
            context->ne[0] != c.dim || context->ne[1] <= 0 || context->ne[1] > 128 || context->ne[2] != 1 ||
            context->ne[3] != 1)
            throw std::runtime_error("invalid training cache dimensions");
        Sample s;
        s.name = p.filename().string();
        s.frames = (int)codes->ne[0];
        s.source = cache.string("ac.cache.source_id");
        if (s.source.empty())
            throw std::runtime_error("cache source_id must not be empty");
        s.caption = cache.string("ac.cache.prompt");
        s.codes.resize((size_t)ggml_nelements(codes));
        ggml_backend_tensor_get(codes, s.codes.data(), 0, s.codes.size() * 4);
        for (auto v : s.codes)
            if (v < 0 || v >= c.card)
                throw std::runtime_error("invalid codec token");
        s.context = ac::train_read_tensor(context);
        s.tokens = (int)context->ne[1];
        tokens = std::max(tokens, s.tokens);
        out.push_back(std::move(s));
    }
    if (out.empty())
        throw std::runtime_error("no cached samples in " + dir);
    return out;
}
} // namespace
int main(int argc, char **argv) {
    try {
        std::string lm_path, cache_path, out_path, validation_path, device = "cpu", type = "dora-rows",
                                                                    evaluate_adapter;
        bool evaluate_only = false;
        int frames = 600, prefix = 0, steps = 500, rank = 32, first = -1, checkpoint = 250, eval_samples = 8;
        float alpha = -1, lr = 1e-4f, dropout = 0.1f;
        ac::TrainControls controls;
        uint32_t seed = 1234;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() {
                if (++i >= argc)
                    throw std::runtime_error(a + " needs a value");
                return std::string(argv[i]);
            };
            if (controls.parse(a, next)) {
            } else if (a == "--lm")
                lm_path = next();
            else if (a == "--cache")
                cache_path = next();
            else if (a == "--out")
                out_path = next();
            else if (a == "--validation")
                validation_path = next();
            else if (a == "--device")
                device = next();
            else if (a == "--adapter-type")
                type = next();
            else if (a == "--evaluate-only")
                evaluate_only = true;
            else if (a == "--evaluate-adapter") {
                evaluate_adapter = next();
                evaluate_only = true;
            } else if (a == "--frames")
                frames = std::stoi(next());
            else if (a == "--prefix-frames")
                prefix = std::stoi(next());
            else if (a == "--steps")
                steps = std::stoi(next());
            else if (a == "--rank")
                rank = std::stoi(next());
            else if (a == "--first-layer")
                first = std::stoi(next());
            else if (a == "--checkpoint-every")
                checkpoint = std::stoi(next());
            else if (a == "--eval-samples")
                eval_samples = std::stoi(next());
            else if (a == "--alpha")
                alpha = std::stof(next());
            else if (a == "--lr")
                lr = std::stof(next());
            else if (a == "--cfg-dropout")
                dropout = std::stof(next());
            else if (a == "--seed")
                seed = (uint32_t)std::stoul(next());
            else if (a == "--help" || a == "-h") {
                fprintf(stderr,
                        "usage: mg-train --lm model.gguf --cache DIR --out adapter.gguf\n"
                        "  --validation DIR --frames 600 --prefix-frames 0 (full-sequence loss; >0 masks a "
                        "prefix) --steps 500 --rank 32\n"
                        "  --alpha RANK --first-layer N (default last 12) --adapter-type dora-rows|lora\n"
                        "  --lr 0.0001 --cfg-dropout 0.1 --seed 1234 --device cpu|gpu\n"
                        "  --checkpoint-every 250 --eval-samples 8 (0=all)\n"
                        "  --evaluate-only or --evaluate-adapter FILE  score without training/writing\n"
                        "Teacher-forced cross-entropy; prefix/special tokens excluded; exact resume via "
                        ".state sidecars.\n");
                fprintf(stderr, "%s", ac::TrainControls::help());
                return 0;
            } else
                throw std::runtime_error("unknown option: " + a);
        }
        controls.validate();
        if (lm_path.empty() || cache_path.empty() || (!evaluate_only && out_path.empty()) || steps <= 0 ||
            checkpoint < 0 || eval_samples < 0 || !std::isfinite(dropout) || dropout < 0 || dropout > 1 ||
            !std::isfinite(lr) || lr <= 0)
            throw std::runtime_error("missing required paths or invalid training options");
        if (alpha == -1)
            alpha = (float)rank;
        if (!evaluate_only &&
            (std::filesystem::exists(out_path) || std::filesystem::exists(ac::prompt_pool_path(out_path))))
            throw std::runtime_error("output already exists");
        auto backend = ac::make_backend(0, device.c_str());
        struct Release {
            ggml_backend_t b;
            ~Release() { ggml_backend_free(b); }
        } release{backend};
        auto model = ac::load_gguf(lm_path.c_str(), backend);
        auto config = ac::LmConfig::from(model);
        if (first == -1)
            first = std::max(0, config.layers - 12);
        int tokens = 0;
        auto samples = read_samples(cache_path, model, config, frames, tokens);
        std::vector<Sample> validation;
        if (!validation_path.empty())
            validation = read_samples(validation_path, model, config, frames, tokens);
        std::set<std::string> sources;
        for (const auto &s : samples)
            sources.insert(s.source);
        for (const auto &s : validation)
            if (sources.count(s.source))
                throw std::runtime_error("training/validation track overlap");
        std::vector<std::string> captions;
        for (const auto &s : samples)
            captions.push_back(s.caption);
        if (std::all_of(captions.begin(), captions.end(), [](const std::string &s) { return s.empty(); }))
            throw std::runtime_error("no training captions for prompt pool");
        ac::MgTrainGraph graph(model, config, frames, tokens, prefix, rank, alpha, first, seed, type,
                               controls.feedforward);
        if (!evaluate_adapter.empty()) {
            auto adapter = ac::load_gguf(evaluate_adapter.c_str(), model.backend);
            const auto map = ac::load_mg_lora(model, adapter);
            if (map.size() != graph.adapters.size() || adapter.f32("ac.lora.alpha") != alpha ||
                adapter.string("ac.lora.type") != type)
                throw std::runtime_error("evaluation adapter configuration mismatch");
            for (auto &p : graph.params) {
                auto t = adapter.get(p.name);
                if (!ggml_are_same_shape(t, p.tensor))
                    throw std::runtime_error("evaluation adapter shape mismatch");
                p.value = ac::train_read_tensor(t);
            }
        }
        auto compute = [&](const Sample &s, bool drop, int crop = -1) {
            auto context = s.context;
            context.resize((size_t)tokens * config.dim, 0);
            if (drop)
                std::fill(context.begin(), context.end(), 0.0f);
            if (crop < 0)
                crop = (s.frames - frames) / 2;
            std::vector<int32_t> codes((size_t)frames * config.n_q);
            for (int q = 0; q < config.n_q; ++q)
                std::copy_n(s.codes.begin() + (size_t)q * s.frames + crop, frames,
                            codes.begin() + (size_t)q * frames);
            return graph.compute(codes, context);
        };
        auto evaluate = [&]() {
            const auto &probes = validation.empty() ? samples : validation;
            size_t n = eval_samples == 0 ? probes.size() : std::min(probes.size(), (size_t)eval_samples);
            double sum = 0;
            for (size_t i = 0; i < n; ++i)
                sum += compute(probes[i], controls.eval_null);
            return sum / n;
        };
        if (evaluate_only) {
            fprintf(stderr, "[eval] %s prefix %d fixed probe %.7f\n",
                    evaluate_adapter.empty() ? "base" : evaluate_adapter.c_str(), prefix, evaluate());
            return 0;
        }
        auto save = [&](const std::string &path) {
            if (std::filesystem::exists(ac::prompt_pool_path(path)))
                throw std::runtime_error("prompt pool already exists");
            graph.save(path);
            ac::write_prompt_pool(ac::prompt_pool_path(path), path, captions, sources.size());
        };
        fprintf(stderr,
                "[train] %zu training / %zu validation windows; %d prefix, %d continuation frames; fixed "
                "probe before %.7f\n",
                samples.size(), validation.size(), prefix, frames - prefix, evaluate());
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uniform(0, 1);
        std::vector<size_t> order(samples.size());
        for (size_t i = 0; i < order.size(); ++i)
            order[i] = i;
        std::string state_signature = controls.signature(lr) + ac::train_file_identity(lm_path) +
                                      ac::train_key(frames, rank, alpha, dropout, seed, prefix, type, first);
        for (const auto &dir : {cache_path, validation_path})
            if (!dir.empty()) {
                std::vector<std::filesystem::path> paths;
                for (const auto &e : std::filesystem::directory_iterator(dir))
                    if (e.path().extension() == ".gguf")
                        paths.push_back(e.path());
                std::sort(paths.begin(), paths.end());
                for (const auto &p : paths)
                    state_signature +=
                        "|" + p.filename().string() + ":" + ac::train_file_identity(p.string());
            }
        int completed = 0;
        size_t seen = 0;
        if (!controls.resume.empty())
            ac::train_state(controls.resume, true, state_signature, graph.params, completed, seen, rng,
                            order);
        if (completed < 0 || completed >= steps || seen != (size_t)completed * controls.accumulation)
            throw std::runtime_error("invalid resume step/exposure or target steps");
        auto save_state = [&](const std::string &path) {
            ac::train_state(path + ".state", false, state_signature, graph.params, completed, seen, rng,
                            order);
        };
        auto start = std::chrono::steady_clock::now();
        for (int step = completed + 1; step <= steps; ++step) {
            ac::TrainAccumulation accumulation;
            double loss = 0;
            for (int micro = 0; micro < controls.accumulation; ++micro) {
                size_t index = seen % order.size();
                if (index == 0)
                    std::shuffle(order.begin(), order.end(), rng);
                const auto &sample = samples[order[index]];
                bool drop = uniform(rng) < dropout;
                int crop = controls.random_crop
                               ? std::uniform_int_distribution<int>(0, sample.frames - frames)(rng)
                               : (sample.frames - frames) / 2;
                loss += compute(sample, drop, crop);
                accumulation.add(graph.params);
                ++seen;
            }
            float applied_lr = controls.rate(lr, step);
            accumulation.update(graph.params, controls, step, applied_lr);
            completed = step;
            fprintf(stderr, "[train] step %d/%d windows %zu lr %.9g loss %.7f\n", step, steps, seen,
                    applied_lr, loss / controls.accumulation);
            if (checkpoint > 0 && step % checkpoint == 0 && step < steps) {
                auto p = std::filesystem::path(out_path);
                auto path =
                    (p.parent_path() / (p.stem().string() + ".step-" + std::to_string(step) + ".gguf"))
                        .string();
                save(path);
                save_state(path);
                fprintf(stderr, "[train] fixed probe step %d %.7f\n", step, evaluate());
            }
        }
        fprintf(stderr, "[train] fixed probe after %.7f; %.2fs elapsed\n", evaluate(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        save(out_path);
        save_state(out_path);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
