#include "mf/train.h"
#include "ac/prompt_pool.h"
#include "ac/train_controls.h"
#include <chrono>
#include <fstream>
#include <set>

namespace {
struct Sample {
    std::string name;
    std::string source_id, caption;
    int tokens = 0, frames = 0;
    bool empty = false;
    std::vector<float> mean, stddev, hidden, pivot;
    float edit_start = 0;
};
std::vector<Sample> read_samples(const std::string &directory, const ac::GgufModel &model,
                                 const ac::DitConfig &c, int frames, int &max_tokens,
                                 const std::string &objective, float t_min) {
    std::vector<std::filesystem::path> paths;
    for (const auto &entry : std::filesystem::directory_iterator(directory))
        if (entry.is_regular_file() && entry.path().extension() == ".gguf")
            paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    std::vector<Sample> out;
    const auto normalization_mean = ac::train_read_tensor(model.get("dit.latent_mean"));
    auto normalization_std = ac::train_read_tensor(model.get("dit.latent_std"));
    for (auto &v : normalization_std)
        v += 1e-5f;
    // Cache files are small and stay on the host. Frozen weights remain on the compute device.
    auto backend = ac::make_backend(1, "cpu");
    struct Release {
        ggml_backend_t b;
        ~Release() { ggml_backend_free(b); }
    } release{backend};
    for (const auto &path : paths) {
        auto cache = ac::load_gguf(path.string().c_str(), backend);
        const bool bridge = objective == "edit-bridge";
        if (cache.string("general.architecture") !=
                (bridge ? "ac-melodyflow-edit-cache" : "ac-melodyflow-train-cache") ||
            cache.u32("ac.cache.version") != 1)
            throw std::runtime_error("unsupported cache: " + path.string());
        if (ac::train_read_tensor(cache.get("normalization_mean")) != normalization_mean ||
            ac::train_read_tensor(cache.get("normalization_std")) != normalization_std)
            throw std::runtime_error("cache normalization does not match the DiT: " + path.string());
        auto mean = cache.get("mean"), stddev = cache.get("stddev"), hidden = cache.get("hidden");
        if (mean->ne[0] < frames || mean->ne[1] != c.latent_dim || mean->ne[2] != 1 || mean->ne[3] != 1 ||
            !ggml_are_same_shape(mean, stddev) || hidden->ne[0] != c.cond_dim || hidden->ne[1] <= 0 ||
            hidden->ne[1] > 128 || hidden->ne[2] != 1 || hidden->ne[3] != 1)
            throw std::runtime_error("cache dimensions disagree with model/--frames: " + path.string());
        Sample s;
        s.name = path.filename().string();
        s.frames = (int)mean->ne[0];
        s.tokens = (int)hidden->ne[1];
        s.source_id = cache.string("ac.cache.source_id");
        if (gguf_find_key(cache.gguf, "ac.cache.prompt") >= 0)
            s.caption = cache.string("ac.cache.prompt");
        s.empty = cache.boolean("ac.cache.empty_prompt");
        s.mean = ac::train_read_tensor(mean);
        s.stddev = ac::train_read_tensor(stddev);
        s.hidden = ac::train_read_tensor(hidden);
        for (float v : s.stddev)
            if (v < 0)
                throw std::runtime_error("cache has invalid posterior stddev");
        s.edit_start = std::max(0.0f, t_min);
        if (gguf_find_key(cache.gguf, "ac.cache.time_min") >= 0) {
            s.edit_start = cache.f32("ac.cache.time_min");
            if (!std::isfinite(s.edit_start) || s.edit_start < 0 || s.edit_start >= 1 ||
                (t_min >= 0 && std::fabs(t_min - s.edit_start) > 1e-6f))
                throw std::runtime_error("cache has invalid or mismatched minimum time");
        }
        if (bridge) {
            if (s.frames != frames)
                throw std::runtime_error("edit-bridge requires fixed cache windows");
            auto pivot = cache.get("edit_pivot");
            if (!ggml_are_same_shape(mean, pivot))
                throw std::runtime_error("edit pivot shape mismatch");
            s.pivot = ac::train_read_tensor(pivot);
            s.edit_start = cache.f32("ac.cache.edit_flowstep");
            if (!std::isfinite(s.edit_start) || s.edit_start < 0 || s.edit_start >= 1 ||
                (t_min >= 0 && std::fabs(t_min - s.edit_start) > 1e-6f) ||
                std::any_of(s.stddev.begin(), s.stddev.end(), [](float v) { return v != 0; }))
                throw std::runtime_error("edit cache needs a fixed target and matching valid pivot time");
        }
        max_tokens = std::max(max_tokens, s.tokens);
        out.push_back(std::move(s));
    }
    if (out.empty())
        throw std::runtime_error("no cached .gguf samples in " + directory);
    return out;
}

void usage() {
    fprintf(stderr, "usage: mf-train --dit model.gguf --cache directory --out adapter.gguf [options]\n"
                    "  --prompt-pool JSON optional training-only pool for legacy caches\n"
                    "  --frames N        cached frames per example (default 50 = 2s)\n"
                    "  --steps N         optimizer steps (default 100)\n"
                    "  --rank N          LoRA rank (default 8)\n"
                    "  --alpha F         LoRA alpha (default rank)\n"
                    "  --adapter-type lora|dora-rows  adaptation formulation (default lora)\n"
                    "  --first-layer N   adapt attention from this layer through the last (default last)\n"
                    "  --lr F            Adam learning rate (default 0.0001)\n"
                    "  --seed N          RNG seed (default 1234)\n"
                    "  --cfg-dropout F   caption dropout probability (default 0.1)\n"
                    "  --objective flow-matching|edit-bridge  target/pairing (default flow-matching)\n"
                    "  --t-min F         minimum physical time (default cached minimum, otherwise 0)\n"
                    "  --t-sampling uniform|logit-normal  training time density (default uniform)\n"
                    "  --t-mean F        logit-normal Gaussian mean (default 0)\n"
                    "  --t-std F         logit-normal Gaussian stddev (default 1)\n"
                    "  --validation DIR  separate cached validation tracks\n"
                    "  --eval-samples N  fixed probe windows (default 4; 0 uses all)\n"
                    "  --eval-draws N    draws per probe window (default 1; >1 stratifies time)\n"
                    "  --checkpoint-every N  save and evaluate every N steps (default 0)\n"
                    "  --device cpu|gpu  compute device (default cpu)\n"
                    "Experimental: exact resume via .state sidecars; no activation checkpointing. Start with "
                    "short windows.\n");
}
} // namespace

int main(int argc, char **argv) {
    try {
        std::string prompt_pool;
        std::string dit_path, cache_path, out_path, validation_path, device = "cpu";
        int frames = 50, steps = 100, rank = 8, first_layer = -1;
        int eval_samples = 4, eval_draws = 1, checkpoint_every = 0;
        ac::TrainControls controls;
        uint32_t seed = 1234;
        float alpha = -1, lr = 1e-4f, dropout = 0.1f;
        std::string t_sampling = "uniform";
        float t_mean = 0, t_std = 1;
        std::string objective = "flow-matching";
        std::string adapter_type = "lora";
        float t_min = -1;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (++i >= argc)
                    throw std::runtime_error(a + " needs a value");
                return argv[i];
            };
            if (controls.parse(a, next)) {
            } else if (a == "--prompt-pool")
                prompt_pool = next();
            else if (a == "--dit")
                dit_path = next();
            else if (a == "--cache")
                cache_path = next();
            else if (a == "--out")
                out_path = next();
            else if (a == "--validation")
                validation_path = next();
            else if (a == "--device")
                device = next();
            else if (a == "--frames")
                frames = std::stoi(next());
            else if (a == "--steps")
                steps = std::stoi(next());
            else if (a == "--rank")
                rank = std::stoi(next());
            else if (a == "--alpha")
                alpha = std::stof(next());
            else if (a == "--adapter-type")
                adapter_type = next();
            else if (a == "--first-layer")
                first_layer = std::stoi(next());
            else if (a == "--lr")
                lr = std::stof(next());
            else if (a == "--cfg-dropout")
                dropout = std::stof(next());
            else if (a == "--t-sampling")
                t_sampling = next();
            else if (a == "--t-mean")
                t_mean = std::stof(next());
            else if (a == "--t-std")
                t_std = std::stof(next());
            else if (a == "--objective")
                objective = next();
            else if (a == "--t-min")
                t_min = std::stof(next());
            else if (a == "--seed")
                seed = (uint32_t)std::stoul(next());
            else if (a == "--eval-samples")
                eval_samples = std::stoi(next());
            else if (a == "--eval-draws")
                eval_draws = std::stoi(next());
            else if (a == "--checkpoint-every")
                checkpoint_every = std::stoi(next());
            else if (a == "--help" || a == "-h") {
                usage();
                fprintf(stderr, "%s", ac::TrainControls::help());
                return 0;
            } else
                throw std::runtime_error("unknown argument: " + a);
        }
        controls.validate();
        if (dit_path.empty() || cache_path.empty() || out_path.empty()) {
            usage();
            return 2;
        }
        if (steps <= 0 || frames <= 0 || frames > 750 || rank <= 0 || eval_samples < 0 || eval_draws <= 0 ||
            checkpoint_every < 0 || !std::isfinite(lr) || lr <= 0 || !std::isfinite(dropout) || dropout < 0 ||
            dropout > 1 || !std::isfinite(t_min) || (t_min != -1 && (t_min < 0 || t_min >= 1)) ||
            (objective != "flow-matching" && objective != "edit-bridge"))
            throw std::runtime_error("invalid training settings");
        if (std::filesystem::exists(out_path))
            throw std::runtime_error("output already exists: " + out_path);
        const ac::MfTimeSampler time_sampler(t_sampling, t_mean, t_std);
        if (alpha == -1)
            alpha = (float)rank;
        auto backend = ac::make_backend(0, device.c_str());
        struct Release {
            ggml_backend_t b;
            ~Release() { ggml_backend_free(b); }
        } release{backend};
        auto model = ac::load_gguf(dit_path.c_str(), backend);
        auto config = ac::DitConfig::from(model);
        if (first_layer == -1)
            first_layer = config.layers - 1;
        int max_tokens = 1;
        auto samples = read_samples(cache_path, model, config, frames, max_tokens, objective, t_min);
        std::vector<Sample> validation;
        if (!validation_path.empty())
            validation = read_samples(validation_path, model, config, frames, max_tokens, objective, t_min);
        std::set<std::string> training_sources;
        for (const auto &s : samples)
            training_sources.insert(s.source_id);
        for (const auto &s : validation)
            if (training_sources.count(s.source_id))
                throw std::runtime_error("source track appears in both training and validation");
        std::vector<std::string> captions;
        for (const auto &s : samples)
            if (!s.caption.empty())
                captions.push_back(s.caption);
        if (!prompt_pool.empty())
            captions = ac::read_prompt_pool(prompt_pool);
        auto save_pool = [&](const std::string &path) {
            if (!captions.empty())
                ac::write_prompt_pool(ac::prompt_pool_path(path), path, captions, training_sources.size());
        };
        ac::MfTrainGraph graph(model, config, frames, max_tokens, rank, alpha, first_layer, seed,
                               adapter_type, controls.feedforward);
        fprintf(stderr, "[train] adapter type: %s\n", adapter_type.c_str());
        graph.training_objective = objective;
        graph.training_time_density = t_sampling;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uniform(0, 1);
        std::vector<size_t> order(samples.size());
        for (size_t i = 0; i < order.size(); ++i)
            order[i] = i;
        auto compute_sample = [&](const Sample &s, std::mt19937 &random, bool drop, float fixed_t = -1.0f,
                                  bool training = false) {
            std::normal_distribution<float> gaussian;
            int crop = training && controls.random_crop
                           ? std::uniform_int_distribution<int>(0, s.frames - frames)(random)
                           : (s.frames - frames) / 2;
            std::vector<float> clean((size_t)frames * config.latent_dim), noise(clean.size());
            for (size_t i = 0; i < clean.size(); ++i) {
                size_t at = (i / frames) * s.frames + crop + i % frames;
                clean[i] = s.mean[at] + s.stddev[at] * gaussian(random);
                noise[i] = gaussian(random);
            }
            auto hidden = s.hidden;
            hidden.resize((size_t)max_tokens * config.cond_dim, 0);
            const float u = fixed_t >= 0 ? fixed_t : training ? time_sampler.sample(random) : uniform(random);
            const float start = s.edit_start;
            const float t = start + (1 - start) * u;
            return objective == "edit-bridge"
                       ? graph.compute_bridge(clean, s.pivot, hidden, t, start, drop || s.empty, s.tokens)
                       : graph.compute(clean, noise, hidden, t, drop || s.empty, s.tokens);
        };
        auto evaluate = [&]() {
            std::mt19937 eval_rng(seed + 1);
            double loss = 0;
            const auto &probes = validation.empty() ? samples : validation;
            const size_t count =
                eval_samples == 0 ? probes.size() : std::min<size_t>((size_t)eval_samples, probes.size());
            for (size_t i = 0; i < count; ++i)
                for (int draw = 0; draw < eval_draws; ++draw)
                    loss += compute_sample(probes[i], eval_rng, controls.eval_null,
                                           eval_draws > 1 ? ((float)draw + 0.5f) / eval_draws : -1.0f);
            return loss / (count * eval_draws);
        };
        fprintf(stderr, "[train] probe: %d windows (0=all), %d draws per window\n", eval_samples, eval_draws);
        fprintf(stderr,
                "[train] training time density: %s (Gaussian mean %.4f, stddev %.4f); validation "
                "uniform/stratified\n",
                t_sampling.c_str(), t_mean, t_std);
        fprintf(stderr, "[train] objective %s; minimum physical time %s\n", objective.c_str(),
                t_min < 0 ? "from cache when present, otherwise 0" : std::to_string(t_min).c_str());
        fprintf(stderr, "[train] %zu training / %zu validation windows; fixed %s probe before %.7f\n",
                samples.size(), validation.size(), validation.empty() ? "training" : "validation",
                evaluate());
        std::string state_signature = controls.signature(lr) + ac::train_file_identity(dit_path) +
                                      ac::train_key(frames, rank, alpha, dropout, seed, objective, t_sampling,
                                                    t_mean, t_std, t_min, adapter_type, first_layer);
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
                loss += compute_sample(sample, rng, drop, -1.f, true);
                accumulation.add(graph.params);
                ++seen;
            }
            float applied_lr = controls.rate(lr, step);
            accumulation.update(graph.params, controls, step, applied_lr);
            completed = step;
            fprintf(stderr, "[train] step %d/%d windows %zu lr %.9g loss %.7f\n", step, steps, seen,
                    applied_lr, loss / controls.accumulation);
            if (checkpoint_every > 0 && step % checkpoint_every == 0 && step < steps) {
                const auto output = std::filesystem::path(out_path);
                const auto checkpoint = output.parent_path() /
                                        (output.stem().string() + ".step-" + std::to_string(step) + ".gguf");
                graph.save(checkpoint.string());
                save_pool(checkpoint.string());
                save_state(checkpoint.string());
                fprintf(stderr, "[train] fixed probe step %d %.7f; saved %s\n", step, evaluate(),
                        checkpoint.string().c_str());
            }
        }
        fprintf(stderr, "[train] fixed probe after %.7f; %.2fs elapsed\n", evaluate(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        graph.save(out_path);
        save_pool(out_path);
        save_state(out_path);
        fprintf(stderr, "[train] saved %s\n", out_path.c_str());
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
