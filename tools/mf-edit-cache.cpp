// Experimental preparation/reconstruction of frozen-base inversion training pairs.
#include "mf/train.h"
#include "mf/pipeline.h"
#include <fstream>
#include <set>
#include <sstream>

namespace {
void raw(const std::filesystem::path& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float)))
        throw std::runtime_error("cannot write " + path.string());
}
void save(const std::filesystem::path& path, const ac::GgufModel& cache,
          const std::vector<float>& clean, const std::vector<float>& pivot,
          const std::vector<float>& hidden, const std::vector<float>& mean,
          const std::vector<float>& std, int frames, int dim, int cond_dim, float start, uint32_t seed) {
    auto g = gguf_init_empty();
    auto ctx = ggml_init({64 * 1024, nullptr, true});
    struct Release { gguf_context* g; ggml_context* c; ~Release() { gguf_free(g); ggml_free(c); } } release{g, ctx};
    if (!ctx) throw std::runtime_error("cache descriptor allocation failed");
    const bool edit = !pivot.empty();
    gguf_set_val_str(g, "general.architecture", edit ? "ac-melodyflow-edit-cache" : "ac-melodyflow-train-cache");
    gguf_set_val_u32(g, "ac.cache.version", 1);
    gguf_set_val_bool(g, "ac.cache.empty_prompt", cache.boolean("ac.cache.empty_prompt"));
    gguf_set_val_str(g, "ac.cache.source_id", cache.string("ac.cache.source_id").c_str());
    if(gguf_find_key(cache.gguf,"ac.cache.prompt")>=0) gguf_set_val_str(g,"ac.cache.prompt",cache.string("ac.cache.prompt").c_str());
    gguf_set_val_str(g, "ac.cache.pairing", "frozen-base-inversion-to-own-fixed-posterior-sample");
    gguf_set_val_u32(g, "ac.cache.pair_seed", seed);
    gguf_set_val_f32(g, "ac.cache.time_min", start);
    if (edit) {
        gguf_set_val_f32(g, "ac.cache.edit_flowstep", start);
        gguf_set_val_str(g, "ac.cache.inversion", "euler25-regularize2-keep1-kl0.2-sway-0.8-null");
    }
    std::vector<float> zeros(clean.size(), 0);
    auto add = [&](const char* name, const std::vector<float>& v, int n0, int n1 = 1) {
        auto t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
        ggml_set_name(t, name); t->data = const_cast<float*>(v.data()); gguf_add_tensor(g, t);
    };
    add("mean", clean, frames, dim); add("stddev", zeros, frames, dim);
    add("hidden", hidden, cond_dim, (int)hidden.size() / cond_dim);
    add("normalization_mean", mean, dim); add("normalization_std", std, dim);
    if (edit) add("edit_pivot", pivot, frames, dim);
    if (std::filesystem::exists(path) || !gguf_write_to_file(g, path.string().c_str(), false))
        throw std::runtime_error("cache write failed or file exists: " + path.string());
}
}
int main(int argc, char** argv) {
    try {
        std::string dit, cache_path, out, lora, device = "cpu", mode;
        int frames = 750, windows_per_source = 1; uint32_t seed = 1234; float start = 0.11f, cfg = 4;
        std::vector<float> starts;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() { if (++i >= argc) throw std::runtime_error("missing " + a); return std::string(argv[i]); };
            if (a == "--dit") dit = next(); else if (a == "--cache") cache_path = next();
            else if (a == "--out") out = next(); else if (a == "--device") device = next();
            else if (a == "--mode") mode = next(); else if (a == "--lora") lora = next();
            else if (a == "--frames") frames = std::stoi(next()); else if (a == "--seed") seed = (uint32_t)std::stoul(next());
            else if (a == "--flowstep") start = std::stof(next()); else if (a == "--cfg-coef") cfg = std::stof(next());
            else if (a == "--windows-per-source") windows_per_source = std::stoi(next());
            else if (a == "--flowsteps") {
                std::stringstream values(next()); std::string value;
                while (std::getline(values, value, ',')) starts.push_back(std::stof(value));
                if (starts.empty()) throw std::runtime_error("empty flowstep list");
            }
            else throw std::runtime_error("unknown " + a);
        }
        if (dit.empty() || cache_path.empty() || out.empty() || (mode != "prepare" && mode != "render") ||
            frames <= 0 || frames > 750 || windows_per_source <= 0 || !std::isfinite(start) || start < 0 || start >= 1 ||
            std::any_of(starts.begin(), starts.end(), [](float t) { return !std::isfinite(t) || t < 0 || t >= 1; }) ||
            !std::isfinite(cfg) || cfg < 0 || (mode == "prepare" && !lora.empty()))
            throw std::runtime_error("usage: mf-edit-cache --mode prepare|render --dit MODEL --cache ROOT|FILE --out ROOT|F32 [--device gpu] [--lora ADAPTER]");
        if (std::filesystem::exists(out) && (mode == "render" || !std::filesystem::is_empty(out)))
            throw std::runtime_error("output exists/nonempty");
        auto backend = ac::make_backend(0, device.c_str());
        struct Release { ggml_backend_t b; ~Release() { ggml_backend_free(b); } } release{backend};
        auto model = ac::load_gguf(dit.c_str(), backend);
        auto c = ac::DitConfig::from(model);
        auto host = ac::make_backend(1, "cpu"); Release release_host{host};
        if (mode == "render") {
            auto cache = ac::load_gguf(cache_path.c_str(), host);
            if (cache.string("general.architecture") != "ac-melodyflow-edit-cache" || cache.u32("ac.cache.version") != 1)
                throw std::runtime_error("render requires an edit cache");
            auto pivot_tensor = cache.get("edit_pivot"), h = cache.get("hidden");
            if (pivot_tensor->ne[0] != frames || pivot_tensor->ne[1] != c.latent_dim || h->ne[0] != c.cond_dim)
                throw std::runtime_error("render cache shape mismatch");
            auto pivot = ac::train_read_tensor(pivot_tensor);
            const auto stats = ac::mf_latent_stats(model, c.latent_dim);
            auto expected_std = stats.std; for (auto& v : expected_std) v += 1e-5f;
            if (ac::train_read_tensor(cache.get("normalization_mean")) != stats.mean ||
                ac::train_read_tensor(cache.get("normalization_std")) != expected_std)
                throw std::runtime_error("render cache normalization mismatch");
            ac::TextCondition cond; cond.hidden = ac::train_read_tensor(h); cond.tokens = (int)h->ne[1];
            cond.empty = cache.boolean("ac.cache.empty_prompt");
            ac::GgufModel adapter; ac::LoraMap map;
            if (!lora.empty()) { adapter = ac::load_gguf(lora.c_str(), backend); map = ac::load_mf_lora(model, adapter); }
            ac::DitRunner runner(model, c, cond, frames, map.empty() ? nullptr : &map);
            ac::FlowParams p; p.source_flowstep = cache.f32("ac.cache.edit_flowstep"); p.cfg_coef = cfg;
            if (!std::isfinite(p.source_flowstep) || p.source_flowstep < 0 || p.source_flowstep >= 1)
                throw std::runtime_error("invalid cached pivot time");
            auto result = ac::flow_solve(p, pivot, c.latent_dim, frames, runner.guided(cfg), {}, {});
            raw(out, result); return 0;
        }
        std::filesystem::create_directories(out);
        ac::TextCondition null; null.tokens = 1; null.hidden.assign(c.cond_dim, 0); null.empty = true;
        ac::DitRunner inverse(model, c, null, frames);
        const auto norm_mean = ac::train_read_tensor(model.get("dit.latent_mean"));
        auto norm_std = ac::train_read_tensor(model.get("dit.latent_std")); for (auto& v : norm_std) v += 1e-5f;
        ac::EditParams edit; edit.target_flowstep = start; edit.regularize_iters = 2; edit.keep_last_k_iters = 1;
        if (starts.empty()) starts.push_back(start);
        std::set<std::string> train_ids; int index = 0;
        for (const char* split : {"train", "validation"}) {
            std::vector<std::filesystem::path> paths;
            for (const auto& e : std::filesystem::directory_iterator(std::filesystem::path(cache_path) / split))
                if (e.path().extension() == ".gguf") paths.push_back(e.path());
            std::sort(paths.begin(), paths.end()); std::map<std::string, int> counts;
            for (const auto& path : paths) {
                auto cache = ac::load_gguf(path.string().c_str(), host);
                if (cache.string("general.architecture") != "ac-melodyflow-train-cache" || cache.u32("ac.cache.version") != 1)
                    throw std::runtime_error("unsupported input cache");
                const auto id = cache.string("ac.cache.source_id");
                if (std::string(split) == "validation" && train_ids.count(id)) throw std::runtime_error("source leakage");
                const int window_index = counts[id]++;
                if (window_index >= windows_per_source) continue;
                start = starts[window_index % starts.size()];
                edit.target_flowstep = start;
                const auto p = ac::edit_inversion_params(edit);
                if (std::string(split) == "train") train_ids.insert(id);
                auto mean = ac::train_read_tensor(cache.get("mean")), std = ac::train_read_tensor(cache.get("stddev"));
                if (cache.get("mean")->ne[0] != frames || cache.get("mean")->ne[1] != c.latent_dim || mean.size() != std.size() ||
                    ac::train_read_tensor(cache.get("normalization_mean")) != norm_mean ||
                    ac::train_read_tensor(cache.get("normalization_std")) != norm_std)
                    throw std::runtime_error("input cache normalization/shape mismatch");
                if (std::any_of(std.begin(), std.end(), [](float v) { return v < 0; }) ||
                    cache.get("hidden")->ne[0] != c.cond_dim || cache.get("hidden")->ne[1] <= 0)
                    throw std::runtime_error("invalid posterior/conditioning cache");
                std::mt19937 rng(seed + index); std::normal_distribution<float> gaussian;
                std::vector<float> clean(mean.size());
                for (size_t i = 0; i < clean.size(); ++i) clean[i] = mean[i] + std[i] * gaussian(rng);
                ac::NoiseFn noise = [&](float* v, size_t n) { for (size_t i = 0; i < n; ++i) v[i] = gaussian(rng); };
                auto pivot = ac::flow_solve(p, clean, c.latent_dim, frames, inverse.guided(0), noise, {});
                if (std::any_of(pivot.begin(), pivot.end(), [](float v) { return !std::isfinite(v); }))
                    throw std::runtime_error("nonfinite frozen-base inversion");
                auto hidden = ac::train_read_tensor(cache.get("hidden"));
                for (const char* kind : {"control", "bridge"}) {
                    auto folder = std::filesystem::path(out) / kind / split; std::filesystem::create_directories(folder);
                    save(folder / path.filename(), cache, clean, std::string(kind) == "bridge" ? pivot : std::vector<float>{},
                         hidden, norm_mean, norm_std, frames, c.latent_dim, c.cond_dim, start, seed + index);
                }
                ++index;
                fprintf(stderr, "[edit-cache] %d %s %s seed %u pivot %.3f\n", index, split, path.filename().string().c_str(), seed + index - 1, start);
            }
        }
        if (!index) throw std::runtime_error("no source caches");
        return 0;
    } catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
