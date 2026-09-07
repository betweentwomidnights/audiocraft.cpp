// ac-dit -- one MelodyFlow velocity prediction.
//
// The Phase 2 parity driver: given a latent, a flow step t and a text prompt, run T5 and
// the DiT once and dump the predicted velocity. Everything is deterministic, so this is
// directly comparable against tools/dump_mf_dit_refs.py.
//
// Raw f32 dumps are in ggml memory order; the latent and the velocity are both
// [seq, latent_dim] with the channel as the slow axis, matching what ac-vae writes.
#include "ac/t5.h"
#include "ac/tokenizer.h"
#include "gguf_model.h"
#include "mf/dit.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct GraphArena {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ~GraphArena() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }
};

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::vector<float> read_f32(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes < 0 || bytes % (long)sizeof(float))
        throw std::runtime_error(path + " is not a whole number of f32 values");
    std::vector<float> out((size_t)bytes / sizeof(float));
    const size_t got = fread(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    if (got != out.size()) throw std::runtime_error("short read from " + path);
    return out;
}

void write_f32(const std::string& path, const std::vector<float>& data) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open " + path);
    const size_t wrote = fwrite(data.data(), sizeof(float), data.size(), f);
    fclose(f);
    if (wrote != data.size()) throw std::runtime_error("short write to " + path);
}

// Run T5-base over one prompt and return its last hidden state as [dim, tokens] in ggml
// order, plus the token count. Same staging as ac-textenc: the encoder is loaded, used and
// released before the DiT is loaded, so the two never share a GPU at once.
std::vector<float> encode_prompt(const std::string& t5_path, const std::string& prompt,
                                 const std::string& device, int& tokens, int& dim) {
    ac::UnigramTokenizer tokenizer = ac::UnigramTokenizer::load(t5_path.c_str());
    ac::GgufModel t5 = ac::load_gguf(t5_path.c_str(),
                                     ac::make_backend(0, device.empty() ? nullptr : device.c_str()));
    const ac::T5EncoderConfig c = ac::T5EncoderConfig::from(t5);
    const std::vector<int32_t> ids = tokenizer.encode(prompt, (int)t5.u32("ac.t5.max_length"));
    const int seq = (int)ids.size();

    GraphArena arena;
    ggml_init_params ip = {(size_t)256 * 1024 * 1024, nullptr, true};
    arena.ctx = ggml_init(ip);
    if (!arena.ctx) throw std::runtime_error("failed to create the T5 graph context");
    ggml_tensor* ids_t = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_I32, seq);
    ggml_tensor* mask_t = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, seq, seq);
    ggml_tensor* rel_t = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_I32, (int64_t)seq * seq);
    for (ggml_tensor* t : {ids_t, mask_t, rel_t}) ggml_set_input(t);
    ggml_tensor* hidden = ggml_cont(arena.ctx, ac::t5_encode(arena.ctx, t5, ids_t, mask_t, rel_t, c));
    ggml_set_output(hidden);
    ggml_cgraph* graph = ggml_new_graph_custom(arena.ctx, 8192, false);
    ggml_build_forward_expand(graph, hidden);
    arena.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(t5.backend));
    if (!arena.alloc || !ggml_gallocr_alloc_graph(arena.alloc, graph))
        throw std::runtime_error("failed to allocate the T5 graph");

    const std::vector<int32_t> rel =
        ac::t5_relative_position_buckets(seq, c.relative_buckets, c.relative_max_distance);
    const std::vector<float> mask((size_t)seq * seq, 0.0f);   // no padding, so nothing to mask
    ggml_backend_tensor_set(ids_t, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));
    ggml_backend_tensor_set(rel_t, rel.data(), 0, rel.size() * sizeof(int32_t));
    std::string error;
    if (!ac::graph_compute_checked(t5.backend, graph, "T5 encoder", error))
        throw std::runtime_error(error);

    std::vector<float> host((size_t)c.dim * seq);
    ggml_backend_tensor_get(hidden, host.data(), 0, host.size() * sizeof(float));
    tokens = seq;
    dim = c.dim;
    return host;
}

void usage() {
    fprintf(stderr,
        "usage: ac-dit --dit <dit.gguf> --latent <z.f32> --t <flowstep> [options]\n"
        "\n"
        "  --t5 <t5.gguf>       text encoder; with --prompt, conditions on that text\n"
        "  --prompt <text>      the conditioning prompt (default: empty)\n"
        "  --cond <f32>         precomputed T5 hidden states [cond_dim, tokens], instead\n"
        "                       of running the encoder\n"
        "  --uncond             classifier-free guidance's null branch: every text key is\n"
        "                       masked out, leaving only the zero-attention key\n"
        "  --out <path>         raw f32 velocity dump [seq, latent_dim]\n"
        "  --device <name>      backend override; also AC_DEVICE / AC_GPU / AC_THREADS\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string dit_path, t5_path, cond_path, latent_path, out_path, device;
    std::string prompt;
    bool uncond = false;
    bool have_t = false;
    float t = 0.0f;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "--dit") dit_path = next("--dit");
        else if (a == "--t5") t5_path = next("--t5");
        else if (a == "--prompt") prompt = next("--prompt");
        else if (a == "--cond") cond_path = next("--cond");
        else if (a == "--latent") latent_path = next("--latent");
        else if (a == "--out") out_path = next("--out");
        else if (a == "--device") device = next("--device");
        else if (a == "--uncond") uncond = true;
        else if (a == "--t") { t = (float)atof(next("--t").c_str()); have_t = true; }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }
    if (dit_path.empty()) { fprintf(stderr, "error: --dit is required\n"); usage(); return 2; }
    if (latent_path.empty()) { fprintf(stderr, "error: --latent is required\n"); usage(); return 2; }
    if (!have_t) { fprintf(stderr, "error: --t is required\n"); usage(); return 2; }
    if (t5_path.empty() == cond_path.empty()) {
        fprintf(stderr, "error: pass exactly one of --t5 (with --prompt) or --cond\n");
        usage();
        return 2;
    }

    try {
        // Peek at the DiT's geometry before loading the encoder, so a shape mismatch is
        // reported before anything large is allocated.
        std::vector<float> cond;
        int tokens = 0, cond_dim = 0;
        if (!t5_path.empty()) {
            const double t0 = now_s();
            cond = encode_prompt(t5_path, prompt, device, tokens, cond_dim);
            fprintf(stderr, "[ac] t5: %d token(s) in %.3fs\n", tokens, now_s() - t0);
        }

        ac::GgufModel dit = ac::load_gguf(
            dit_path.c_str(), ac::make_backend(0, device.empty() ? nullptr : device.c_str()));
        const ac::DitConfig c = ac::DitConfig::from(dit);

        if (!cond_path.empty()) {
            cond = read_f32(cond_path);
            cond_dim = c.cond_dim;
            if (cond.size() % cond_dim)
                throw std::runtime_error("conditioning is not a whole number of tokens");
            tokens = (int)(cond.size() / cond_dim);
        }
        if (cond_dim != c.cond_dim)
            throw std::runtime_error("text encoder width does not match the DiT's cond_dim");

        const std::vector<float> latent = read_f32(latent_path);
        if (latent.size() % c.latent_dim)
            throw std::runtime_error("latent is not a whole number of frames");
        const int seq = (int)(latent.size() / c.latent_dim);

        fprintf(stderr, "[ac] dit: dim %d, %d layers, %d heads, latent %d x %d frames, "
                        "%d cond token(s), t=%.6f%s\n",
                c.dim, c.layers, c.heads, c.latent_dim, seq, tokens, t,
                uncond ? " (null branch)" : "");

        const double t0 = now_s();
        GraphArena arena;
        ggml_init_params ip = {(size_t)1024 * 1024 * 1024, nullptr, true};
        arena.ctx = ggml_init(ip);
        if (!arena.ctx) throw std::runtime_error("failed to create the DiT graph context");

        ggml_tensor* latent_t = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, seq, c.latent_dim);
        ggml_tensor* cond_t = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, c.cond_dim, tokens);
        ggml_tensor* tfeat_t = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_F32, c.time_dim);
        ggml_tensor* pos_t = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_I32, seq);
        // The mask covers the zero-attention key plus every text token, for each query.
        ggml_tensor* mask_t = uncond
            ? ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, tokens + 1, seq)
            : nullptr;
        for (ggml_tensor* x : {latent_t, cond_t, tfeat_t, pos_t}) ggml_set_input(x);
        if (mask_t) ggml_set_input(mask_t);

        ggml_tensor* velocity = ggml_cont(arena.ctx,
            ac::dit_forward(arena.ctx, dit, latent_t, t, tfeat_t, cond_t, pos_t, mask_t, c));
        ggml_set_output(velocity);
        ggml_cgraph* graph = ggml_new_graph_custom(arena.ctx, ac::dit_graph_nodes(c), false);
        ggml_build_forward_expand(graph, velocity);
        arena.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dit.backend));
        if (!arena.alloc || !ggml_gallocr_alloc_graph(arena.alloc, graph))
            throw std::runtime_error("failed to allocate the DiT graph");
        const double t1 = now_s();

        std::vector<int32_t> positions((size_t)seq);
        for (int i = 0; i < seq; ++i) positions[(size_t)i] = i;
        // 10000 is TimestepEmbedding.timestep_embedding's own hardcoded max_period, which
        // is a different constant from the rotary base even though both happen to be 10000.
        const std::vector<float> tfeat = ac::timestep_features(t, c.time_dim);

        ggml_backend_tensor_set(latent_t, latent.data(), 0, latent.size() * sizeof(float));
        ggml_backend_tensor_set(cond_t, cond.data(), 0, cond.size() * sizeof(float));
        ggml_backend_tensor_set(tfeat_t, tfeat.data(), 0, tfeat.size() * sizeof(float));
        ggml_backend_tensor_set(pos_t, positions.data(), 0, positions.size() * sizeof(int32_t));
        if (mask_t) {
            // log(0) for every real key, 0 for the zero key -- which is what
            // ClassifierFreeGuidanceDropout leaves behind, and the only reason the null
            // branch's softmax is defined at all.
            std::vector<float> mask((size_t)(tokens + 1) * seq, -INFINITY);
            for (int qi = 0; qi < seq; ++qi) mask[(size_t)qi * (tokens + 1)] = 0.0f;
            ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));
        }

        std::string error;
        if (!ac::graph_compute_checked(dit.backend, graph, "MelodyFlow DiT", error))
            throw std::runtime_error(error);
        const double t2 = now_s();

        std::vector<float> host((size_t)seq * c.latent_dim);
        ggml_backend_tensor_get(velocity, host.data(), 0, host.size() * sizeof(float));
        fprintf(stderr, "[ac] dit: build %.3fs, forward %.3fs\n", t1 - t0, t2 - t1);

        if (!out_path.empty()) {
            write_f32(out_path, host);
            fprintf(stderr, "[ac] wrote %s ([%d frames, %d])\n",
                    out_path.c_str(), seq, c.latent_dim);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
