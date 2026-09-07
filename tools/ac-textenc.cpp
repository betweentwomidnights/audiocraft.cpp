// ac-textenc -- run the T5-base text encoder and dump its last hidden state.
//
// This is the parity driver for Phase 0: it reproduces what audiocraft's T5Conditioner
// computes before `output_proj`, i.e. `self.t5(**inputs).last_hidden_state`. Compare the
// dump against tools/dump_t5_refs.py with tools/cossim.py.
//
// audiocraft does not pad to a fixed length -- it pads to the longest prompt in the batch
// and masks the rest. Because the encoder is bidirectional and padded keys get -inf, the
// hidden states at valid positions are identical whether or not padding is present, so by
// default we encode each prompt at its own token count. --seq forces a padded run, which
// is what exercises the mask path.
#include "ac/t5.h"
#include "ac/tokenizer.h"
#include "gguf_model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

void usage() {
    fprintf(stderr,
            "usage: ac-textenc --t5 <t5.gguf> --prompt <text> [options]\n"
            "\n"
            "  --prompt <text>   repeatable; each prompt is encoded separately\n"
            "  --out <path>      write the hidden state as raw f32 [dim, seq] (ne0 fastest)\n"
            "  --seq N           pad/truncate to N tokens and use an -inf key mask\n"
            "                    (default: each prompt's own token count, no padding)\n"
            "  --device <name>   backend override; also AC_DEVICE / AC_GPU / AC_THREADS\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string t5_path;
    std::string out_path;
    std::string device;
    std::vector<std::string> prompts;
    int forced_seq = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "--t5") t5_path = next("--t5");
        else if (a == "--prompt") prompts.push_back(next("--prompt"));
        else if (a == "--out") out_path = next("--out");
        else if (a == "--seq") forced_seq = std::atoi(next("--seq").c_str());
        else if (a == "--device") device = next("--device");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }
    if (t5_path.empty()) { fprintf(stderr, "error: --t5 is required\n"); usage(); return 2; }
    if (prompts.empty()) { fprintf(stderr, "error: at least one --prompt is required\n"); usage(); return 2; }
    if (forced_seq < 0) { fprintf(stderr, "error: --seq must be >= 0\n"); return 2; }
    if (prompts.size() > 1 && !out_path.empty() && forced_seq == 0) {
        fprintf(stderr, "error: --out with multiple prompts needs --seq so the dumps share a shape\n");
        return 2;
    }

    try {
        const ac::UnigramTokenizer tokenizer = ac::UnigramTokenizer::load(t5_path.c_str());
        ac::GgufModel t5 = ac::load_gguf(t5_path.c_str(),
                                         ac::make_backend(0, device.empty() ? nullptr : device.c_str()));
        const ac::T5EncoderConfig c = ac::T5EncoderConfig::from(t5);
        const int capacity = (int)t5.u32("ac.t5.max_length");

        for (size_t p = 0; p < prompts.size(); ++p) {
            std::vector<int32_t> ids = tokenizer.encode(prompts[p], forced_seq > 0 ? forced_seq : capacity);
            const int valid = (int)ids.size();
            const int seq = forced_seq > 0 ? forced_seq : valid;
            if (valid > seq) { fprintf(stderr, "error: prompt does not fit in --seq %d\n", seq); return 1; }
            ids.resize((size_t)seq, tokenizer.pad_id);

            GraphArena arena;
            ggml_init_params ip = {(size_t)128 * 1024 * 1024, nullptr, true};
            arena.ctx = ggml_init(ip);
            if (!arena.ctx) throw std::runtime_error("failed to create T5 graph context");
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
                throw std::runtime_error("failed to allocate T5 graph");

            const std::vector<int32_t> rel =
                ac::t5_relative_position_buckets(seq, c.relative_buckets, c.relative_max_distance);
            std::vector<float> mask((size_t)seq * seq, -INFINITY);
            for (int q = 0; q < seq; ++q) std::fill_n(mask.data() + (size_t)q * seq, valid, 0.0f);

            ggml_backend_tensor_set(ids_t, ids.data(), 0, ids.size() * sizeof(int32_t));
            ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));
            // gallocr may recycle input storage between executions; re-upload every time.
            ggml_backend_tensor_set(rel_t, rel.data(), 0, rel.size() * sizeof(int32_t));

            std::string error;
            if (!ac::graph_compute_checked(t5.backend, graph, "T5 encoder", error))
                throw std::runtime_error(error);

            std::vector<float> host((size_t)c.dim * seq);
            ggml_backend_tensor_get(hidden, host.data(), 0, host.size() * sizeof(float));
            // audiocraft zeroes padded positions after output_proj; do it here too so a
            // padded run and an unpadded run of the same prompt dump identical bytes.
            for (int token = valid; token < seq; ++token)
                std::fill_n(host.data() + (size_t)token * c.dim, c.dim, 0.0f);

            printf("prompt %zu: %d token(s), seq %d, hidden [%d, %d]\n",
                   p, valid, seq, c.dim, seq);

            if (!out_path.empty()) {
                const std::string path =
                    prompts.size() == 1 ? out_path : out_path + "." + std::to_string(p);
                FILE* f = fopen(path.c_str(), "wb");
                if (!f) throw std::runtime_error("failed to open " + path);
                const size_t wrote = fwrite(host.data(), sizeof(float), host.size(), f);
                fclose(f);
                if (wrote != host.size()) throw std::runtime_error("short write to " + path);
                printf("  wrote %s (%zu floats)\n", path.c_str(), host.size());
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
