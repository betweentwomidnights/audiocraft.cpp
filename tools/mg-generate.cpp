// mg-generate -- run MusicGen's language model and write the codes it produces.
//
// This is the Phase 4 driver: text and an optional EnCodec prompt in, `n_q x timesteps`
// codes out. Turning those codes back into audio is the codec's job and lands in Phase 5,
// so the output here is a raw i32 dump that `tools/cossim.py`'s reference counterpart can
// be compared against token for token.
//
// Use `--greedy` for that comparison. Sampling depends on the RNG stream and ours is not
// torch's, so a seed does not carry across; greedy decoding is the only mode in which two
// implementations can be expected to agree exactly.
#include "gguf_model.h"
#include "mg/pipeline.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::vector<int32_t> read_i32(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes < 0 || bytes % (long)sizeof(int32_t))
        throw std::runtime_error(path + " is not a whole number of i32 values");
    std::vector<int32_t> out((size_t)bytes / sizeof(int32_t));
    const size_t got = fread(out.data(), sizeof(int32_t), out.size(), f);
    fclose(f);
    if (got != out.size()) throw std::runtime_error("short read from " + path);
    return out;
}

void write_i32(const std::string& path, const std::vector<int32_t>& data) {
    if (path.empty()) return;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open " + path);
    const size_t wrote = fwrite(data.data(), sizeof(int32_t), data.size(), f);
    fclose(f);
    if (wrote != data.size()) throw std::runtime_error("short write to " + path);
    fprintf(stderr, "[ac] wrote %s (%zu codes)\n", path.c_str(), data.size());
}

void usage() {
    fprintf(stderr,
        "usage: mg-generate --lm <lm.gguf> --t5 <t5.gguf> --out-codes <codes.i32> [options]\n"
        "\n"
        "  --prompt <text>          text conditioning (default: empty)\n"
        "  --prompt-codes <i32>     EnCodec codes to continue from, [n_q, prompt_len]\n"
        "  --duration <s>           output length in seconds (default 30)\n"
        "  --frame-rate <hz>        codec frame rate (default 50, EnCodec 32 kHz)\n"
        "  --greedy                 argmax decoding -- use this to compare against torch\n"
        "  --top-k <n>              top-k sampling (default 250)\n"
        "  --temperature <f>        sampling temperature (default 1.0)\n"
        "  --cfg-coef <f>           guidance (default: the model's; 0 disables it)\n"
        "  --seed <n>               sampling seed (default 1234; not torch's stream)\n"
        "  --uncond-cross           give the null branch an explicit all-zero context\n"
        "                           instead of skipping its cross-attention; the two are\n"
        "                           identical and this exists to prove it\n"
        "  --device <name>          backend override; also AC_DEVICE / AC_GPU / AC_THREADS\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string lm_path, t5_path, prompt_codes_path, out_codes, out_logits, device, prompt;
    double duration = 30.0;
    int frame_rate = 50, top_k = 250;
    float temperature = 1.0f, cfg_coef = -1.0f;
    uint64_t seed = 1234;
    bool greedy = false, uncond_cross = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "--lm") lm_path = next("--lm");
        else if (a == "--t5") t5_path = next("--t5");
        else if (a == "--prompt") prompt = next("--prompt");
        else if (a == "--prompt-codes") prompt_codes_path = next("--prompt-codes");
        else if (a == "--out-codes") out_codes = next("--out-codes");
        else if (a == "--out-logits") out_logits = next("--out-logits");
        else if (a == "--duration") duration = atof(next("--duration").c_str());
        else if (a == "--frame-rate") frame_rate = atoi(next("--frame-rate").c_str());
        else if (a == "--greedy") greedy = true;
        else if (a == "--top-k") top_k = atoi(next("--top-k").c_str());
        else if (a == "--temperature") temperature = (float)atof(next("--temperature").c_str());
        else if (a == "--cfg-coef") cfg_coef = (float)atof(next("--cfg-coef").c_str());
        else if (a == "--seed") seed = strtoull(next("--seed").c_str(), nullptr, 10);
        else if (a == "--uncond-cross") uncond_cross = true;
        else if (a == "--device") device = next("--device");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }
    if (lm_path.empty()) { fprintf(stderr, "error: --lm is required\n"); usage(); return 2; }
    if (t5_path.empty()) { fprintf(stderr, "error: --t5 is required\n"); usage(); return 2; }

    try {
        double t0 = now_s();
        const ac::TextCondition cond = ac::encode_prompt(t5_path, prompt, device);
        fprintf(stderr, "[ac] t5: %d token(s) in %.3fs -- \"%s\"\n",
                cond.tokens, now_s() - t0, prompt.c_str());

        ac::GgufModel lm = ac::load_gguf(
            lm_path.c_str(), ac::make_backend(0, device.empty() ? nullptr : device.c_str()));
        const ac::LmConfig c = ac::LmConfig::from(lm);
        if (c.cond_dim != (int)(cond.hidden.size() / cond.tokens))
            throw std::runtime_error("text encoder width does not match the LM's cond_dim");

        std::vector<int32_t> prompt_codes;
        int prompt_len = 0;
        if (!prompt_codes_path.empty()) {
            prompt_codes = read_i32(prompt_codes_path);
            if (prompt_codes.size() % c.n_q)
                throw std::runtime_error("prompt codes are not a whole number of timesteps");
            prompt_len = (int)(prompt_codes.size() / c.n_q);
            for (int32_t v : prompt_codes)
                if (v < 0 || v >= c.card)
                    throw std::runtime_error("a prompt code is outside [0, card)");
        }

        ac::GenerateParams params;
        params.max_gen_len = (int)(duration * frame_rate);
        params.use_sampling = !greedy;
        params.temperature = temperature;
        params.top_k = top_k;
        params.cfg_coef = cfg_coef >= 0.0f ? cfg_coef : c.cfg_coef;
        params.seed = seed;
        params.uncond_cross = uncond_cross;

        fprintf(stderr, "[ac] lm: dim %d, %d layers, %d heads, %d codebooks of %d\n",
                c.dim, c.layers, c.heads, c.n_q, c.card);
        fprintf(stderr, "[ac] generate: %.1fs at %d Hz = %d timesteps, prompt %d, "
                        "%s, cfg %.2f\n",
                duration, frame_rate, params.max_gen_len, prompt_len,
                greedy ? "greedy" : "sampled", params.cfg_coef);

        int last = -1;
        auto progress = [&last](int done, int total) {
            const int pct = total > 0 ? done * 100 / total : 0;
            if (pct == last) return;
            last = pct;
            fprintf(stderr, "\r[ac] %6d / %6d  (%3d%%)", done, total, pct);
            fflush(stderr);
        };

        ac::GenerateReport report;
        const std::vector<int32_t> codes =
            ac::mg_generate(lm, c, cond, prompt_codes, prompt_len, params, progress, report);
        fprintf(stderr, "\r[ac] prefill %d token(s) in %.2fs, %d decode step(s) in %.2fs "
                        "(%.1f steps/s), %d forward(s)\n",
                report.prefill_tokens, report.prefill_seconds, report.decode_steps,
                report.decode_seconds,
                report.decode_seconds > 0 ? report.decode_steps / report.decode_seconds : 0.0,
                report.forwards);

        if (!out_logits.empty()) {
            FILE* f = fopen(out_logits.c_str(), "wb");
            if (!f) throw std::runtime_error("cannot open " + out_logits);
            fwrite(report.first_logits.data(), sizeof(float), report.first_logits.size(), f);
            fclose(f);
            fprintf(stderr, "[ac] wrote %s (%zu floats)\n", out_logits.c_str(),
                    report.first_logits.size());
        }
        write_i32(out_codes, codes);
        fprintf(stderr, "[ac] codes: [%d, %d]\n", c.n_q, params.max_gen_len);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
