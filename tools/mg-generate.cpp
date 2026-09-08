// mg-generate -- run MusicGen's language model and write the codes it produces.
//
// This is gary's transform: audio and a description in, 30 s of audio out. Given a codec it
// runs the whole thing -- encode the prompt, decode 1500 timesteps with the language model,
// synthesize -- and without one it stops at the codes, which is what the parity check
// compares.
//
// Use `--greedy` for that comparison. Sampling depends on the RNG stream and ours is not
// torch's, so a seed does not carry across; greedy decoding is the only mode in which two
// implementations can be expected to agree exactly.
#include "ac/codec.h"
#include "gguf_model.h"
#include "mg/encodec.h"
#include "mg/pipeline.h"
#include "wav.h"

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
        "usage: mg-generate --lm <lm.gguf> --t5 <t5.gguf> [--codec <encodec.gguf>] [options]\n"
        "\n"
        "  --prompt <text>          text conditioning (default: empty)\n"
        "  --codec <encodec.gguf>   EnCodec 32 kHz; needed to read or write audio\n"
        "  --input <in.wav>         audio to continue from (needs --codec)\n"
        "  --prompt-duration <s>    how much of --input to use as the prompt (default 6)\n"
        "  --from-start             take the head of --input rather than the tail; gary's\n"
        "                           process_audio does this, continue_music does not\n"
        "  --out <out.wav>          write the generated audio (needs --codec)\n"
        "  --prompt-codes <i32>     EnCodec codes to continue from, instead of --input\n"
        "  --duration <s>           output length in seconds (default 30)\n"
        "  --frame-rate <hz>        codec frame rate (default 50, EnCodec 32 kHz)\n"
        "  --greedy                 argmax decoding -- use this to compare against torch\n"
        "  --top-k <n>              top-k sampling (default 250)\n"
        "  --temperature <f>        sampling temperature (default 1.0)\n"
        "  --cfg-coef <f>           guidance (default: the model's; 0 disables it)\n"
        "  --seed <n>               sampling seed (default 1234; not torch's stream)\n"
        "  --device <name>          backend override; also AC_DEVICE / AC_GPU / AC_THREADS\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string lm_path, t5_path, codec_path, in_path, out_path;
    std::string prompt_codes_path, out_codes, out_logits, device, prompt;
    double duration = 30.0, prompt_duration = 6.0;
    bool from_start = false;
    int frame_rate = 50, top_k = 250;
    float temperature = 1.0f, cfg_coef = -1.0f;
    uint64_t seed = 1234;
    bool greedy = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "--lm") lm_path = next("--lm");
        else if (a == "--t5") t5_path = next("--t5");
        else if (a == "--prompt") prompt = next("--prompt");
        else if (a == "--codec") codec_path = next("--codec");
        else if (a == "--input") in_path = next("--input");
        else if (a == "--out") out_path = next("--out");
        else if (a == "--prompt-duration") prompt_duration = atof(next("--prompt-duration").c_str());
        else if (a == "--from-start") from_start = true;
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
        else if (a == "--device") device = next("--device");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }
    if (lm_path.empty()) { fprintf(stderr, "error: --lm is required\n"); usage(); return 2; }
    if (t5_path.empty()) { fprintf(stderr, "error: --t5 is required\n"); usage(); return 2; }

    if (codec_path.empty() && (!in_path.empty() || !out_path.empty())) {
        fprintf(stderr, "error: --input and --out need --codec\n");
        usage();
        return 2;
    }
    if (!in_path.empty() && !prompt_codes_path.empty()) {
        fprintf(stderr, "error: pass either --input or --prompt-codes, not both\n");
        return 2;
    }

    try {
        double t0 = now_s();
        const ac::TextCondition cond = ac::encode_prompt(t5_path, prompt, device);
        fprintf(stderr, "[ac] t5: %d token(s) in %.3fs -- \"%s\"\n",
                cond.tokens, now_s() - t0, prompt.c_str());

        // --- encode the prompt, if there is audio to continue ---------------------------
        //
        // The codec is loaded, used and released before the LM is, then loaded again to
        // synthesize. Both fit alongside each other easily -- the codec is 58M parameters
        // -- but staging keeps the peak where the LM's KV cache is, which is the number
        // that actually decides whether a card is big enough.
        std::vector<int32_t> prompt_codes;
        int prompt_len = 0, sample_rate = 0;
        if (!in_path.empty()) {
            ac::GgufModel codec = ac::load_gguf(
                codec_path.c_str(), ac::make_backend(0, device.empty() ? nullptr : device.c_str()));
            const ac::SeanetConfig cc = ac::SeanetConfig::from(codec);
            if (!cc.rvq()) throw std::runtime_error("--codec must be an RVQ codec");
            sample_rate = cc.sample_rate;
            frame_rate = cc.frame_rate;

            int n_samples = 0, n_ch = 0, rate = 0;
            std::vector<float> audio = ac::read_wav_planar(in_path, n_samples, n_ch, rate);
            // gary's continue_music takes the *last* few seconds; process_audio takes the
            // first. Both then feed exactly that as the prompt.
            ac::conform_audio(audio, n_samples, n_ch, rate, cc,
                              (int)(prompt_duration * cc.frame_rate), !from_start);
            prompt_len = n_samples / cc.hop_length;
            t0 = now_s();
            const std::vector<float> latent =
                ac::codec_encode(codec, cc, audio, n_samples, n_ch);
            const ac::RvqCodebooks books =
                ac::rvq_load(codec, cc.rvq_n_q, cc.rvq_bins, cc.latent_dim);
            prompt_codes = ac::rvq_encode(books, latent, prompt_len);
            fprintf(stderr, "[ac] encode: %.2fs of audio -> %d frames in %.3fs\n",
                    (double)n_samples / rate, prompt_len, now_s() - t0);
        }

        ac::GgufModel lm = ac::load_gguf(
            lm_path.c_str(), ac::make_backend(0, device.empty() ? nullptr : device.c_str()));
        const ac::LmConfig c = ac::LmConfig::from(lm);
        if (c.cond_dim != (int)(cond.hidden.size() / cond.tokens))
            throw std::runtime_error("text encoder width does not match the LM's cond_dim");

        if (!prompt_codes_path.empty()) {
            prompt_codes = read_i32(prompt_codes_path);
            if (prompt_codes.size() % c.n_q)
                throw std::runtime_error("prompt codes are not a whole number of timesteps");
            prompt_len = (int)(prompt_codes.size() / c.n_q);
        }
        for (int32_t v : prompt_codes)
            if (v < 0 || v >= c.card)
                throw std::runtime_error("a prompt code is outside [0, card)");

        ac::GenerateParams params;
        params.max_gen_len = (int)(duration * frame_rate);
        params.use_sampling = !greedy;
        params.temperature = temperature;
        params.top_k = top_k;
        params.cfg_coef = cfg_coef >= 0.0f ? cfg_coef : c.cfg_coef;
        params.seed = seed;

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

        // --- synthesize ------------------------------------------------------------------
        //
        // The codes include the prompt: `generate_continuation` returns it, and gary splices
        // the result onto the original track at the seam rather than trimming it here.
        if (!out_path.empty()) {
            ac::GgufModel codec = ac::load_gguf(
                codec_path.c_str(), ac::make_backend(0, device.empty() ? nullptr : device.c_str()));
            const ac::SeanetConfig cc = ac::SeanetConfig::from(codec);
            if (cc.rvq_n_q != c.n_q)
                throw std::runtime_error("the codec and the LM disagree about the codebook count");
            const ac::RvqCodebooks books =
                ac::rvq_load(codec, cc.rvq_n_q, cc.rvq_bins, cc.latent_dim);
            t0 = now_s();
            const std::vector<float> latent =
                ac::rvq_decode(books, codes, params.max_gen_len);
            int64_t out_samples = 0;
            const std::vector<float> audio =
                ac::codec_decode(codec, cc, latent, params.max_gen_len, out_samples);
            fprintf(stderr, "[ac] decode: %.3fs -> %lld samples (%.2fs)\n",
                    now_s() - t0, (long long)out_samples,
                    (double)out_samples / cc.sample_rate);
            ac::write_wav_planar(out_path, audio.data(), (int)out_samples, cc.channels,
                                 cc.sample_rate);
            fprintf(stderr, "[ac] wrote %s\n", out_path.c_str());
            (void)sample_rate;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
