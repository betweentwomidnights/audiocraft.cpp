// mf-edit -- MelodyFlow's `edit`, end to end: audio in, edited audio out.
//
// This is terry's whole transform. Given a wav, a target prompt and the editing parameters,
// it runs T5, the SEANet encoder, the two flow solves and the decoder, and writes a wav.
// With no options beyond the models and the prompt it reproduces terry's defaults exactly:
// euler, 25 steps, target_flowstep 0.12, regularize with 2 iterations keeping the last 1,
// lambda_kl 0.2, guidance 4.0.
//
// The models are loaded one at a time -- text encoder, codec, DiT, codec again -- because
// at the full 30 s window each of them wants gigabytes of compute buffer and an 8 GB card
// cannot hold two.
//
// Noise. Two things draw from the RNG, in this order: the VAE posterior sample of the
// encoded input, then two draws per regularized solver step. `--dump-noise` records that
// stream and `--noise` replays one, which is how a run is compared against torch: seeds do
// not carry across frameworks, so the noise is made an input instead of a parameter.
#include "gguf_model.h"
#include "mf/pipeline.h"
#include "mf/vae.h"
#include "rng.h"
#include "wav.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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
    if (path.empty()) return;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open " + path);
    const size_t wrote = fwrite(data.data(), sizeof(float), data.size(), f);
    fclose(f);
    if (wrote != data.size()) throw std::runtime_error("short write to " + path);
    fprintf(stderr, "[ac] wrote %s (%zu floats)\n", path.c_str(), data.size());
}

// Either an RNG or a recording being replayed, optionally logging what it handed out.
struct NoiseSource {
    ac::Rng rng;
    std::vector<float> replay;
    size_t at = 0;
    bool recording = false;
    std::vector<float> record;

    explicit NoiseSource(uint64_t seed) : rng(seed) {}

    void operator()(float* dst, size_t n) {
        if (!replay.empty()) {
            if (at + n > replay.size())
                throw std::runtime_error("the replayed noise stream is too short");
            std::copy(replay.begin() + (long long)at, replay.begin() + (long long)(at + n), dst);
            at += n;
        } else {
            rng.fill_normal(dst, n);
        }
        if (recording) record.insert(record.end(), dst, dst + n);
    }

    std::vector<float> draw(size_t n) {
        std::vector<float> out(n);
        (*this)(out.data(), n);
        return out;
    }
};

void usage() {
    fprintf(stderr,
        "usage: mf-edit --dit <dit.gguf> --codec <codec.gguf> --t5 <t5.gguf>\n"
        "               --input <in.wav> --prompt <text> --out <out.wav> [options]\n"
        "\n"
        "  --solver euler|midpoint  ODE solver (default euler)\n"
        "  --steps <n>              solver steps (default 25 euler, 64 midpoint)\n"
        "  --flowstep <f>           the pivot the inversion runs down to (default 0.12)\n"
        "  --cfg-coef <f>           guidance on the generation pass (default: the model's)\n"
        "  --no-regularize          skip the KL regularization of the inversion\n"
        "  --regularize-iters <n>   inner iterations per inversion step (default 2)\n"
        "  --keep-last-k <n>        of those, how many carry the KL term (default 1)\n"
        "  --lambda-kl <f>          KL regularization weight (default 0.2)\n"
        "  --seconds <n>            crop the input (default: the model's 30 s window)\n"
        "  --seed <n>               noise seed (default 1234)\n"
        "  --noise <f32>            replay a recorded noise stream instead of the RNG\n"
        "  --dump-noise <f32>       record the stream this run consumed\n"
        "  --out-prompt-latent <f32>  the normalized latent that entered the inversion\n"
        "  --out-intermediate <f32>   the latent at the pivot, between the two passes\n"
        "  --out-latent <f32>         the edited latent, still normalized\n"
        "  --device <name>          backend override; also AC_DEVICE / AC_GPU / AC_THREADS\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string dit_path, codec_path, t5_path, in_path, out_path, device, prompt;
    std::string noise_in, noise_out, out_prompt_latent, out_intermediate, out_latent;
    std::string solver = "euler";
    int steps = 0;
    double seconds = 0.0;
    uint64_t seed = 1234;
    float flowstep = 0.12f, cfg_coef = -1.0f, lambda_kl = 0.2f;
    int regularize_iters = 2, keep_last_k = 1;
    bool regularize = true;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "--dit") dit_path = next("--dit");
        else if (a == "--codec") codec_path = next("--codec");
        else if (a == "--t5") t5_path = next("--t5");
        else if (a == "--input") in_path = next("--input");
        else if (a == "--prompt") prompt = next("--prompt");
        else if (a == "--out") out_path = next("--out");
        else if (a == "--solver") solver = next("--solver");
        else if (a == "--steps") steps = atoi(next("--steps").c_str());
        else if (a == "--flowstep") flowstep = (float)atof(next("--flowstep").c_str());
        else if (a == "--cfg-coef") cfg_coef = (float)atof(next("--cfg-coef").c_str());
        else if (a == "--no-regularize") regularize = false;
        else if (a == "--regularize-iters") regularize_iters = atoi(next("--regularize-iters").c_str());
        else if (a == "--keep-last-k") keep_last_k = atoi(next("--keep-last-k").c_str());
        else if (a == "--lambda-kl") lambda_kl = (float)atof(next("--lambda-kl").c_str());
        else if (a == "--seconds") seconds = atof(next("--seconds").c_str());
        else if (a == "--seed") seed = strtoull(next("--seed").c_str(), nullptr, 10);
        else if (a == "--noise") noise_in = next("--noise");
        else if (a == "--dump-noise") noise_out = next("--dump-noise");
        else if (a == "--out-prompt-latent") out_prompt_latent = next("--out-prompt-latent");
        else if (a == "--out-intermediate") out_intermediate = next("--out-intermediate");
        else if (a == "--out-latent") out_latent = next("--out-latent");
        else if (a == "--device") device = next("--device");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }
    struct Required { const std::string* value; const char* name; };
    for (const Required& r : {Required{&dit_path, "--dit"}, Required{&codec_path, "--codec"},
                              Required{&t5_path, "--t5"}, Required{&in_path, "--input"}}) {
        if (!r.value->empty()) continue;
        fprintf(stderr, "error: %s is required\n", r.name);
        usage();
        return 2;
    }

    try {
        ac::EditParams params;
        params.solver = ac::parse_flow_solver(solver);
        // terry overrides both of these together when the request asks for midpoint.
        params.steps = steps > 0 ? steps
                                 : (params.solver == ac::FlowSolverKind::Midpoint ? 64 : 25);
        params.target_flowstep = flowstep;
        params.regularize = regularize && params.solver == ac::FlowSolverKind::Euler;
        params.regularize_iters = regularize_iters;
        params.keep_last_k_iters = keep_last_k;
        params.lambda_kl = lambda_kl;

        NoiseSource noise(seed);
        if (!noise_in.empty()) {
            noise.replay = read_f32(noise_in);
            fprintf(stderr, "[ac] replaying %zu noise value(s) from %s\n",
                    noise.replay.size(), noise_in.c_str());
        }
        noise.recording = !noise_out.empty();

        // --- text ------------------------------------------------------------------------
        double t0 = now_s();
        const ac::TextCondition cond = ac::encode_prompt(t5_path, prompt, device);
        fprintf(stderr, "[ac] t5: %d token(s) in %.3fs -- \"%s\"\n",
                cond.tokens, now_s() - t0, prompt.c_str());

        // --- encode ----------------------------------------------------------------------
        std::vector<float> encoded;
        int frames = 0, latent_dim = 0, channels = 0, frame_rate = 0;
        {
            ac::GgufModel codec = ac::load_gguf(
                codec_path.c_str(), ac::make_backend(0, device.empty() ? nullptr : device.c_str()));
            const ac::SeanetConfig c = ac::SeanetConfig::from(codec);
            if (c.quantizer != "no_quant")
                throw std::runtime_error("mf-edit needs the quantizer-free codec; this GGUF is '" +
                                         c.quantizer + "'");
            latent_dim = c.latent_dim;
            channels = c.channels;
            frame_rate = c.frame_rate;

            int n_samples = 0, n_channels = 0, rate = 0;
            std::vector<float> audio = ac::read_wav_planar(in_path, n_samples, n_channels, rate);
            // terry crops to max_duration * frame_rate latents and no further.
            const int max_frames = seconds > 0.0 ? (int)(seconds * c.frame_rate)
                                                 : (int)(c.frame_rate * 30);
            ac::conform_audio(audio, n_samples, n_channels, rate, c, max_frames);
            frames = n_samples / c.hop_length;
            fprintf(stderr, "[ac] input: %.2fs, %d samples, %dch -> %d frames\n",
                    (double)n_samples / rate, n_samples, n_channels, frames);

            t0 = now_s();
            encoded = ac::vae_encode(codec, c, audio, n_samples, n_channels);
            fprintf(stderr, "[ac] encode: %.3fs\n", now_s() - t0);
        }

        // --- the two solves ---------------------------------------------------------------
        std::vector<float> edited;
        ac::EditReport report;
        {
            ac::GgufModel dit = ac::load_gguf(
                dit_path.c_str(), ac::make_backend(0, device.empty() ? nullptr : device.c_str()));
            const ac::DitConfig dc = ac::DitConfig::from(dit);
            if (dc.latent_dim != latent_dim)
                throw std::runtime_error("the DiT and the codec disagree about latent_dim");
            if (frames > (int)std::lround((double)dc.max_duration * frame_rate))
                fprintf(stderr, "[ac] warning: %d frames exceeds the model's trained window\n",
                        frames);
            params.cfg_coef = cfg_coef >= 0.0f ? cfg_coef : dc.cfg_coef;

            const ac::LatentStats stats = ac::mf_latent_stats(dit, latent_dim);
            // The posterior sample is the first thing terry's seed feeds.
            std::vector<float> latent = ac::vae_sample(
                encoded, frames, latent_dim, noise.draw((size_t)frames * latent_dim));
            ac::latent_normalize(latent, frames, latent_dim, stats.mean, stats.std);
            write_f32(out_prompt_latent, latent);

            fprintf(stderr, "[ac] edit: %s, %d steps, flowstep %.3f, cfg %.2f, %s -- "
                            "%d forward(s)\n",
                    solver.c_str(), params.steps, params.target_flowstep, params.cfg_coef,
                    params.regularize ? "regularized" : "no regularization",
                    ac::edit_forward_count(params));

            ac::DitRunner runner(dit, dc, cond, frames);
            int last = -1;
            auto progress = [&last](int done, int total) {
                const int pct = total > 0 ? done * 100 / total : 0;
                if (pct == last) return;
                last = pct;
                fprintf(stderr, "\r[ac] %6d / %6d  (%3d%%)", done, total, pct);
                fflush(stderr);
            };
            edited = ac::mf_edit_latent(runner, latent, frames, latent_dim, params,
                                        std::ref(noise), progress, report);
            fprintf(stderr, "\r[ac] inversion %d forward(s) in %.2fs, generation %d in %.2fs\n",
                    report.inversion_forwards, report.inversion_seconds,
                    report.generation_forwards, report.generation_seconds);

            write_f32(out_intermediate, report.intermediate);
            write_f32(out_latent, edited);
            ac::latent_denormalize(edited, frames, latent_dim, stats.mean, stats.std);
        }

        // --- decode ------------------------------------------------------------------------
        {
            ac::GgufModel codec = ac::load_gguf(
                codec_path.c_str(), ac::make_backend(0, device.empty() ? nullptr : device.c_str()));
            const ac::SeanetConfig c = ac::SeanetConfig::from(codec);
            t0 = now_s();
            int64_t out_samples = 0;
            std::vector<float> audio = ac::vae_decode(codec, c, edited, frames, out_samples);
            fprintf(stderr, "[ac] decode: %.3fs -> %lld samples\n",
                    now_s() - t0, (long long)out_samples);
            if (!out_path.empty()) {
                if (out_path.size() > 4 && out_path.compare(out_path.size() - 4, 4, ".f32") == 0)
                    write_f32(out_path, audio);
                else {
                    ac::write_wav_planar(out_path, audio.data(), (int)out_samples, channels,
                                         c.sample_rate);
                    fprintf(stderr, "[ac] wrote %s\n", out_path.c_str());
                }
            }
        }

        if (!noise_out.empty()) write_f32(noise_out, noise.record);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
