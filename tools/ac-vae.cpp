// ac-vae -- run MelodyFlow's quantizer-free SEANet codec: encode audio to latents, and
// decode latents back to audio.
//
// This is the Phase 1 parity driver. Encoding and decoding are exposed separately and both
// are deterministic, so they can be compared against torch without needing RNG parity --
// the sampling step between them (`vae_sample`) is the only stochastic part and is
// exercised only by --roundtrip.
//
// Shapes follow audiocraft: the encoder emits `mean||scale` (2 x latent_dim channels) and
// the decoder consumes latent_dim channels. Raw f32 dumps are in ggml memory order.
#include "ac/seanet.h"
#include "audio_post.h"
#include "gguf_model.h"
#include "rng.h"
#include "wav.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
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

// A named intermediate the caller wants read back alongside the result.
struct Tap {
    std::string name;
    ggml_tensor* tensor = nullptr;
};

// Build, allocate and run a one-input graph, returning the output on the host. `build` may
// append to `taps`; each tap is marked as a graph output and dumped when --taps is on.
std::vector<float> run_graph(
    const ac::GgufModel& W, const char* what,
    int64_t in0, int64_t in1, const std::vector<float>& input,
    const std::function<ggml_tensor*(ggml_context*, ggml_tensor*, std::vector<Tap>&)>& build,
    int64_t& out0, int64_t& out1, const std::string& taps_dir, size_t nodes,
    size_t arena_mb = 4096) {
    GraphArena arena;
    ggml_init_params ip = {arena_mb * 1024 * 1024, nullptr, true};
    arena.ctx = ggml_init(ip);
    if (!arena.ctx) throw std::runtime_error("failed to create the graph context");
    ggml_tensor* x = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, in0, in1);
    ggml_set_input(x);
    std::vector<Tap> taps;
    ggml_tensor* y = ggml_cont(arena.ctx, build(arena.ctx, x, taps));
    ggml_set_output(y);
    ggml_cgraph* graph = ggml_new_graph_custom(arena.ctx, nodes, false);
    if (!taps_dir.empty()) {
        for (Tap& t : taps) {
            ggml_set_output(t.tensor);
            ggml_build_forward_expand(graph, t.tensor);
        }
    }
    ggml_build_forward_expand(graph, y);
    arena.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    if (!arena.alloc || !ggml_gallocr_alloc_graph(arena.alloc, graph))
        throw std::runtime_error(std::string("failed to allocate the ") + what + " graph");

    if (input.size() != (size_t)in0 * in1)
        throw std::runtime_error(std::string(what) + ": input size does not match its shape");
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    std::string error;
    if (!ac::graph_compute_checked(W.backend, graph, what, error))
        throw std::runtime_error(error);

    if (!taps_dir.empty()) {
        for (const Tap& t : taps) {
            std::vector<float> host((size_t)ggml_nelements(t.tensor));
            ggml_backend_tensor_get(t.tensor, host.data(), 0, host.size() * sizeof(float));
            write_f32(taps_dir + "/" + t.name + ".f32", host);
            fprintf(stderr, "[ac] tap %s (%zu floats)\n", t.name.c_str(), host.size());
        }
    }

    out0 = y->ne[0];
    out1 = y->ne[1];
    std::vector<float> host((size_t)out0 * out1);
    ggml_backend_tensor_get(y, host.data(), 0, host.size() * sizeof(float));
    return host;
}

// Dump an intermediate tensor when --taps is on. Named to match tools/cossim.py's LAYOUT
// table, so a divergence can be walked layer by layer against the torch --taps dump.
void tap(const std::string& dir, const char* name, const std::vector<float>& data) {
    if (dir.empty()) return;
    write_f32(dir + "/" + name + ".f32", data);
    fprintf(stderr, "[ac] tap %s (%zu floats)\n", name, data.size());
}

void usage() {
    fprintf(stderr,
        "usage: ac-vae --codec <codec.gguf> <mode> [options]\n"
        "\n"
        "modes:\n"
        "  --encode <in.wav>        encode audio; --out-latent writes [frames, encoder_dim]\n"
        "  --decode <latent.f32>    decode a latent [frames, latent_dim]; --out writes a wav\n"
        "  --roundtrip <in.wav>     encode, sample the posterior, decode; --out writes a wav\n"
        "\n"
        "options:\n"
        "  --out <path>             output wav (decode / roundtrip)\n"
        "  --out-latent <path>      raw f32 latent dump (encode / roundtrip)\n"
        "  --seconds <n>            crop the input to n seconds (default: the model window)\n"
        "  --seed <n>               posterior sampling seed for --roundtrip (default 1234)\n"
        "  --taps <dir>             dump intermediate tensors for bisecting a divergence\n"
        "  --device <name>          backend override; also AC_DEVICE / AC_GPU / AC_THREADS\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string codec_path, encode_path, decode_path, roundtrip_path;
    std::string out_path, out_latent, taps_dir, device;
    double seconds = 0.0;
    uint64_t seed = 1234;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "--codec") codec_path = next("--codec");
        else if (a == "--encode") encode_path = next("--encode");
        else if (a == "--decode") decode_path = next("--decode");
        else if (a == "--roundtrip") roundtrip_path = next("--roundtrip");
        else if (a == "--out") out_path = next("--out");
        else if (a == "--out-latent") out_latent = next("--out-latent");
        else if (a == "--taps") taps_dir = next("--taps");
        else if (a == "--seconds") seconds = atof(next("--seconds").c_str());
        else if (a == "--seed") seed = strtoull(next("--seed").c_str(), nullptr, 10);
        else if (a == "--device") device = next("--device");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }
    if (codec_path.empty()) { fprintf(stderr, "error: --codec is required\n"); usage(); return 2; }
    const int modes = !encode_path.empty() + !decode_path.empty() + !roundtrip_path.empty();
    if (modes != 1) {
        fprintf(stderr, "error: pass exactly one of --encode, --decode, --roundtrip\n");
        usage();
        return 2;
    }

    try {
        ac::GgufModel codec = ac::load_gguf(
            codec_path.c_str(), ac::make_backend(0, device.empty() ? nullptr : device.c_str()));
        const ac::SeanetConfig c = ac::SeanetConfig::from(codec);
        if (c.quantizer != "no_quant")
            throw std::runtime_error("ac-vae handles the quantizer-free codec; this GGUF is '" +
                                     c.quantizer + "'");
        fprintf(stderr, "[ac] codec: %d Hz, %dch, hop %d (%d Hz), latent %d, encoder %d, lstm %d\n",
                c.sample_rate, c.channels, c.hop_length, c.frame_rate, c.latent_dim,
                c.encoder_dim, c.lstm_layers);

        // ---- encode ------------------------------------------------------------------
        std::vector<float> latent;   // [frames, encoder_dim] in ggml order
        int frames = 0;
        const std::string in_wav = !encode_path.empty() ? encode_path : roundtrip_path;
        if (!in_wav.empty()) {
            int n_samples = 0, n_ch = 0, rate = 0;
            std::vector<float> audio = ac::read_wav_planar(in_wav, n_samples, n_ch, rate);
            if (rate != c.sample_rate) {
                int resampled = 0;
                audio = ac::resample_planar_linear(audio, n_samples, n_ch, rate, c.sample_rate,
                                                   resampled);
                n_samples = resampled;
                rate = c.sample_rate;
            }
            // Match the service: mono is duplicated to stereo, extra channels are dropped.
            if (n_ch != c.channels) {
                std::vector<float> fixed((size_t)n_samples * c.channels);
                for (int ch = 0; ch < c.channels; ++ch) {
                    const int src = n_ch == 1 ? 0 : (ch < n_ch ? ch : n_ch - 1);
                    std::copy_n(audio.data() + (size_t)src * n_samples, n_samples,
                                fixed.data() + (size_t)ch * n_samples);
                }
                audio.swap(fixed);
                n_ch = c.channels;
            }
            // terry crops to the model window; --seconds overrides for shorter test runs.
            const int max_frames = seconds > 0.0 ? (int)(seconds * c.frame_rate) : 0;
            int keep = n_samples;
            if (max_frames > 0) keep = std::min(keep, max_frames * c.hop_length);
            keep -= keep % c.hop_length;
            if (keep <= 0) throw std::runtime_error("input is shorter than one codec frame");
            if (keep != n_samples) {
                std::vector<float> cropped((size_t)keep * n_ch);
                for (int ch = 0; ch < n_ch; ++ch)
                    std::copy_n(audio.data() + (size_t)ch * n_samples, keep,
                                cropped.data() + (size_t)ch * keep);
                audio.swap(cropped);
                n_samples = keep;
            }
            frames = n_samples / c.hop_length;
            fprintf(stderr, "[ac] input: %d samples, %dch -> %d frames\n", n_samples, n_ch, frames);
            tap(taps_dir, "mf_vae_input", audio);

            const double t0 = now_s();
            int64_t o0 = 0, o1 = 0;
            latent = run_graph(
                codec, "SEANet encoder", n_samples, n_ch, audio,
                [&](ggml_context* ctx, ggml_tensor* x, std::vector<Tap>& taps) {
                    ggml_tensor* h = ac::seanet_encode_pre(ctx, codec, x, c);
                    taps.push_back({"mf_vae_enc_pre", h});
                    h = ac::lstm_graph(ctx, codec, "codec.encoder.lstm", h, c.lstm_layers);
                    taps.push_back({"mf_vae_enc_lstm", h});
                    return ac::seanet_encode_post(ctx, codec, h, c);
                }, o0, o1, taps_dir, ac::seanet_graph_nodes(c, frames));
            const double t1 = now_s();
            if (o0 != frames || o1 != c.encoder_dim)
                throw std::runtime_error("encoder produced an unexpected latent shape");
            fprintf(stderr, "[ac] encode: %.3fs\n", t1 - t0);

            if (!out_latent.empty()) {
                write_f32(out_latent, latent);
                fprintf(stderr, "[ac] wrote %s ([%d frames, %d])\n",
                        out_latent.c_str(), frames, c.encoder_dim);
            }
        }

        // ---- pick the latent to decode -------------------------------------------------
        std::vector<float> z;   // [frames, latent_dim]
        if (!decode_path.empty()) {
            z = read_f32(decode_path);
            if (z.size() % c.latent_dim)
                throw std::runtime_error("latent is not a whole number of frames");
            frames = (int)(z.size() / c.latent_dim);
            fprintf(stderr, "[ac] latent: %d frames x %d\n", frames, c.latent_dim);
        } else if (!roundtrip_path.empty()) {
            // vae_sample: randn_like(mean) * (softplus(scale) + 1e-4) + mean.
            // Channels are the slow axis, so mean is the first latent_dim rows of [frames, C].
            ac::Rng rng(seed);
            std::vector<float> noise((size_t)frames * c.latent_dim);
            rng.fill_normal(noise.data(), noise.size());
            z.resize((size_t)frames * c.latent_dim);
            for (int ch = 0; ch < c.latent_dim; ++ch) {
                const float* mean = latent.data() + (size_t)ch * frames;
                const float* scale = latent.data() + (size_t)(ch + c.latent_dim) * frames;
                float* dst = z.data() + (size_t)ch * frames;
                const float* n = noise.data() + (size_t)ch * frames;
                for (int t = 0; t < frames; ++t) {
                    const float stdev = std::log1p(std::exp(-std::fabs(scale[t]))) +
                                        std::max(scale[t], 0.0f) + 1e-4f;
                    dst[t] = n[t] * stdev + mean[t];
                }
            }
        }

        // ---- decode ---------------------------------------------------------------------
        if (!z.empty()) {
            const double t0 = now_s();
            int64_t o0 = 0, o1 = 0;
            std::vector<float> audio = run_graph(
                codec, "SEANet decoder", frames, c.latent_dim, z,
                [&](ggml_context* ctx, ggml_tensor* x, std::vector<Tap>& taps) {
                    ggml_tensor* h = ac::seanet_decode_pre(ctx, codec, x, c);
                    taps.push_back({"mf_vae_dec_pre", h});
                    h = ac::lstm_graph(ctx, codec, "codec.decoder.lstm", h, c.lstm_layers);
                    taps.push_back({"mf_vae_dec_lstm", h});
                    return ac::seanet_decode_post(ctx, codec, h, c);
                }, o0, o1, taps_dir, ac::seanet_graph_nodes(c, frames));
            const double t1 = now_s();
            if (o1 != c.channels)
                throw std::runtime_error("decoder produced an unexpected channel count");
            fprintf(stderr, "[ac] decode: %.3fs -> %lld samples\n", t1 - t0, (long long)o0);

            if (!out_path.empty()) {
                if (out_path.size() > 4 && out_path.compare(out_path.size() - 4, 4, ".f32") == 0) {
                    write_f32(out_path, audio);
                } else {
                    ac::write_wav_planar(out_path, audio.data(), (int)o0, (int)o1, c.sample_rate);
                }
                fprintf(stderr, "[ac] wrote %s\n", out_path.c_str());
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
