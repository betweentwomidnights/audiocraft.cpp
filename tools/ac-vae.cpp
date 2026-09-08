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
#include "gguf_model.h"
#include "mf/vae.h"
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

        ac::TapFn tap_sink;
        if (!taps_dir.empty()) {
            tap_sink = [&](const char* name, const std::vector<float>& data) {
                tap(taps_dir, name, data);
            };
        }

        // ---- encode ------------------------------------------------------------------
        std::vector<float> latent;   // [frames, encoder_dim] in ggml order
        int frames = 0;
        const std::string in_wav = !encode_path.empty() ? encode_path : roundtrip_path;
        if (!in_wav.empty()) {
            int n_samples = 0, n_ch = 0, rate = 0;
            std::vector<float> audio = ac::read_wav_planar(in_wav, n_samples, n_ch, rate);
            // Resample, force the codec's channel count, and crop to whole frames --
            // terry's own input handling, shared with mf-edit. --seconds shortens it
            // further for test runs.
            const int max_frames = seconds > 0.0 ? (int)(seconds * c.frame_rate) : 0;
            ac::conform_audio(audio, n_samples, n_ch, rate, c, max_frames);
            frames = n_samples / c.hop_length;
            fprintf(stderr, "[ac] input: %d samples, %dch -> %d frames\n", n_samples, n_ch, frames);
            tap(taps_dir, "mf_vae_input", audio);

            const double t0 = now_s();
            latent = ac::vae_encode(codec, c, audio, n_samples, n_ch, tap_sink);
            fprintf(stderr, "[ac] encode: %.3fs\n", now_s() - t0);

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
            ac::Rng rng(seed);
            std::vector<float> noise((size_t)frames * c.latent_dim);
            rng.fill_normal(noise.data(), noise.size());
            z = ac::vae_sample(latent, frames, c.latent_dim, noise);
        }

        // ---- decode ---------------------------------------------------------------------
        if (!z.empty()) {
            const double t0 = now_s();
            int64_t out_samples = 0;
            std::vector<float> audio = ac::vae_decode(codec, c, z, frames, out_samples, tap_sink);
            fprintf(stderr, "[ac] decode: %.3fs -> %lld samples\n",
                    now_s() - t0, (long long)out_samples);

            if (!out_path.empty()) {
                if (out_path.size() > 4 && out_path.compare(out_path.size() - 4, 4, ".f32") == 0) {
                    write_f32(out_path, audio);
                } else {
                    ac::write_wav_planar(out_path, audio.data(), (int)out_samples, c.channels,
                                         c.sample_rate);
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
