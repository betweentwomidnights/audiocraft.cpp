// ac-encodec -- run MusicGen's EnCodec 32 kHz on its own: audio to codes, codes to audio.
//
// The Phase 5 parity driver. Both directions are deterministic -- the RVQ is a
// nearest-neighbour search, not a sample -- so each can be compared against torch
// independently, which is what separates a codec bug from an LM bug when the end-to-end
// audio is off.
//
// Codes are raw i32 [n_q, frames] in the same order the LM uses; audio dumps are raw f32
// planar [samples, channels] in ggml memory order.
#include "ac/codec.h"
#include "gguf_model.h"
#include "mg/encodec.h"
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

template <typename T>
void write_raw(const std::string& path, const std::vector<T>& data, const char* unit) {
    if (path.empty()) return;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open " + path);
    const size_t wrote = fwrite(data.data(), sizeof(T), data.size(), f);
    fclose(f);
    if (wrote != data.size()) throw std::runtime_error("short write to " + path);
    fprintf(stderr, "[ac] wrote %s (%zu %s)\n", path.c_str(), data.size(), unit);
}

void usage() {
    fprintf(stderr,
        "usage: ac-encodec --codec <encodec.gguf> <mode> [options]\n"
        "\n"
        "  --encode <in.wav>        encode audio; --out-codes writes [n_q, frames] i32\n"
        "  --decode <codes.i32>     decode codes; --out writes a wav\n"
        "  --roundtrip <in.wav>     encode then decode, to hear what the codec costs\n"
        "\n"
        "  --out <path>             output wav (.f32 writes raw planar floats instead)\n"
        "  --out-codes <path>       raw i32 code dump\n"
        "  --out-latent <path>      raw f32 encoder output, before quantization\n"
        "  --out-quantized <path>   raw f32 latent the codes decode back to\n"
        "  --seconds <n>            crop the input\n"
        "  --from-end               take the tail of the input, as gary's continue does\n"
        "  --device <name>          backend override; also AC_DEVICE / AC_GPU / AC_THREADS\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string codec_path, encode_path, decode_path, roundtrip_path;
    std::string out_path, out_codes, out_latent, out_quantized, out_input, device;
    double seconds = 0.0;
    bool from_end = false;

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
        else if (a == "--out-codes") out_codes = next("--out-codes");
        else if (a == "--out-latent") out_latent = next("--out-latent");
        else if (a == "--out-quantized") out_quantized = next("--out-quantized");
        else if (a == "--out-input") out_input = next("--out-input");
        else if (a == "--seconds") seconds = atof(next("--seconds").c_str());
        else if (a == "--from-end") from_end = true;
        else if (a == "--device") device = next("--device");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }
    if (codec_path.empty()) { fprintf(stderr, "error: --codec is required\n"); usage(); return 2; }
    if (!encode_path.empty() + !decode_path.empty() + !roundtrip_path.empty() != 1) {
        fprintf(stderr, "error: pass exactly one of --encode, --decode, --roundtrip\n");
        usage();
        return 2;
    }

    try {
        ac::GgufModel codec = ac::load_gguf(
            codec_path.c_str(), ac::make_backend(0, device.empty() ? nullptr : device.c_str()));
        const ac::SeanetConfig c = ac::SeanetConfig::from(codec);
        if (!c.rvq())
            throw std::runtime_error("ac-encodec needs an RVQ codec; this GGUF is '" +
                                     c.quantizer + "' (use ac-vae for MelodyFlow's)");
        fprintf(stderr, "[ac] codec: %d Hz, %dch, hop %d (%d Hz), latent %d, rvq %dx%d\n",
                c.sample_rate, c.channels, c.hop_length, c.frame_rate, c.latent_dim,
                c.rvq_n_q, c.rvq_bins);
        const ac::RvqCodebooks books =
            ac::rvq_load(codec, c.rvq_n_q, c.rvq_bins, c.latent_dim);

        // ---- encode ------------------------------------------------------------------
        std::vector<int32_t> codes;
        int frames = 0;
        const std::string in_wav = !encode_path.empty() ? encode_path : roundtrip_path;
        if (!in_wav.empty()) {
            int n_samples = 0, n_ch = 0, rate = 0;
            std::vector<float> audio = ac::read_wav_planar(in_wav, n_samples, n_ch, rate);
            const int max_frames = seconds > 0.0 ? (int)(seconds * c.frame_rate) : 0;
            ac::conform_audio(audio, n_samples, n_ch, rate, c, max_frames, from_end);
            frames = n_samples / c.hop_length;
            fprintf(stderr, "[ac] input: %.2fs, %d samples, %dch -> %d frames\n",
                    (double)n_samples / rate, n_samples, n_ch, frames);
            // The conformed audio, before the encoder sees it. When a comparison against
            // torch disagrees this is the first place to look: resampling and channel
            // mixing happen here and are the easiest things to get subtly different.
            write_raw(out_input, audio, "floats");

            double t0 = now_s();
            const std::vector<float> latent =
                ac::codec_encode(codec, c, audio, n_samples, n_ch);
            const double encoded_at = now_s();
            codes = ac::rvq_encode(books, latent, frames);
            fprintf(stderr, "[ac] encode: %.3fs conv + %.3fs quantize\n",
                    encoded_at - t0, now_s() - encoded_at);
            write_raw(out_latent, latent, "floats");
            write_raw(out_codes, codes, "codes");
        }

        // ---- decode ---------------------------------------------------------------------
        if (!decode_path.empty()) {
            codes = read_i32(decode_path);
            if (codes.size() % c.rvq_n_q)
                throw std::runtime_error("codes are not a whole number of frames");
            frames = (int)(codes.size() / c.rvq_n_q);
            fprintf(stderr, "[ac] codes: %d x %d frames\n", c.rvq_n_q, frames);
        }
        if (!codes.empty() && decode_path.empty() && roundtrip_path.empty()) return 0;

        if (!codes.empty()) {
            const double t0 = now_s();
            const std::vector<float> latent = ac::rvq_decode(books, codes, frames);
            write_raw(out_quantized, latent, "floats");
            int64_t out_samples = 0;
            const std::vector<float> audio =
                ac::codec_decode(codec, c, latent, frames, out_samples);
            fprintf(stderr, "[ac] decode: %.3fs -> %lld samples\n",
                    now_s() - t0, (long long)out_samples);

            if (!out_path.empty()) {
                if (out_path.size() > 4 && out_path.compare(out_path.size() - 4, 4, ".f32") == 0)
                    write_raw(out_path, audio, "floats");
                else {
                    ac::write_wav_planar(out_path, audio.data(), (int)out_samples, c.channels,
                                         c.sample_rate);
                    fprintf(stderr, "[ac] wrote %s\n", out_path.c_str());
                }
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
