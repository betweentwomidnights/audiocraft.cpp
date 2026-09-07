// shared_headers_test -- every header ported from sa3.cpp compiles standalone and its
// pure-host helpers behave.
//
// These headers land ahead of the code that uses them (wav/audio_post are Phase 1, rng and
// encoding are Phase 3+), so without this they would rot silently until the phase that
// needs them. It is a compile gate first and a behaviour spot-check second.
#include "audio_post.h"
#include "encoding.h"
#include "gguf_model.h"
#include "nn.h"
#include "rng.h"
#include "wav.h"
#include "ac/t5.h"
#include "ac/tokenizer.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

} // namespace

int main() {
    // encoding.h: the --encoding vocabulary, shared with sa3.cpp.
    check(ac::encoding_suffix("q4_k_m") == "Q4_K_M", "encoding_suffix canonicalizes");
    check(ac::encoding_suffix("q4_km") == "Q4_K_M", "encoding_suffix accepts the alias");
    check(ac::encoding_suffix("nonsense").empty(), "encoding_suffix rejects garbage");
    check(ac::encoding_labels().size() == 6, "encoding_labels lists every canonical tier");

    // rng.h: the same seed must reproduce the same stream, different seeds must not.
    // MelodyFlow's edit() draws noise twice (VAE posterior, then per regularized solver
    // step), so a reproducible stream is what makes a seeded transform reproducible.
    std::vector<float> a(4096), b(4096), c(4096);
    ac::Rng(1234).fill_normal(a.data(), a.size());
    ac::Rng(1234).fill_normal(b.data(), b.size());
    ac::Rng(1235).fill_normal(c.data(), c.size());
    check(a == b, "Rng is deterministic for a seed");
    check(a != c, "Rng differs across seeds");
    double mean = 0.0, sq = 0.0;
    for (float v : a) { mean += v; sq += (double)v * v; }
    mean /= (double)a.size();
    const double stddev = std::sqrt(sq / (double)a.size() - mean * mean);
    check(std::fabs(mean) < 0.05, "Rng is approximately zero-mean");
    check(std::fabs(stddev - 1.0) < 0.05, "Rng is approximately unit-variance");

    // audio_post.h: MusicGen prompts must land on exactly 32 kHz mono and MelodyFlow input
    // on exactly 48 kHz stereo, so the resampled length has to be exact, not approximate.
    std::vector<float> mono(1000, 0.25f);
    int out_samples = 0;
    std::vector<float> up = ac::resample_planar_linear(mono, 1000, 1, 16000, 32000, out_samples);
    check(out_samples == 2000, "resample 16k -> 32k doubles the sample count");
    check(up.size() == 2000, "resample 16k -> 32k returns that many samples");
    std::vector<float> down = ac::resample_planar_linear(up, out_samples, 1, 32000, 16000, out_samples);
    check(out_samples == 1000, "resample 32k -> 16k halves it again");
    check(down.size() == 1000, "round-trip length is preserved");
    std::vector<float> same = ac::resample_planar_linear(mono, 1000, 1, 48000, 48000, out_samples);
    check(out_samples == 1000 && same == mono, "equal rates pass through untouched");

    // wav.h: a planar buffer serializes to a RIFF header plus 16-bit interleaved samples.
    std::vector<float> stereo(2 * 128, 0.0f);
    const std::string wav = ac::wav_planar_bytes(stereo.data(), 128, 2, 48000);
    check(wav.size() == 44 + (size_t)128 * 2 * 2, "wav_planar_bytes emits header + PCM16 data");
    check(wav.compare(0, 4, "RIFF") == 0, "wav starts with RIFF");
    check(wav.compare(8, 4, "WAVE") == 0, "wav declares WAVE");

    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("shared_headers_test: ok\n");
    return 0;
}
