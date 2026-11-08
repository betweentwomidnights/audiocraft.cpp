// mg/continuation.h -- what gary's `continue_music` hands back.
//
// MusicGen's output starts with the prompt it was primed with, so the generated window is
// only the tail of the answer. gary returns the whole input with the continuation appended
// (`_continue_music_impl_v2` in g4laudio.py):
//
//     song = song.repeat(2, 1) if mono
//     output = resample(output, 32000, sr)
//     combined = torch.cat([song[..., :-int(prompt_duration * sr)], output], dim=1)
//
// The input is kept up to where the prompt began and the output replaces the rest, so the
// result is (input length - prompt) + generated length, at the input's rate.
#pragma once

#include "audio_post.h"

#include <algorithm>
#include <vector>

namespace ac {

// `input` and `generated` are planar. The result is at `in_rate`, with the input's channel
// count widened to stereo as gary widens it; a mono `generated` is duplicated across them.
inline std::vector<float> append_continuation(const std::vector<float>& input, int in_samples,
                                              int in_ch, int in_rate,
                                              const std::vector<float>& generated,
                                              int gen_samples, int gen_ch, int gen_rate,
                                              double prompt_seconds, int& out_samples,
                                              int& out_ch) {
    // `song[..., :-n]` is empty both when n covers the whole song and when n is 0; the
    // second is also what a zero prompt means here, since the whole input was the prompt.
    const int prompt = (int)(prompt_seconds * in_rate);
    const int head = prompt > 0 ? std::max(0, in_samples - prompt) : 0;

    int tail = 0;
    const std::vector<float> resampled =
        resample_planar_sinc(generated, gen_samples, gen_ch, gen_rate, in_rate, tail);

    out_ch = std::max(in_ch, 2);
    out_samples = head + tail;
    std::vector<float> out((size_t)out_samples * out_ch);
    for (int ch = 0; ch < out_ch; ++ch) {
        float* dst = out.data() + (size_t)ch * out_samples;
        const int src_in = ch < in_ch ? ch : in_ch - 1;
        const int src_gen = ch < gen_ch ? ch : (gen_ch == 1 ? 0 : gen_ch - 1);
        std::copy_n(input.data() + (size_t)src_in * in_samples, head, dst);
        std::copy_n(resampled.data() + (size_t)src_gen * tail, tail, dst + head);
    }
    return out;
}

} // namespace ac
