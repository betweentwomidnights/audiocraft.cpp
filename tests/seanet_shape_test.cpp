// seanet_shape_test -- pin SEANet's padding arithmetic to audiocraft's.
//
// This is where the quiet bugs live. StreamableConv1d pads asymmetrically whenever
// `padding_total` is odd (MelodyFlow's ratio-5 stage does), adds a tail so the last window
// is full, and applies all of it as one reflect pad; StreamableConvTranspose1d then trims
// the mirror image of that. Get any of it slightly wrong and the codec still produces
// plausible audio, just misaligned -- so the reference formulas are transcribed here
// straight from audiocraft/modules/conv.py and checked against our closed forms.
#include "ac/seanet.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what, long long got, long long want) {
    if (ok) return;
    fprintf(stderr, "FAIL %s: got %lld, want %lld\n", what, got, want);
    ++failures;
}

// audiocraft/modules/conv.py::get_extra_padding_for_conv1d, transcribed literally.
int64_t reference_extra_padding(int64_t length, int kernel, int stride, int padding_total) {
    const double n_frames = (double)(length - kernel + padding_total) / stride + 1.0;
    const int64_t ideal = (int64_t)(std::ceil(n_frames) - 1) * stride + (kernel - padding_total);
    return ideal - length;
}

void check_conv(int64_t frames, int kernel, int stride, int dilation) {
    char what[128];
    const ac::ConvGeometry g = ac::seanet_conv_geometry(frames, kernel, stride, dilation);

    const int effective = (kernel - 1) * dilation + 1;
    const int padding_total = effective - stride;
    const int64_t want_extra = reference_extra_padding(frames, effective, stride, padding_total);
    snprintf(what, sizeof(what), "extra(T=%lld,k=%d,s=%d,d=%d)",
             (long long)frames, kernel, stride, dilation);
    check(g.extra == want_extra, what, g.extra, want_extra);

    // StreamableConv1d: padding_right = padding_total // 2, padding_left = the rest.
    snprintf(what, sizeof(what), "pad_right(k=%d,s=%d,d=%d)", kernel, stride, dilation);
    check(g.pad_right == padding_total / 2, what, g.pad_right, padding_total / 2);
    snprintf(what, sizeof(what), "pad_left(k=%d,s=%d,d=%d)", kernel, stride, dilation);
    check(g.pad_left == padding_total - padding_total / 2, what, g.pad_left,
          padding_total - padding_total / 2);
    check(g.pad_left + g.pad_right == padding_total, "padding halves sum to the total",
          g.pad_left + g.pad_right, padding_total);

    // The conv over the padded signal must produce exactly ceil(T / stride) frames --
    // that is the whole point of the extra tail.
    const int64_t padded = frames + g.pad_left + g.pad_right + g.extra;
    const int64_t produced = (padded - effective) / stride + 1;
    snprintf(what, sizeof(what), "out_frames(T=%lld,k=%d,s=%d,d=%d)",
             (long long)frames, kernel, stride, dilation);
    check(produced == g.out_frames, what, produced, g.out_frames);
    const int64_t want_out = (frames + stride - 1) / stride;
    check(g.out_frames == want_out, "out_frames == ceil(T / stride)", g.out_frames, want_out);
}

void check_transpose(int64_t frames, int kernel, int stride) {
    char what[128];
    const ac::TransposeGeometry g = ac::seanet_transpose_geometry(frames, kernel, stride);
    const int padding_total = kernel - stride;

    snprintf(what, sizeof(what), "tr pad_right(k=%d,s=%d)", kernel, stride);
    check(g.pad_right == padding_total / 2, what, g.pad_right, padding_total / 2);
    check(g.pad_left + g.pad_right == padding_total, "tr padding halves sum to the total",
          g.pad_left + g.pad_right, padding_total);

    // torch: ConvTranspose1d gives (T-1)*s + k, then unpad1d drops pad_left and pad_right.
    const int64_t torch_full = (frames - 1) * stride + kernel;
    const int64_t torch_out = torch_full - g.pad_left - g.pad_right;
    snprintf(what, sizeof(what), "tr out_frames(T=%lld,k=%d,s=%d)",
             (long long)frames, kernel, stride);
    check(g.out_frames == torch_out, what, g.out_frames, torch_out);
    check(g.out_frames == frames * stride, "tr out_frames == T * stride",
          g.out_frames, frames * stride);

    // We ask col2im to crop pad_right from both sides, then drop the difference on the
    // left. The two must land on exactly the torch window.
    check(g.col2im_frames - g.drop_left() == torch_out, "col2im crop reaches the torch window",
          g.col2im_frames - g.drop_left(), torch_out);
    check(g.drop_left() == g.pad_left - g.pad_right, "left drop is the padding asymmetry",
          g.drop_left(), g.pad_left - g.pad_right);
    check(g.drop_left() == padding_total % 2, "left drop is 0 or 1",
          g.drop_left(), padding_total % 2);
}

} // namespace

int main() {
    // MelodyFlow's real encoder geometry: 30 s at 48 kHz through ratios [5, 6, 8, 8].
    // Ratio 5 is the asymmetric case (padding_total = 5 -> left 3, right 2).
    check_conv(1440000, 7, 1, 1);          // input conv
    check_conv(1440000, 3, 1, 1);          // residual conv1
    check_conv(1440000, 1, 1, 1);          // residual conv2
    check_conv(1440000, 10, 5, 1);         // downsample, ratio 5
    check_conv(288000, 12, 6, 1);          // ratio 6
    check_conv(48000, 16, 8, 1);           // ratio 8
    check_conv(6000, 16, 8, 1);            // ratio 8
    check_conv(750, 7, 1, 1);              // output conv

    // MusicGen's EnCodec 32 kHz geometry: ratios [4, 4, 5, 8] on the encoder side, and
    // n_residual_layers 1 with dilation_base 2 (so dilation 1 here, but cover 2 and 4).
    check_conv(960000, 7, 1, 1);
    check_conv(960000, 8, 4, 1);
    check_conv(240000, 8, 4, 1);
    check_conv(60000, 10, 5, 1);
    check_conv(12000, 16, 8, 1);
    check_conv(1500, 3, 1, 2);
    check_conv(1500, 3, 1, 4);

    // Lengths that are not a multiple of the stride are exactly when `extra` bites.
    for (int64_t frames : {1000, 1001, 1002, 1003, 1004, 1005, 999, 7, 13}) {
        for (int stride : {1, 2, 3, 5, 6, 8}) {
            check_conv(frames, 2 * stride, stride, 1);
        }
    }

    // Decoder side: kernel is always 2 * ratio, so padding_total == ratio.
    check_transpose(750, 16, 8);
    check_transpose(6000, 16, 8);
    check_transpose(48000, 12, 6);
    check_transpose(288000, 10, 5);        // odd padding_total -> one frame dropped left
    for (int stride : {2, 3, 4, 5, 6, 7, 8}) {
        for (int64_t frames : {1, 2, 17, 750}) check_transpose(frames, 2 * stride, stride);
    }

    // A full MelodyFlow encode must land on exactly 25 Hz, and a decode must invert it.
    {
        int64_t frames = 1440000;
        for (int ratio : {5, 6, 8, 8}) frames = ac::seanet_conv_geometry(frames, 2 * ratio, ratio, 1).out_frames;
        check(frames == 750, "30 s at 48 kHz encodes to 750 latents", frames, 750);
        for (int ratio : {8, 8, 6, 5}) frames = ac::seanet_transpose_geometry(frames, 2 * ratio, ratio).out_frames;
        check(frames == 1440000, "750 latents decode back to 30 s", frames, 1440000);
    }

    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("seanet_shape_test: ok\n");
    return 0;
}
