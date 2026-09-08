# MusicGen's EnCodec 32 kHz

The last piece of gary. The language model works in discrete codes; this is what turns audio
into them and back.

```
wav ─► resample ─► mono ─► crop ─► SEANet encoder ─► RVQ ─► codes ─► LM
                                                                     │
                        wav ◄─ SEANet decoder ◄─ RVQ⁻¹ ◄─────────────┘
```

The conv stack is the same `ac/seanet.h` MelodyFlow's VAE uses — same padding arithmetic,
same residual blocks, same LSTM bottleneck — differing only in activation (ELU rather than
snake), ratios and channel count. What is new is the bottleneck (`src/mg/encodec.h`) and, as
it turned out, the resampler.

## It is not audiocraft's codec

MusicGen's `compression_state_dict.bin` is 78 bytes and says `pretrained:
facebook/encodec_32khz`. `load_compression_model` sees that and hands off to
`CompressionModel.get_pretrained`, which loads the model through **HuggingFace
`transformers`** as an `EncodecModel` wrapped in `HFEncodecCompressionModel`.

So the checkpoint is HF's layout, not audiocraft's: `encoder.layers.N.conv.weight_g` rather
than `encoder.model.N.conv.conv.parametrizations.weight.original0`, and legacy
`weight_g`/`weight_v` rather than torch parametrizations. `transformers`' implementation is
a faithful port of the same SEANet — the padding, the residual blocks and the module order
all line up line for line — so only the names differ, and `tools/convert_seanet.py` grew a
notion of source dialect rather than a second copy of the walk.

The layer *indices* are identical in both, which looks like luck and is not: activations
occupy a module slot without owning parameters, so both encoders run
0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15. `tests/seanet_converter_test.py` converts the same
miniature topology through both readers and asserts the emitted names and shapes match,
because `ac/seanet.h` loads both codecs through one set of names and a drift in either
reader would break the other model silently.

Pointing the converter at MusicGen's redirect file says so:

```
error: compression_state_dict.bin is a pointer to 'facebook/encodec_32khz', not a
checkpoint: MusicGen's codec is loaded through HuggingFace transformers. Convert the
downloaded model instead, e.g. --src .../models--facebook--encodec_32khz/.../model.safetensors
```

**On `betweentwomidnights/encodec.cpp`:** it is a fork of `PABannier/encodec.cpp`, which
targets Meta's original 24 kHz EnCodec. It was listed as prior art in the original plan and
it is not on the path here. `ac/seanet.h` already *was* this architecture from Phase 1, its
tensor names are Meta's rather than HF's, and it carries its own binary format and its own
ggml pin. Worth reading for graph shape if you are starting from nothing; nothing to import.

## The residual vector quantizer

Four codebooks of 2048 vectors in 128 dimensions, applied in series: the first quantizes the
latent, the second quantizes what the first got wrong, and so on. Encoding picks the nearest
vector and subtracts it; decoding sums one lookup per level.

`EncodecEuclideanCodebook.quantize` writes the distance as `-(‖x‖² − 2x·eᵀ + ‖e‖²)` and takes
the max. `‖x‖²` is the same for every candidate, so `mg/encodec.h` maximizes `2x·e − ‖e‖²`
instead — same argmax, one fewer pass over the frame, and `‖e‖²` precomputed once at load.

Both directions run on the host. Decoding is four lookups per frame; encoding a 6 s prompt
is 300 frames × 4 levels × 2048 candidates × 128 dims, about 0.3 GFLOP, and takes 0.1 s. It
happens once per request and moving it into a graph would buy little.

**How many codebooks?** Not stored — derived, and by two different formulas.
`EncodecConfig.num_quantizers` hardcodes ten bits per codebook
(`1000·bw // (frame_rate·10)`); audiocraft's `HFEncodecCompressionModel` uses
`log2(cardinality)` and calls `set_num_codebooks(max(...))`. For the shipped model both give
4 and the difference never surfaces. The converter follows audiocraft's, because that is the
one that selects the count at inference; `tests/gen_codec_fixture.py` has to use HF's,
because that is what decides how many layers get built.

The checkpoint also carries `cluster_size`, `embed_avg` and `inited` alongside each
codebook. That is EMA bookkeeping from training and the converter drops it *by name* rather
than by prefix, so a checkpoint carrying something genuinely new still trips the
unrecognized-tensor check.

## The resampler was the hard part

gary's inputs arrive at 44.1 or 48 kHz and MusicGen wants 32 kHz, so this is on the path of
every request. `resample_planar_linear` — the sa3.cpp port — reproduced **97.4%** of torch's
RVQ codes from a 44.1 kHz file. That is the worst kind of number: close enough to look like
agreement, far enough to be a different piece of music by the end of a 30 s continuation, and
audibly worse besides, because linear interpolation across that ratio aliases.

`resample_planar_sinc` is `torchaudio.functional.resample` transcribed (`sinc_interp_hann`,
`lowpass_filter_width=6`, `rolloff=0.99`). Rates are reduced by their GCD first, which keeps
the filter bank small — 44100 → 32000 becomes 441 → 320, so 320 filters of 459 taps rather
than 32000 of them — and the whole resample is one strided convolution.

It then reproduced **75.5%**, which is worse. Two bugs' worth of lesson:

**Order matters, and gary's is not the obvious one.** `resample_for_model` runs on the track
as loaded and `safe_musicgen_continuation_v2` mixes to mono *after*. Resampling a mono mix
is not the same computation as mixing two resampled channels. The reference dumper had it
backwards.

**torchaudio's output length goes through a float32.** It truncates to
`ceil(new · length / orig)`, but computes that with `torch.as_tensor(...)`, which builds a
**float32** — so the ratio loses its fraction before the ceiling is taken. For a 123 s
44.1 kHz file, 3949715.011 rounds to 3949715.0 and `ceil` leaves it there: one sample short
of the true ceiling. One sample sounds like nothing, and it shifts a tail crop onto different
audio, which moved a quarter of the RVQ codes.

With both fixed the conformed audio matches torchaudio at cossim 0.9999999 and the codes
match exactly. `tests/codec_test.cpp` pins the resampler against torchaudio fixtures and
states the length quirk as data, so a future simplification to "just use the exact ceiling"
fails rather than drifting.

## Measured parity

`facebook/encodec_32khz` against torch on CPU. 6 s of audio, 300 frames.

| checkpoint | cossim | max abs err |
|---|---|---|
| conformed audio, from a raw 44.1 kHz stereo wav | 0.9999999 | 6.4e-06 |
| encoder latent, before quantization | 1.0000000 | 9.3e-04 |
| **RVQ codes** | **1200 / 1200 exact** | — |
| the latent those codes decode back to | 1.0000000 | 0.0 |
| decoded audio | 1.0000000 | 7.2e-07 |

And the whole of gary, end to end, greedy, from that same raw wav:

| | |
|---|---|
| prompt codes (6 s) | 1200 / 1200 exact |
| **generated codes (30 s)** | **6000 / 6000 exact** |
| decoded audio (30 s) | cossim 1.0000000, rms-envelope 1.0000000, log-spectrum 1.0000000 |

Every stage of gary's transform now reproduces torch-on-CPU exactly, from a real file rather
than from a pre-conformed fixture.

CUDA still diverges in the tokens for the reason [MUSICGEN_LM.md](MUSICGEN_LM.md) explains —
`argmax` is discontinuous and a 2.8e-2 logit difference flips it — but the codec itself is
device-independent enough that its own checkpoints match on either.

## Speed

30 s of audio from a 6 s prompt: resample, encode, 1202 LM decode steps, synthesize.

| | encode | LM decode | synthesize | wall |
|---|---|---|---|---|
| torch CPU | — | — | — | 127.3 s |
| audiocraft.cpp CPU F32 | 0.66 s | 94.8 s (12.7 steps/s) | 3.5 s | **101.4 s** |
| audiocraft.cpp CUDA F32 | 0.20 s | 24.0 s (50.0 steps/s) | 0.42 s | 27.7 s |
| audiocraft.cpp CUDA F16 | 0.22 s | 19.8 s (60.7 steps/s) | 0.44 s | **23.2 s** |

torch's figure is the LM loop plus the codec decode with the model already resident; ours are
whole-process, including loading three models. torch on CUDA does the same work in 32.7 s
(see [MUSICGEN_LM.md](MUSICGEN_LM.md)), so **the C++ is ahead on both devices**, and the LM
loop still runs its two guidance streams sequentially where audiocraft batches them.

Codec time is not the interesting number here — it is under 2% of the total on GPU — but note
the CPU decode is 3.5 s against 0.42 s on CUDA, which is the F32 im2col from
[MELODYFLOW_VAE.md](MELODYFLOW_VAE.md) showing up again.

## Reproducing

```bash
# convert (the repo venv has gguf, torch and safetensors)
ENC=~/.cache/huggingface/hub/models--facebook--encodec_32khz/snapshots/<rev>
.venv/Scripts/python.exe tools/convert_seanet.py \
  --src "$ENC/model.safetensors" --out models/encodec-32khz-v1.0-F32.gguf

# the codec on its own, both directions (gary's venv for the reference)
PYTHONPATH=.../services/gary .../env/Scripts/python.exe tools/dump_mg_codec_refs.py \
    --input testdata/music.wav --seconds 6 --from-end --out refcodec
ac-encodec --codec models/encodec-32khz-v1.0-F32.gguf \
  --roundtrip testdata/music.wav --seconds 6 --from-end \
  --out-input cppcodec/conformed.f32 --out-latent cppcodec/mg_codec_latent.f32 \
  --out-codes cppcodec/mg_codec_codes.i32 --out-quantized cppcodec/mg_codec_quantized.f32 \
  --out cppcodec/mg_codec_audio.f32
python tools/cossim.py --ref refcodec --cpp cppcodec --channels 1

# and gary's whole transform
mg-generate --lm models/musicgen-vanya-dnb-0.4B-v1.0-F16.gguf \
            --t5 models/t5-base-encoder-0.1B-v1.0-F16.gguf \
            --codec models/encodec-32khz-v1.0-F32.gguf \
            --input in.wav --prompt "drum and bass" --out out.wav
```

`ac-encodec --out-input` is the first thing to check when a comparison disagrees: resampling
and channel mixing happen there and are the easiest things to get subtly different.

## Remaining integration gates

- **No `gary-server` yet.** Phase 6 owns `:8000`, the job queue and the `session_id` polling,
  plus splicing the continuation back onto the original track the way `continue_music` does.
- **Whole-file resampling.** `conform_audio` resamples the entire input before cropping,
  because that is what gary does and matching it means doing the same. For a two-minute track
  that is a couple of seconds of work to keep six seconds of audio. A windowed resample would
  fix it but changes the filter's phase relative to the reference, so it needs its own parity
  check.
- **Model coverage.** One codec, one LM checkpoint. Every `thepatch/*` finetune points at the
  same `facebook/encodec_32khz`, so the codec should be shared across all of them — but
  "should" is not "did".
- **Quantized tiers.** The codec is F32 only; `convert_seanet.py` refuses anything else,
  which was the right call for MelodyFlow's sixteen stacked convolutions and has not been
  revisited for this one.

## Deferred

- **`normalize` and chunking.** `facebook/encodec_32khz` sets `normalize: false` and
  `chunk_length_s: null`, so there is no per-chunk scale and no overlap-add. The converter
  rejects a checkpoint that turns either on rather than silently ignoring it.
- **Bandwidth selection.** audiocraft calls `set_num_codebooks(max(possible))` and never
  changes it, so the codec always runs all four codebooks. Fewer would be a smaller LM
  vocabulary and a different model.
- **Encoding on the GPU.** The RVQ search is host-side and takes 0.1 s per request.
