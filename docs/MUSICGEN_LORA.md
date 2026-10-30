# MusicGen DoRA / LoRA experiments

`mg-train` trains attention adapters while freezing MusicGen, T5 and EnCodec.
`mg-generate` and `musicgen-server` apply them to both text generation and audio
continuations. The shared adapter implementation also serves MelodyFlow.

The baseline objective is **full-sequence teacher-forced next-code cross-entropy**:
`--prefix-frames 0` (the default). Continuation is the same causal prediction process
with an audio prefix supplied during inference; it does not need a separate
continuation loss. `--prefix-frames N` remains an optional experiment that excludes
the first N audio frames from the loss, without removing them from the context.

Codebook delays are [0, 1, 2, 3]. Inputs begin with the special token; logits at
sequence step s predict the tokens at s+1. Loss gathers real target positions only,
including the valid delayed tail. It never scores special tokens. Every codebook
has equal numbers of scored tokens, so their mean equals the mean of their individual
cross-entropies. This includes the final delayed tail, whereas AudioCraft's default
`keep_only_valid_steps=True` truncates that tail. The opt-in oracle compares against
its supported `keep_only_valid_steps=False` path.

Use ordinary flow matching with logit-normal time sampling as the MelodyFlow
baseline. The edit-bridge objective remains experimental; a generation-trained
adapter also changes edit-time velocity predictions. Whether that produces useful
edits remains an open empirical question.

## Prepare and train

See [TRAINING_CONTROLS.md](TRAINING_CONTROLS.md) for shared warmup, accumulation,
whole-track cache coverage, random crops, AdamW, FFN targets and exact resume.

Convert the official AudioCraft `state_dict.bin`, rather than the Transformers
layout. Our experiment uses `facebook/musicgen-small`, revision
`4c8334b02c6ec4e8664a91979669a501ec497792`, converted to F16 and F32.
The native LM has 420,371,456 parameters, excluding the separate text encoder/codec.

```powershell
.venv/Scripts/python.exe tools/prepare_mg_lora.py `
  --dataset C:/Users/thegr/Downloads/koan_dset --out build-cuda/koan-musicgen-cache `
  --bin build-cuda/bin/Release --lm models/musicgen-small-0.4B-v1.0-F16.gguf `
  --codec models/encodec-32khz-v1.0-F32.gguf `
  --t5 models/t5-base-encoder-0.1B-v1.0-F32.gguf `
  --seconds 12 --windows-per-track 4 --device gpu

build-cuda/bin/Release/mg-train.exe `
  --lm models/musicgen-small-0.4B-v1.0-F16.gguf `
  --cache build-cuda/koan-musicgen-cache/train `
  --validation build-cuda/koan-musicgen-cache/validation `
  --out koan-generation-dora.gguf --frames 600 --prefix-frames 0 `
  --rank 32 --first-layer 12 --steps 500 --lr 0.0001 `
  --checkpoint-every 250 --device gpu
```

Track splitting happens before window selection. The Koan pilot has 34 training
tracks / 8 held-out tracks, giving 136 / 32 windows. Captions are read as data,
deduplicated verbatim, and retain BPM/key. T5 states are projected using the exact
converted base's conditioner weight and bias; empty captions are zeroed **after**
the biased projection. Caption dropout also uses projected zeros.

Rank/alpha default to 32/32, DoRA rows is the default, and the final 12 layers'
self/cross attention projections are adapted. Batch size is one, Adam uses
0.9/0.999, epsilon 1e-8 and global gradient clipping at 1. There is no
activation checkpointing or codec/text-encoder training. Exact resume is available
through the new trainer-state sidecars. Short windows
are a memory choice, not a claim that they are optimal.

`--evaluate-only` scores the base, while `--evaluate-adapter FILE` loads an adapter
for scoring without updating or writing. Supply matching rank/alpha/first-layer
settings and `--eval-samples 0` for all validation windows. Compare objectives on
the same scoring mask; their training losses are not directly comparable.

## Shared prompt-pool contract

For `adapter.gguf`, train and publish `adapter.prompts.json` beside it. Checkpoints
receive their own corresponding sidecars. The SA3-compatible schema is:

```json
{
  "version": 1,
  "source": { "lora": "adapter.gguf", "files": 34, "unique_prompts": 33 },
  "dice": { "instrumental": ["glitch hop, neurofunk, 92 bpm, D minor"] }
}
```

Only actual **training** captions belong in this pool. Held-out captions must not
leak into it. Both new MusicGen and MelodyFlow caches retain captions, and both
trainers export sidecars. For legacy MelodyFlow caches, pass a verified
training-only `--prompt-pool JSON`; old caches without captions remain readable.
The existing Koan MelodyFlow DoRA and ordinary logit-normal checkpoints have had
sidecars backfilled from their original training manifest.

Both `mg-generate` and `mf-edit` accept `--smart-dice`, with the pool inferred from
`--lora adapter.gguf`, or supplied using `--prompt-pool FILE`. It is mutually
exclusive with an explicit prompt. Selection uses a separate seeded RNG and does
not consume audio-sampling randomness. A pool can be used with the base model for
fair comparisons. Empty/malformed pools fail before model execution.

Both services accept `--adapters-dir DIR` (default `models/adapters`), expose
`GET /api/adapters` and `GET /api/adapters/prompts?name=NAME`, and accept request
fields `adapter`, `adapter_strength`, and `smart_dice`. Adapter selection uses
registered names rather than client-supplied paths. MusicGen uses `description`;
MelodyFlow uses `custom_prompt`/`prompt`, with its existing variation field still
required. Requesting smart dice alongside an explicit prompt is an error.
MelodyFlow also accepts `adapter_pass` (`both`, the default, or `generation`) to
control whether the adapter participates in inversion as well as generation.

Use `--prompt-file FILE` for UTF-8 captions on Windows; narrow command-line
arguments can replace characters such as musical flat signs. Both cache builders
now use UTF-8 files and mark `ac.cache.caption_transport=utf8-file`. The MusicGen
pilot below was retrained after repairing affected conditioning tensors without
changing audio codes or track splits. Backfilled legacy MelodyFlow prompt pools
preserve the original captions, but do not repair older cached text conditioning.

## Verification

`mg_train_cli` checks checkpoint/evaluation invariance, training-only prompt pools,
invalid settings, frozen base files and disjoint source tracks.

`tests/mg_train_torch_parity.py` is an opt-in oracle using the installed AudioCraft
`LMModel.compute_predictions`, materialized DoRA weights and PyTorch Adam. It
checks conditional, padded, null and full-sequence/prefix losses, zero/nonzero B,
all adapter gradients and three optimizer updates. CPU F32 and CUDA F16-base
gradient relative L2 errors were below 5e-7. The native probe additionally compares
cached prefill logits with the training forward, checks intact prefix codes and
requires exact zero-strength base codes/logits. CUDA serving uses its normal
inference arithmetic, with a 0.002 absolute-logit bound; CPU uses 0.00003.

See [DORA_ROWS.md](DORA_ROWS.md) for the shared row-normalization and strength
convention. These checks establish implementation correctness, not musical quality.

GGML 0.27 adds Vulkan cross-entropy forward/backward support. MusicGen training
now runs on CPU, CUDA and Vulkan; the earlier 0.17 Vulkan limitation no longer
applies. Unsupported training ops are still rejected before graph allocation.
See [GGML 0.27 training validation](TRAINING_GGML_027_VALIDATION.md).

## First Koan pilot

Both objectives were trained from the same initial DoRA weights for 500 steps,
with a checkpoint at 250. Thirty-two held-out windows were then scored under both
masks (lower cross-entropy is better):

| Adapter | Full-sequence CE | Final-six-second CE |
| --- | ---: | ---: |
| Base | 4.4502983 | 4.3450924 |
| Generation, 250 steps | 4.4492819 | 4.3465145 |
| Generation, 500 steps | 4.4835715 | 4.3816088 |
| Continuation, 500 steps | 4.5061727 | 4.4021823 |

The full-sequence 250-step checkpoint is effectively tied with the base. Longer
training worsened both measures. The generation-trained 500-step adapter scores
better than the continuation-trained 500-step adapter under **both** masks, which
supports keeping full-sequence training as the baseline. It does not establish
that any adapter makes better music.

`build-cuda/koan-musicgen-dora-utf8-study/listen.html` contains nineteen 30-second WAV
comparisons: unconditional and captioned Bluey and Gary continuations (identical
six-second prefixes across variants) plus captioned text-only generation.
Unconditional continuations use an empty description and skip text cross-attention
and text guidance; audio-prefix conditioning remains. Captions/seeds are matched within each group;
generated-section RMS is matched for listening. Original levels, generated codec
tokens, commands, logs, model conversion hashes, validation and HTTP checks are
retained alongside the page.
