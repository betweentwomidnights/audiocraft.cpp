# The two services

`melodyflow-server` and `musicgen-server` put each model behind HTTP: audio and a prompt in,
audio out, with a session id to poll while it runs. They are the long-running form of
`mf-edit` and `mg-generate` and depend on nothing outside this repository.

| | melodyflow-server | musicgen-server |
|---|---|---|
| default port | 8002 | 8000 |
| model | MelodyFlow `edit` | MusicGen `generate_continuation` |
| audio | 48 kHz stereo, 30 s window | 32 kHz mono, 30 s output |

The routes and reply shapes deliberately match gary4local's `localhost_melodyflow.py` and
`g4l_localhost.py`, because a client already speaks them — gary4juce polls
`/api/juce/poll_status/<id>` and reads particular field names. That is a compatibility
choice, not a coupling: nothing here imports from gary4local or assumes it is present.

```bash
melodyflow-server --port 8002 --models-dir models
musicgen-server  --port 8000 --models-dir models
```

`AC_MODELS_DIR` and `AC_DEVICE` work as they do for the CLI tools, so the Tauri side stays
declarative and never has to pass arguments.

## Routes

**terry**

```
GET  /health                        {status, service, model_loaded, models_dir, device}
GET  /variations                    {variations: {name: {prompt, flowstep}}}   33 presets
POST /transform                     JSON in, the WAV itself out (audio/wav)
POST /api/juce/transform_audio      -> {success, session_id, seed, message, note}
GET  /api/juce/poll_status/<id>     -> {status, progress, transform_in_progress, seed,
                                        audio_data on completion, error on failure}
POST /api/juce/undo_transform       -> {success, audio_data}   the audio that was sent in
```

`transform_audio` takes `audio_data` (base64 WAV), `variation` (a preset name), and
optionally `flowstep`, `solver` (`euler` or `midpoint`), `custom_prompt` and `seed`. A custom
prompt replaces the preset's rather than adding to it, which is what `process_audio` does.

**gary**

```
GET  /health                        {status, service, session_store, models, ...}
GET  /api/models                    {models: {small: [...], medium: [...], large: [...]}}
POST /api/juce/process_audio        continue from the FIRST prompt_duration seconds
POST /api/juce/continue_music       continue from the LAST prompt_duration seconds
POST /api/juce/retry_music          the same input again, with new parameters
GET  /api/juce/poll_status/<id>     -> {status, progress, generation_in_progress,
                                        generation_seed, audio_data, error}
```

The generation routes take `audio_data`, `model_name`, `prompt_duration` (default 6),
`description`, and optionally `duration` (default 30), `top_k`, `temperature`, `cfg_coef`
and `seed`. `retry_music` takes `session_id` and re-runs that session's **input** — not its
result, which would compound the continuation.

### What the prompt window ends on decides the seam

`continue_music` takes the tail of the clip, and that is its whole point — but it means
the model's last 6 s of context end wherever the user's audio happens to stop. If that lands
in the gap between hits, the model continues the gap.

Measured on a 9.7 s drum loop dragged out of gary4juce, whose final 100 ms sit at about a
ninth of the clip's average level: every seed goes near-silent (-55 to -78 dBFS) for about a
beat at exactly the prompt boundary, then recovers. The same clip through `process_audio`,
whose window ends at 6.0 s on a transient at full level, walks through the seam with no dip
at any seed.

This is MusicGen's behaviour and not something the port introduced: greedy decoding on that
clip and that window is **2400/2400 identical to torch**, so the hole is in audiocraft's own
codes. It is worth knowing about because it looks exactly like a bug in the codec or the
delay pattern, and it is neither — it is a weak prompt.

## Deliberate differences

Three, all of them improvements rather than shortcuts.

**`/api/models` reports what is on disk.** The Python service returns a hardcoded list of
fourteen `thepatch/*` repositories and downloads whichever the user picks. A native service
cannot fetch and convert a PyTorch checkpoint on demand, so offering a model that has not
been converted would fail at generation time instead of in the picker. `musicgen-server` scans
`AC_MODELS_DIR` for `musicgen-*.gguf`, names each by the part between the prefix and the
version (`musicgen-vanya-dnb-0.4B-v1.0-F16.gguf` becomes `vanya-dnb`), and groups them by the
parameter count in the file rather than by guessing from the name. F16 wins over F32 for the
same model: half the file, for a difference that never reaches the tokens gary samples.

**`/variations` works.** The Python reads `VARIATIONS[name]['flowstep']` where the table
defines `default_flowstep`, so that endpoint raises a `KeyError` and 500s today. The preset
table here is generated from the service's own `variations.py` by
`tools/gen_variations.py`, so it stays the service's data without being transcribed by hand.

**`/transform` is JSON only.** The Python also accepts a multipart upload and a server-side
`audio_file_path`. Nothing in gary4local sends either — the plugin uses the `/api/juce/*`
routes — so this takes `audio_data` and replies with the WAV bytes, which is what
`send_file` did.

One difference that is *not* a choice: an unknown session id returns 200 with
`status: "unknown"` on gary and 404 on terry, because that is what each Python service does
and gary4juce treats the two differently.

## How a request runs

Both services hold **one generation at a time**, as the Python ones do, and for the same
reason: a request holds most of a GPU, so a second concurrent job does not go faster, it goes
out of memory. A request returns a `session_id` immediately, a worker thread takes the
generation lock, and the client polls.

Models are loaded, used and released *within* a request, in the same order and by the same
calls the validated CLI tools use — text encoder, codec, model, codec again. Keeping them
resident would save roughly three seconds a request and cost the headroom that keeps a 30 s
window inside 8 GB, which is the wrong trade for the machines this ships to. It is a flag
away if that changes.

Finished sessions are swept once there are more than 64 of them. Each one holds a base64 WAV
— about 8 MB for a 30 s stereo result — and the client has long since polled it; `undo_transform`
is the only thing that reaches back, and it does so within a session's lifetime.

## Measured

Both driven end to end against a real request, on an RTX 5070 Laptop.

| | request | result | wall |
|---|---|---|---|
| terry `transform_audio`, accordion_folk, euler | 30 s stereo 48 kHz | 30 s stereo | ~30 s |
| gary `continue_music`, 6 s prompt, 8 s output | 123 s 44.1 kHz stereo (29 MB of JSON) | 8 s mono 32 kHz | ~12 s |

Progress reaches the client at the same granularity as the Python: terry reports across both
solver passes against one total, gary across the delay-pattern steps.

Audio comes back as 16-bit PCM WAV, which is what `torchaudio.save` and `soundfile.write`
default to and therefore what the plugin already handles.

## Remaining integration gates

- **Not yet wired into gary4local.** `service_manager.rs` still points at the Python
  entrypoints. Switching it over is a one-line change per service plus dropping the venv
  build, but it has not been done or tested from the Tauri side.
- **Model publication.** Only `thepatch/vanya_ai_dnb_0.1` has been converted. The other
  thirteen `thepatch/*` finetunes need converting and publishing as GGUF before
  `/api/models` can offer what the Python service does. The converter is driven entirely by
  `xp.cfg` and should handle medium and large unchanged.
- **Quantized tiers.** Q8_0 and below are unmeasured for both models. `tools/cossim.py`
  reports rms-envelope and log-spectrum alongside raw cosine for exactly this.
- **No queue depth.** `queue_status` is always `{}`. The Python services report a position
  and an estimate; nothing in the plugin depends on it, and one job at a time makes it
  meaningless, but a client that displays it will show nothing.
- **No `/unload`.** Models are already released after each request, so there is nothing to
  unload — but the Tauri orchestrator may expect the route to exist.
