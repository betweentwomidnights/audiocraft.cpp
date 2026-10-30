import os
"""Check that evaluation/checkpointing does not alter optimizer sampling or weights."""
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import gguf
    import numpy as np
except ImportError:
    print("SKIP: gguf/numpy are unavailable")
    sys.exit(0)


def write(path, architecture, tensors, metadata):
    writer = gguf.GGUFWriter(str(path), architecture)
    for key, value in metadata.items():
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        else:
            writer.add_string(key, value)
    for key, value in tensors.items():
        writer.add_tensor(key, np.ascontiguousarray(value, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main(executable, edit_cache_tool=None):
    rng = np.random.default_rng(44)
    shapes = {"dit.in_proj.weight": (8, 4), "dit.out_proj.weight": (4, 8),
              "dit.time_embed.0.weight": (8, 8), "dit.time_embed.2.weight": (8, 8),
              "dit.cond_proj.weight": (8, 6), "dit.cond_proj.bias": (8,),
              "dit.rope_freq_factors": (2,), "dit.skip.0.weight": (8, 16),
              "dit.out_norm.weight": (8,), "dit.out_norm.bias": (8,),
              "dit.latent_mean": (4,), "dit.latent_std": (4,)}
    for i in range(4):
        prefix = f"dit.{i}."
        for norm in ("norm1", "norm_cross", "norm2"):
            shapes[prefix + norm + ".weight"] = (8,)
            shapes[prefix + norm + ".bias"] = (8,)
        for stem, shape in (("self.qkv", (24, 8)), ("self.out", (8, 8)),
                            ("cross.q", (8, 8)), ("cross.kv", (16, 8)),
                            ("cross.out", (8, 8)), ("ff.0", (16, 8)), ("ff.2", (8, 16))):
            shapes[prefix + stem + ".weight"] = shape
    tensors = {name: rng.normal(0, 0.15, shape).astype(np.float32) for name, shape in shapes.items()}
    for name, values in tensors.items():
        if "norm" in name:
            values.fill(0 if name.endswith("bias") else 1)
    tensors["dit.rope_freq_factors"].fill(1)
    tensors["dit.latent_mean"].fill(0)
    tensors["dit.latent_std"].fill(1)
    config = {"general.name": "Tiny CLI test", "ac.architecture": "audiocraft", "ac.format_version": 1,
              "ac.dit.family": "melodyflow", "ac.dit.dim": 8, "ac.dit.layers": 4,
              "ac.dit.heads": 2, "ac.dit.head_dim": 4, "ac.dit.ff_dim": 16,
              "ac.dit.latent_dim": 4, "ac.dit.cond_dim": 6, "ac.dit.time_dim": 8,
              "ac.dit.n_skip": 1, "ac.dit.rope_base": 10000.0, "ac.dit.norm_eps": 1e-5,
              "ac.dit.cfg_coef": 4.0, "ac.dit.max_duration": 30.0, "ac.dit.add_zero_attn": True}
    with tempfile.TemporaryDirectory(prefix="ac-mf-train-cli-") as directory:
        root = Path(directory)
        write(root / "base.gguf", "audiocraft", tensors, config)
        for split in ("train", "validation"):
            (root / split).mkdir()
            for i in range(2):
                cache = {"mean": rng.normal(0, 0.2, (4, 3)), "stddev": np.full((4, 3), 0.1),
                         "hidden": rng.normal(0, 0.1, (2, 6)), "normalization_mean": np.zeros(4),
                         "normalization_std": np.full(4, np.float32(1) + np.float32(1e-5))}
                write(root / split / f"sample-{i}.gguf", "ac-melodyflow-train-cache", cache,
                      {"ac.cache.version": 1, "ac.cache.empty_prompt": False,
                       "ac.cache.source_id": f"{split}-{i}"})
        common = [executable, "--dit", str(root / "base.gguf"), "--cache", str(root / "train"),
                  "--validation", str(root / "validation"), "--device", os.environ.get("AC_TRAIN_TEST_DEVICE","cpu"), "--frames", "3",
                  "--rank", "2", "--first-layer", "0", "--steps", "4", "--seed", "77"]
        plain = subprocess.run(common + ["--out", str(root / "plain.gguf")], capture_output=True, text=True)
        assert plain.returncode == 0, plain.stderr
        checkpointed = subprocess.run(common + ["--out", str(root / "checkpointed.gguf"),
            "--eval-samples", "0", "--eval-draws", "3", "--checkpoint-every", "2"], capture_output=True, text=True)
        assert checkpointed.returncode == 0, checkpointed.stderr
        assert (root / "checkpointed.step-2.gguf").is_file()
        assert "fixed probe step 2" in checkpointed.stderr
        assert (root / "plain.gguf").read_bytes() == (root / "checkpointed.gguf").read_bytes(), \
            "evaluation/checkpointing altered final trained weights"
        # Fresh crops, accumulation, AdamW, warmup and FFN targets must survive exact resume.
        controls = ['--accumulation-steps','3','--warmup-steps','3','--lr-schedule','inverse',
                    '--weight-decay','0.01','--beta2','0.95','--random-crop','--train-ffn']
        cropped = common + ['--frames','2'] + controls
        for name, extra in [('controls', ['--checkpoint-every','2']),
                            ('resumed', ['--checkpoint-every','0','--resume',str(root/'controls.step-2.gguf.state')])]:
            run = subprocess.run(cropped + ['--out',str(root/(name+'.gguf'))] + extra, capture_output=True,text=True)
            assert run.returncode == 0, run.stderr
        assert (root/'controls.gguf').read_bytes() == (root/'resumed.gguf').read_bytes(), 'resume changed weights'
        assert (root/'controls.gguf.state').read_bytes() == (root/'resumed.gguf.state').read_bytes()
        a = gguf.GGUFReader(str(root/'controls.gguf'))
        assert any('.ff.0.' in t.name for t in a.tensors), 'FFN targets missing'
        del a
        changed = subprocess.run(cropped + ['--out',str(root/'bad-resume.gguf'),'--resume',str(root/'controls.step-2.gguf.state'),'--lr','0.0002'],capture_output=True,text=True)
        assert changed.returncode != 0 and not (root/'bad-resume.gguf').exists()
        for flag,value in [('--accumulation-steps','0'),('--warmup-steps','-1'),('--beta2','1'),('--weight-decay','nan'),('--lr-schedule','bogus')]:
            bad = subprocess.run(common+['--out',str(root/'bad-controls.gguf'),flag,value],capture_output=True,text=True)
            assert bad.returncode != 0 and not (root/'bad-controls.gguf').exists()
        logit = subprocess.run(common + ["--out", str(root / "logit.gguf"), "--t-sampling", "logit-normal"],
                               capture_output=True, text=True)
        logit_checked = subprocess.run(common + ["--out", str(root / "logit-checked.gguf"),
            "--t-sampling", "logit-normal", "--checkpoint-every", "2"], capture_output=True, text=True)
        assert logit.returncode == 0 and logit_checked.returncode == 0, logit.stderr + logit_checked.stderr
        assert (root / "logit.gguf").read_bytes() == (root / "logit-checked.gguf").read_bytes(), \
            "checkpointing changed logit-normal training"
        assert (root / "logit.gguf").read_bytes() != (root / "plain.gguf").read_bytes(), \
            "logit-normal option did not change training"
        def initial_probe(output):
            return next(line for line in output.splitlines() if 'probe before' in line)
        assert initial_probe(logit.stderr) == initial_probe(plain.stderr), "sampling option changed validation probe"
        dora = subprocess.run(common + ['--out', str(root/'dora.gguf'), '--adapter-type', 'dora-rows',
            '--t-sampling', 'logit-normal'], capture_output=True, text=True)
        dora_checked = subprocess.run(common + ['--out', str(root/'dora-checked.gguf'), '--adapter-type', 'dora-rows',
            '--t-sampling', 'logit-normal', '--checkpoint-every', '2'], capture_output=True, text=True)
        assert dora.returncode == 0 and dora_checked.returncode == 0, dora.stderr+dora_checked.stderr
        assert (root/'dora.gguf').read_bytes() == (root/'dora-checked.gguf').read_bytes()
        invalid_type = subprocess.run(common + ['--out', str(root/'invalid-type.gguf'), '--adapter-type', 'dora-cols'],
                                      capture_output=True, text=True)
        assert invalid_type.returncode != 0 and not (root/'invalid-type.gguf').exists()
        for extra in (["--t-sampling", "unknown"], ["--t-std", "0"], ["--t-mean", "nan"]):
            invalid = subprocess.run(common + ["--out", str(root / "invalid-time.gguf")] + extra,
                                     capture_output=True, text=True)
            assert invalid.returncode != 0 and not (root / "invalid-time.gguf").exists()
        bad = subprocess.run(common + ["--out", str(root / "bad.gguf"), "--eval-draws", "0"], capture_output=True, text=True)
        assert bad.returncode != 0 and not (root / "bad.gguf").exists()
        if edit_cache_tool:
            prepare = subprocess.run([edit_cache_tool, "--mode", "prepare", "--dit", str(root / "base.gguf"),
                "--cache", str(root), "--out", str(root / "paired"), "--frames", "3", "--device", "cpu"],
                capture_output=True, text=True)
            assert prepare.returncode == 0, prepare.stderr
            for split in ('train', 'validation'):
                for i in range(2):
                    control = gguf.GGUFReader(str(root / 'paired/control' / split / f'sample-{i}.gguf'))
                    bridge = gguf.GGUFReader(str(root / 'paired/bridge' / split / f'sample-{i}.gguf'))
                    ct, bt = ({t.name: t.data.copy() for t in reader.tensors} for reader in (control, bridge))
                    assert np.array_equal(ct['mean'], bt['mean']) and not ct['stddev'].any() and not bt['stddev'].any()
                    assert np.isfinite(bt['edit_pivot']).all() and not np.array_equal(bt['edit_pivot'], bt['mean'])
                    del control, bridge
            paired = common.copy()
            paired[paired.index('--cache') + 1] = str(root / 'paired/bridge/train')
            paired[paired.index('--validation') + 1] = str(root / 'paired/bridge/validation')
            paired += ['--objective', 'edit-bridge', '--t-min', '0.11', '--t-sampling', 'logit-normal']
            for name, extra in [('bridge', []), ('bridge-checked', ['--checkpoint-every', '2'])]:
                result = subprocess.run(paired + ['--out', str(root / (name + '.gguf'))] + extra, capture_output=True, text=True)
                assert result.returncode == 0, result.stderr
            assert (root / 'bridge.gguf').read_bytes() == (root / 'bridge-checked.gguf').read_bytes()
            wrong = subprocess.run(common + ['--objective', 'edit-bridge', '--out', str(root / 'wrong.gguf')],
                                   capture_output=True, text=True)
            assert wrong.returncode != 0 and not (root / 'wrong.gguf').exists()
            mismatch = subprocess.run(paired + ['--t-min', '0.12', '--out', str(root / 'mismatch.gguf')],
                                      capture_output=True, text=True)
            assert mismatch.returncode != 0 and not (root / 'mismatch.gguf').exists()
            rendered = subprocess.run([edit_cache_tool, '--mode', 'render', '--dit', str(root / 'base.gguf'),
                '--cache', str(root / 'paired/bridge/validation/sample-0.gguf'), '--out', str(root / 'reconstruction.f32'),
                '--frames', '3', '--device', 'cpu', '--lora', str(root / 'controls.gguf')], capture_output=True, text=True)
            assert rendered.returncode == 0, rendered.stderr
            assert np.isfinite(np.fromfile(root / 'reconstruction.f32', dtype='<f4')).all()
            # Two windows per source, balanced across two pivot times. Both objectives
            # must sample identical physical time intervals from paired metadata.
            for split in ('train', 'validation'):
                for i in range(2):
                    (root / split / f'window-{i}.gguf').write_bytes((root / split / f'sample-{i}.gguf').read_bytes())
            multi = subprocess.run([edit_cache_tool, '--mode', 'prepare', '--dit', str(root/'base.gguf'),
                '--cache', str(root), '--out', str(root/'multi'), '--frames', '3', '--device', 'cpu',
                '--windows-per-source', '2', '--flowsteps', '0.03,0.11'], capture_output=True, text=True)
            assert multi.returncode == 0, multi.stderr
            for split in ('train', 'validation'):
                paths = sorted((root/'multi/bridge'/split).glob('*.gguf'))
                assert len(paths) == 4
                for path in paths:
                    b = gguf.GGUFReader(str(path))
                    c = gguf.GGUFReader(str(root/'multi/control'/split/path.name))
                    assert np.array_equal(b.fields['ac.cache.time_min'].parts[-1], c.fields['ac.cache.time_min'].parts[-1])
                    assert np.array_equal(b.fields['ac.cache.time_min'].parts[-1], b.fields['ac.cache.edit_flowstep'].parts[-1])
                    del b, c
            for objective, folder in [('flow-matching', 'control'), ('edit-bridge', 'bridge')]:
                command = common.copy()
                command[command.index('--cache')+1] = str(root/'multi'/folder/'train')
                command[command.index('--validation')+1] = str(root/'multi'/folder/'validation')
                trained = subprocess.run(command + ['--objective', objective, '--out', str(root/(folder+'-multi.gguf'))],
                                         capture_output=True, text=True)
                assert trained.returncode == 0, trained.stderr
                invalid = subprocess.run(command + ['--objective', objective, '--t-min', '0.08',
                    '--out', str(root/(folder+'-invalid.gguf'))], capture_output=True, text=True)
                assert invalid.returncode != 0, 'mixed pivot metadata accepted conflicting global time'
    print("PASS: checkpoint roundtrip, training time densities, fixed evaluation and invalid settings")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
