"""Check staged/extracted runtime probes from an unrelated, model-free directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('version')
    p.add_argument('--arch', choices=['arm64', 'x86_64'])
    p.add_argument('--standalone', action='store_true')
    args = p.parse_args()
    directory = args.directory.resolve()
    version = args.version.removeprefix('v')
    prefix = ['arch', '-' + args.arch] if args.arch else []
    suffix = '.exe' if os.name == 'nt' else ''
    with tempfile.TemporaryDirectory(prefix='audiocraft-package-') as scratch:
        env = os.environ.copy()
        env.update(AC_MODELS_DIR=str(Path(scratch)/'missing-models'), AC_PORT='invalid')
        # Prevent the build tree/toolkit's PATH from hiding missing packaged DLLs.
        if os.name == 'nt':
            win = Path(os.environ['SystemRoot'])
            env['PATH'] = os.pathsep.join([str(win/'System32'), str(win)])
        for service in ['musicgen', 'melodyflow']:
            exe = directory/(service+'-server'+suffix)
            def run(flag):
                r = subprocess.run(prefix+[str(exe), flag], cwd=scratch, env=env,
                                   capture_output=True, text=True, timeout=60)
                assert r.returncode == 0, (service, flag, r.stderr)
                return r.stdout.strip()
            assert run('--version') == version
            props = json.loads(run('--props'))
            assert props['success'] is True and props['service'] == service
            assert props['runtime'] == 'audiocraft' and props['version'] == version
            assert any(d['backend'].upper().startswith('CPU') for d in props['devices'])
            for device in props['devices']:
                assert isinstance(device['memory_total_bytes'], int)
                assert device['memory_total_bytes'] >= 0 and device['description']
            # All shipped tools must start from the extracted folder.
        for tool in ['mg-generate', 'mg-train', 'mf-edit', 'mf-train', 'mf-edit-cache',
                     'ac-textenc', 'ac-tokenize', 'ac-vae', 'ac-dit', 'ac-encodec']:
            r = subprocess.run(prefix+[str(directory/(tool+suffix)), '--help'], cwd=scratch,
                               env=env, capture_output=True, timeout=30)
            assert r.returncode == 0, (tool, r.stderr.decode(errors='replace'))
        assert not list(Path(scratch).iterdir()), 'Offline probes wrote runtime files'
    info = json.loads((directory/'BUILD-INFO.json').read_bytes())
    assert info['service'] == 'audiocraft' and info['version'] == 'v'+version
    assert info['commit'] and info['ggml_commit']
    if info['platform'] == 'windows-x64':
        assert info['build_flavor'] in ['portable','cpu-smoke','gpu-smoke']
        if not args.standalone:
            assert not (directory/'ggml-cuda.dll').exists(), 'GPU backend leaked into core'
            assert not (directory/'ggml-vulkan.dll').exists(), 'GPU backend leaked into core'
    print('PASS staged MusicGen/MelodyFlow versions, CPU devices, metadata and CLI startup')


if __name__ == '__main__':
    main()
