"""Verify checksums, flat archive layouts and startup after extraction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('version')
    p.add_argument('--platform', choices=['windows-x64', 'macos-universal'], required=True)
    p.add_argument('--cpu-only', action='store_true')
    a = p.parse_args()
    directory = a.directory.resolve()
    windows = a.platform == 'windows-x64'
    prefix = f'audiocraft-{a.version}-{a.platform}'
    sums = (directory/('SHA256SUMS' if windows else 'SHA256SUMS-macos')).read_bytes()
    assert not sums.startswith(b'\xef\xbb\xbf') and b'\r' not in sums
    entries = {}
    for line in sums.decode('ascii').splitlines():
        digest, name = line.split('  ', 1)
        assert Path(name).name == name and name not in entries
        entries[name] = digest
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest, name
    expected = {prefix+'-core.zip'} if windows else {prefix+'.zip'}
    if windows and not a.cpu_only:
        expected |= {prefix+'-'+part+'.zip' for part in ['cuda','vulkan','standalone']}
    assert expected <= set(entries)
    assert set(entries) == {f.name for f in directory.glob('*.zip')}
    checker = Path(__file__).with_name('check-runtime-package.py')
    with tempfile.TemporaryDirectory(prefix='audiocraft-extracted-') as temp:
        for name in entries:
            with zipfile.ZipFile(directory/name) as z:
                names = z.namelist()
                assert names and len(names) == len(set(names))
                assert all('/' not in n and '\\' not in n and ':' not in n and n not in ['.','..'] for n in names), name
                if windows and name == prefix+'-cuda.zip':
                    assert names == ['ggml-cuda.dll']
                elif windows and name == prefix+'-vulkan.zip':
                    assert names == ['ggml-vulkan.dll']
                elif name in [prefix+'-core.zip', prefix+'.zip', prefix+'-standalone.zip']:
                    assert 'BUILD-INFO.json' in names
                    standalone = name == prefix+'-standalone.zip'
                    if standalone:
                        assert {'musicgen-server.exe','melodyflow-server.exe','ggml-cuda.dll',
                                'ggml-vulkan.dll','cudart64_12.dll','cublas64_12.dll',
                                'cublasLt64_12.dll','NVIDIA-CUDA-EULA.txt','README.txt'} <= set(names)
                    else:
                        assert not any(n in names for n in ['ggml-cuda.dll','ggml-vulkan.dll'])
                    stage = Path(temp)/('standalone' if standalone else 'runtime');stage.mkdir()
                    z.extractall(stage)
                    if not windows:
                        for info in z.infolist():
                            mode = (info.external_attr >> 16) & 0o777
                            if mode: (stage/info.filename).chmod(mode)
                    for arch in ([None] if windows else ['arm64','x86_64']):
                        cmd = [sys.executable,str(checker),str(stage),a.version]
                        if arch: cmd += ['--arch',arch]
                        if standalone: cmd += ['--standalone']
                        subprocess.run(cmd,check=True)
                    if not windows:
                        for file in stage.iterdir():
                            if file.suffix == '.dylib' or file.name.endswith('-server'):
                                subprocess.run(['codesign','--verify','--strict',str(file)],check=True)
    print('PASS final archives, checksums, flat layout and extracted runtime startup')


if __name__ == '__main__':
    main()
