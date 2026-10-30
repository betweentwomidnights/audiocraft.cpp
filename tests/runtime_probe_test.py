import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

version = sys.argv[1]
with tempfile.TemporaryDirectory(prefix='ac-runtime-probe-') as scratch:
    env = os.environ.copy()
    env.update(AC_PORT='invalid', AC_MODELS_DIR=str(Path(scratch)/'missing'))
    for service, binary in zip(['musicgen','melodyflow'], sys.argv[2:]):
        def run(flag):
            return subprocess.run([binary,flag],cwd=scratch,env=env,capture_output=True,text=True,timeout=30)
        r = run('--version')
        assert r.returncode == 0 and r.stdout.strip() == version, r
        r = run('--props')
        assert r.returncode == 0, r.stderr
        props = json.loads(r.stdout)
        assert props['service'] == service and props['runtime'] == 'audiocraft'
        assert props['version'] == version and props['success'] is True
        assert any(d['backend'].upper().startswith('CPU') for d in props['devices'])
        assert run('--unknown-runtime-option').returncode != 0
    assert not list(Path(scratch).iterdir()), 'Model-free probe wrote files'
print('PASS offline runtime versions, JSON devices and no model/port requirements')
