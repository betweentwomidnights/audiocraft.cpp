import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from train_cache_windows import window_starts

for length in [30, 31, 60, 67, 1001]:
    starts = window_starts(length, 30, 4, True)
    assert starts[0] == 0 and starts[-1] + 30 == length
    assert starts == sorted(set(starts))
    assert all(b <= a + 30 for a, b in zip(starts, starts[1:]))
assert window_starts(100, 30, 4) == [0, 23, 47, 70]
assert window_starts(100, 30, 1) == [0]
try:
    window_starts(29, 30, 4)
except ValueError:
    pass
else:
    raise AssertionError('short track accepted')
print('PASS whole-track tiling, bounds, tail coverage, fixed windows')
