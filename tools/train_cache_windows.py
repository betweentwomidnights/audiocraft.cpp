"""Deterministic source coverage; split tracks before calling this helper."""
def window_starts(length, window, count, cover_track=False):
    if length < window or window <= 0 or count <= 0:
        raise ValueError('invalid source/window length')
    if cover_track:
        # Last window is anchored at the end; no padding or uncovered tail.
        return sorted(set(range(0, length - window + 1, window)) | {length - window})
    return [round(i * (length - window) / max(1, count - 1)) for i in range(count)]
