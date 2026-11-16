"""Energy endpointer: the pause detector the reference server uses.

The rule (``chinda_realtime.Endpointer`` in the model repo): a 20 ms frame counts
as speech when it is louder than *both* the loudest recent level minus 30 dB and
the quietest recent level plus 8 dB.  Stretches with no speech are skipped.

Here it is used for two things:

* cutting a long recording at pauses before decoding it piece by piece, because
  this model drops words after a long quiet pause when a whole recording is fed
  to it as one stream;
* trimming the leading/trailing silence of a push-to-talk utterance.
"""

from __future__ import annotations

import numpy as np

from .features import SAMPLE_RATE

__all__ = ["frame_levels", "speech_mask", "split_at_pauses", "trim_silence"]

FRAME_MS = 20
HOP_MS = 10
DROP_DB = 30.0
RISE_DB = 8.0
FLOOR_DB = -100.0


def frame_levels(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    frame_ms: int = FRAME_MS,
    hop_ms: int = HOP_MS,
) -> np.ndarray:
    """Frame RMS levels in dBFS, one value per hop."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame = max(1, int(sample_rate * frame_ms / 1000))
    hop = max(1, int(sample_rate * hop_ms / 1000))
    if x.size < frame:
        if x.size == 0:
            return np.empty(0, dtype=np.float32)
        return np.array([20.0 * np.log10(float(np.sqrt(np.mean(x**2)) + 1e-12))], dtype=np.float32)
    n = 1 + (x.size - frame) // hop
    idx = np.arange(n) * hop
    # mean of squares per frame without materialising a full window matrix
    sq = x.astype(np.float64) ** 2
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    power = (csum[idx + frame] - csum[idx]) / frame
    return (10.0 * np.log10(power + 1e-12)).astype(np.float32)


def _rolling(values: np.ndarray, window: int, reduce_fn, block: int = 16384) -> np.ndarray:
    """Rolling ``reduce_fn`` over ``window`` centred samples, chunked for memory."""
    from numpy.lib.stride_tricks import sliding_window_view

    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return v.copy()
    pad = window // 2
    vp = np.pad(v, (pad, window - 1 - pad), mode="edge")
    out = np.empty(v.size, dtype=np.float32)
    for start in range(0, v.size, block):
        stop = min(start + block, v.size)
        view = sliding_window_view(vp[start : stop + window - 1], window)
        out[start:stop] = reduce_fn(view, axis=1)
    return out


def speech_mask(
    levels: np.ndarray,
    *,
    window_s: float = 2.0,
    hop_ms: int = HOP_MS,
    drop_db: float = DROP_DB,
    rise_db: float = RISE_DB,
) -> np.ndarray:
    """Boolean per-frame speech decision from dB levels."""
    if levels.size == 0:
        return np.zeros(0, dtype=bool)
    window = max(1, int(window_s * 1000 / hop_ms))
    loudest = _rolling(levels, window, np.max)
    quietest = _rolling(levels, window, np.min)
    return (levels > loudest - drop_db) & (levels > quietest + rise_db)


def _runs(mask: np.ndarray, value: bool = True) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` runs where ``mask == value``."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    target = m if value else ~m
    d = np.diff(np.concatenate(([False], target, [False])).view(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    pad_s: float = 0.15,
    window_s: float = 2.0,
    frame_ms: int = FRAME_MS,
    hop_ms: int = HOP_MS,
    drop_db: float = DROP_DB,
    rise_db: float = RISE_DB,
) -> np.ndarray:
    """Drop leading and trailing non-speech, keeping ``pad_s`` of margin."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    levels = frame_levels(x, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
    mask = speech_mask(levels, window_s=window_s, hop_ms=hop_ms, drop_db=drop_db, rise_db=rise_db)
    if not mask.any():
        return x
    hop = max(1, int(sample_rate * hop_ms / 1000))
    pad = int(pad_s * sample_rate)
    first = int(np.flatnonzero(mask)[0]) * hop
    last = int(np.flatnonzero(mask)[-1]) * hop + int(sample_rate * frame_ms / 1000)
    return x[max(0, first - pad) : min(x.size, last + pad)]


def split_at_pauses(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    min_silence_s: float = 0.7,
    min_speech_s: float = 0.25,
    pad_s: float = 0.15,
    max_piece_s: float = 30.0,
    trim: bool = True,
    window_s: float = 2.0,
) -> list[tuple[int, int]]:
    """Cut ``audio`` into ``[start, end)`` sample ranges at long-enough pauses.

    Speech runs separated by less than ``min_silence_s`` stay in the same piece;
    each piece gets ``pad_s`` of margin so no word is clipped.  Pieces longer
    than ``max_piece_s`` are cut again at their quietest frame, so a recording
    with a single long turn still reaches the encoder in chunks.  With
    ``trim=False`` even very short blips are kept as their own piece.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return []
    hop = max(1, int(sample_rate * HOP_MS / 1000))
    frame_len = int(sample_rate * FRAME_MS / 1000)
    levels = frame_levels(x, sample_rate)
    mask = speech_mask(levels, window_s=window_s)
    min_gap = int(min_silence_s * sample_rate)
    min_piece = int(min_speech_s * sample_rate)
    pad = int(pad_s * sample_rate)

    if mask.any():
        merged: list[list[int]] = []
        for start, end in _runs(mask, True):
            s = max(0, start * hop)
            e = min(x.size, end * hop + frame_len)
            if merged and s - merged[-1][1] < min_gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        pieces = [(max(0, s - pad), min(x.size, e + pad)) for s, e in merged]
        if trim:
            pieces = [p for p in pieces if p[1] - p[0] >= min_piece]
    else:
        # No speech detected (e.g. a constant-level recording): still hand the
        # whole thing over so the length cap below can cut it at quiet points.
        pieces = [(0, x.size)]

    # Force-split anything still long, at its quietest frame.
    max_samples = int(max_piece_s * sample_rate)
    out: list[tuple[int, int]] = []
    for start, end in pieces:
        while end - start > max_samples:
            hi = start + max_samples
            lo = start + int(max_piece_s * 0.5 * sample_rate)
            f_lo = max(0, lo // hop)
            f_hi = min(int(levels.size), max(f_lo + 1, hi // hop))
            cut = hi
            if f_hi > f_lo:
                cut = start + (f_lo + int(np.argmin(levels[f_lo:f_hi]))) * hop
                cut = min(max(cut, lo), hi)  # stay inside the window we searched
            if cut <= start:
                cut = hi
            out.append((start, cut))
            start = cut
        out.append((start, end))
    return out or [(0, x.size)]
