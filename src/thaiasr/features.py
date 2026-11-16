"""NeMo-compatible log-mel features.

Reimplements ``nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor``
(which itself wraps ``FilterbankFeatures``) with numpy only, so that the ONNX
encoder can be fed without PyTorch.

The parameters below were read from the exported model itself
(``model_config.yaml`` inside ``openthai2.0-asr-streaming_bias-1.0.nemo``):

    sample_rate: 16000
    normalize: NA
    window_size: 0.025      -> 400 samples
    window_stride: 0.01     -> 160 samples
    window: hann            -> periodic Hann (torch.hann_window default)
    features: 128
    n_fft: 512
    log: true
    dither: 1.0e-05         -> skipped: amplitude 1e-5, negligible
    pad_to: 0

Defaults not spelled out in the config: ``preemph=0.97``, ``mag_power=2.0``,
``log_zero_guard_value=2**-24`` and the Slaney mel filterbank
(``htk=False, norm='slaney', fmin=0, fmax=sample_rate/2``).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 128
PREEMPH = 0.97
LOG_ZERO_GUARD = 2.0 ** -24
FMIN = 0.0
FMAX = SAMPLE_RATE / 2.0

__all__ = [
    "SAMPLE_RATE",
    "N_FFT",
    "WIN_LENGTH",
    "HOP_LENGTH",
    "N_MELS",
    "PREEMPH",
    "LOG_ZERO_GUARD",
    "hz_to_mel",
    "mel_to_hz",
    "mel_frequencies",
    "mel_filterbank",
    "hann_window",
    "preemphasis",
    "log_mel",
]


def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    """Slaney-style Hz -> mel (librosa default, ``htk=False``)."""
    f = np.asarray(f, dtype=np.float64)
    f_min = 0.0
    f_sp = 200.0 / 3.0
    mels = (f - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if mels.ndim == 0:
        if f >= min_log_hz:
            mels = min_log_mel + np.log(f / min_log_hz) / logstep
        return mels
    hi = f >= min_log_hz
    mels = np.asarray(mels)
    mels[hi] = min_log_mel + np.log(f[hi] / min_log_hz) / logstep
    return mels


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    """Slaney-style mel -> Hz (inverse of :func:`hz_to_mel`)."""
    m = np.asarray(m, dtype=np.float64)
    f_min = 0.0
    f_sp = 200.0 / 3.0
    freqs = f_min + f_sp * m
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if freqs.ndim == 0:
        if m >= min_log_mel:
            freqs = min_log_hz * np.exp(logstep * (m - min_log_mel))
        return freqs
    freqs = np.asarray(freqs)
    hi = m >= min_log_mel
    freqs[hi] = min_log_hz * np.exp(logstep * (m[hi] - min_log_mel))
    return freqs


def mel_frequencies(n_mels: int, fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """``n_mels`` mel band edges in Hz, spaces evenly on the mel scale."""
    min_mel = hz_to_mel(float(fmin))
    max_mel = hz_to_mel(float(fmax))
    return mel_to_hz(np.linspace(min_mel, max_mel, n_mels))


def mel_filterbank(
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    sample_rate: int = SAMPLE_RATE,
    fmin: float = FMIN,
    fmax: float | None = None,
    norm: str | None = "slaney",
) -> np.ndarray:
    """Slaney-normalised triangular mel filterbank, shape ``[n_mels, n_fft//2+1]``."""
    if fmax is None:
        fmax = sample_rate / 2.0
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    mel_f = mel_frequencies(n_mels + 2, fmin, fmax)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)  # [n_mels+2, n_bins]
    weights = np.zeros((n_mels, fft_freqs.shape[0]), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    if norm == "slaney":
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    elif norm is not None:
        raise ValueError(f"unsupported mel norm: {norm!r}")
    return weights


def hann_window(n: int = WIN_LENGTH) -> np.ndarray:
    """Periodic Hann window, matching ``torch.hann_window(n)``."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)


def preemphasis(x: np.ndarray, coeff: float = PREEMPH) -> np.ndarray:
    """NeMo's pre-emphasis: keep sample 0, then ``x[i] - coeff * x[i-1]``."""
    if coeff is None or coeff == 0.0:
        return x
    out = np.empty_like(x)
    out[0] = x[0]
    if x.shape[0] > 1:
        out[1:] = x[1:] - coeff * x[:-1]
    return out


_MEL_CACHE: dict[tuple, np.ndarray] = {}
_WINDOW_CACHE: dict[int, np.ndarray] = {}


def _get_mel(n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
    key = (n_mels, n_fft, sample_rate)
    fb = _MEL_CACHE.get(key)
    if fb is None:
        fb = mel_filterbank(n_mels, n_fft, sample_rate)
        _MEL_CACHE[key] = fb
    return fb


def _get_window(n_fft: int, win_length: int) -> np.ndarray:
    key = (n_fft, win_length)
    w = _WINDOW_CACHE.get(key)
    if w is None:
        win = hann_window(win_length)
        padded = np.zeros(n_fft, dtype=np.float64)
        left = (n_fft - win_length) // 2
        padded[left : left + win_length] = win  # torch.stft centres the window
        _WINDOW_CACHE[key] = padded
        w = padded
    return w


def num_mel_frames(num_samples: int, hop: int = HOP_LENGTH, n_fft: int = N_FFT) -> int:
    """Frame count produced by ``center=True`` STFT (no ``pad_to``).

    ``n_fft//2`` zeros are padded on each side, so the number of frames is
    ``1 + num_samples // hop`` (same as librosa/torch).
    """
    if num_samples <= 0:
        return 0
    return 1 + num_samples // hop


def log_mel(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
    preemph: float = PREEMPH,
    log_zero_guard: float = LOG_ZERO_GUARD,
    block_frames: int = 8192,
) -> np.ndarray:
    """Compute NeMo log-mel features.

    Args:
        audio: mono waveform, float32/float64, already at ``sample_rate``.

    Returns:
        ``float32`` array shaped ``[n_mels, T]`` ready for ``encoder.onnx``
        (which wants ``[1, 128, T]``).
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sample_rate}")
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)

    x = preemphasis(x, preemph)

    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="constant")
    window = _get_window(n_fft, win_length)
    fb = _get_mel(n_mels, n_fft, sample_rate)

    n_frames = num_mel_frames(x.size, hop_length, n_fft)
    n_bins = n_fft // 2 + 1
    out = np.empty((n_mels, n_frames), dtype=np.float32)

    # A strided view costs nothing; each block is materialised only to run the
    # FFT, which bounds peak memory for long recordings.
    frames_all = sliding_window_view(xp, n_fft)[::hop_length]
    for start in range(0, n_frames, block_frames):
        stop = min(start + block_frames, n_frames)
        frames = frames_all[start:stop] * window  # [B, n_fft]
        spec = np.fft.rfft(frames, n=n_fft, axis=1)  # [B, n_bins]
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        mel = power @ fb.T.astype(np.float32)
        out[:, start:stop] = np.log(mel + np.float32(log_zero_guard)).T
    return out
