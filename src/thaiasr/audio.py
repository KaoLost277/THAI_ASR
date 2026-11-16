"""Audio input: microphone capture and WAV loading, numpy + sounddevice only."""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np

from .features import SAMPLE_RATE

__all__ = ["load_wav", "resample", "MicStream", "AudioError"]


class AudioError(RuntimeError):
    """Raised when audio cannot be captured or decoded."""


# --------------------------------------------------------------------- loading
def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        val = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        val = np.where(val >= 0x800000, val - 0x1000000, val)
        return val.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise AudioError(f"unsupported sample width: {sample_width * 8} bit")


def load_wav(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read a PCM WAV file as mono float32 at ``sample_rate``."""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
    except wave.Error as exc:
        raise AudioError(
            f"cannot read {path.name}: {exc} (only PCM WAV is supported; "
            "convert other formats to 16 kHz mono WAV)"
        ) from exc
    x = _decode_pcm(frames, width)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    if sr != sample_rate:
        x = resample(x, sr, sample_rate)
    return np.ascontiguousarray(x, dtype=np.float32)


# ------------------------------------------------------------------- resampling
def _lowpass(x: np.ndarray, cutoff: float, taps: int = 129) -> np.ndarray:
    """Zero-phase-ish windowed-sinc low-pass; ``cutoff`` is a fraction of Nyquist."""
    if cutoff >= 1.0:
        return x
    n = np.arange(taps) - (taps - 1) / 2.0
    h = cutoff * np.sinc(cutoff * n)
    h *= np.blackman(taps)
    h /= h.sum()
    pad = taps // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, h, mode="valid").astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample a mono float32 signal (anti-aliasing low-pass + interpolation)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if sr_in == sr_out or x.size == 0:
        return x
    ratio = sr_out / sr_in
    y = _lowpass(x, min(1.0, ratio) * 0.9)
    n_out = int(round(x.size * ratio))
    pos = np.arange(n_out, dtype=np.float64) / ratio
    idx = np.arange(x.size, dtype=np.float64)
    return np.interp(pos, idx, y).astype(np.float32)


# ------------------------------------------------------------------ microphone
class MicStream:
    """Push-to-talk microphone capture at 16 kHz mono float32.

    Tries to open the device directly at 16 kHz; if the device refuses, it opens
    at the device default rate and resamples each block.
    """

    def __init__(self, *, device: int | str | None = None, blocksize_ms: int = 100):
        self.device = device
        self.blocksize_ms = blocksize_ms
        self.samplerate = SAMPLE_RATE
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._level = 0.0

    # -- helpers
    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        block = np.asarray(indata, dtype=np.float32).reshape(-1)
        if self.samplerate != SAMPLE_RATE:
            block = resample(block, self.samplerate, SAMPLE_RATE)
        with self._lock:
            self._chunks.append(block.copy())
            self._level = float(np.sqrt(np.mean(block**2))) if block.size else 0.0

    @property
    def level(self) -> float:
        """RMS of the most recent block (0..1-ish)."""
        return self._level

    @property
    def recording(self) -> bool:
        return self._stream is not None

    # -- lifecycle
    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        with self._lock:
            self._chunks = []
            self._level = 0.0
        block = max(160, int(SAMPLE_RATE * self.blocksize_ms / 1000))
        attempts = [(SAMPLE_RATE, block)]
        try:
            info = sd.query_devices(self.device, "input")
            default_sr = int(info["default_samplerate"])
        except Exception:  # device query failure is reported by the open below
            default_sr = SAMPLE_RATE
        if default_sr != SAMPLE_RATE:
            attempts.append((default_sr, max(160, int(default_sr * self.blocksize_ms / 1000))))
        last_error: Exception | None = None
        for sr, blocksize in attempts:
            try:
                stream = sd.InputStream(
                    samplerate=sr,
                    blocksize=blocksize,
                    channels=1,
                    dtype="float32",
                    device=self.device,
                    callback=self._callback,
                )
                stream.start()
                self._stream = stream
                self.samplerate = sr
                return
            except Exception as exc:  # noqa: BLE001 - keep trying other rates
                last_error = exc
        raise AudioError(f"cannot open microphone: {last_error}")

    def stop(self) -> np.ndarray:
        """Stop capture and return everything recorded so far."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001 - closing must not mask the audio
                pass
        with self._lock:
            chunks = self._chunks
            self._chunks = []
            self._level = 0.0
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    def snapshot(self) -> np.ndarray:
        """Copy of the audio captured so far (safe while recording)."""
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._chunks).astype(np.float32)

    def cancel(self) -> None:
        """Stop capture and drop the audio."""
        self.stop()
