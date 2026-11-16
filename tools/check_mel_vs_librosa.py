"""Dev-only check: does ``thaiasr.features`` match librosa/NeMo numerics?

Not part of the app (librosa is deliberately not a runtime dependency).  Run it
in a throwaway venv:

    uv venv .venv-devcheck
    uv pip install --python .venv-devcheck/Scripts/python.exe librosa
    .venv-devcheck/Scripts/python.exe tools/check_mel_vs_librosa.py

It compares the mel filterbank and the whole log-mel pipeline against librosa
using the exact parameters from the model's ``model_config.yaml`` (plus NeMo's
documented defaults: preemph 0.97, power 2.0, log guard 2**-24, Slaney norm).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thaiasr import features as F  # noqa: E402


def librosa_reference(x: np.ndarray) -> np.ndarray:
    import librosa

    y = np.concatenate((x[:1], x[1:] - F.PREEMPH * x[:-1])).astype(np.float32)
    spec = librosa.stft(
        y,
        n_fft=F.N_FFT,
        hop_length=F.HOP_LENGTH,
        win_length=F.WIN_LENGTH,
        window="hann",
        center=True,
        pad_mode="constant",
    )
    power = np.abs(spec) ** 2
    fb = librosa.filters.mel(
        sr=F.SAMPLE_RATE, n_fft=F.N_FFT, n_mels=F.N_MELS,
        fmin=F.FMIN, fmax=F.FMAX, htk=False, norm="slaney",
    )
    mel = fb @ power
    return np.log(mel + F.LOG_ZERO_GUARD).astype(np.float32)


def main() -> int:
    rng = np.random.default_rng(0)
    t = np.arange(int(2.5 * F.SAMPLE_RATE)) / F.SAMPLE_RATE
    x = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)
    wav = Path(__file__).resolve().parents[1] / "examples"
    wavs = sorted(wav.glob("*.wav"))
    if wavs:
        from thaiasr.audio import load_wav

        x = load_wav(wavs[0])
        print(f"using {wavs[0].name} ({x.size / F.SAMPLE_RATE:.2f}s)")

    import librosa

    ours_fb = F.mel_filterbank()
    ref_fb = librosa.filters.mel(
        sr=F.SAMPLE_RATE, n_fft=F.N_FFT, n_mels=F.N_MELS,
        fmin=F.FMIN, fmax=F.FMAX, htk=False, norm="slaney",
    )
    print("filterbank shape ours=%s librosa=%s" % (ours_fb.shape, ref_fb.shape))
    print("filterbank max|diff| = %.3e" % np.abs(ours_fb - ref_fb).max())

    ours = F.log_mel(x)
    ref = librosa_reference(x)
    n = min(ours.shape[1], ref.shape[1])
    print("mel shape ours=%s librosa=%s" % (ours.shape, ref.shape))
    diff = np.abs(ours[:, :n] - ref[:, :n])
    print("mel max|diff| = %.3e   mean|diff| = %.3e" % (diff.max(), diff.mean()))
    ok = np.abs(ours_fb - ref_fb).max() < 1e-7 and diff.max() < 1e-3
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
