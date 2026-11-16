"""``python -m thaiasr --selftest``: prove the pipeline works on real audio.

Checks, using the example clips in ``examples/``:

1. every clip produces Thai text;
2. decoding is deterministic (same text twice);
3. a recording glued together from two clips is cut at its pause and still
   recognised (the model drops words when a long stream is decoded in one go,
   which is why the app splits at pauses).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .asr import OnnxAsr
from .audio import load_wav
from .config import Config
from .features import SAMPLE_RATE

__all__ = ["run_selftest"]

_THAI = ("\u0e00", "\u0e7f")


def _is_thai(text: str) -> bool:
    return any(_THAI[0] <= ch <= _THAI[1] for ch in text)


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    common = sum(1 for x, y in zip(a, b) if x == y)
    return common / max(len(a), len(b))


def run_selftest(cfg: Config, *, examples_dir: str | Path | None = None, verbose: bool = True) -> int:
    folder = Path(examples_dir) if examples_dir else cfg.root / "examples"
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        print(f"no example wav files in {folder}")
        print("run:  python tools/fetch_examples.py")
        return 2

    asr = OnnxAsr(
        cfg.model_dir(),
        precision=str(cfg["precision"]),
        device=str(cfg["device"]),
        num_threads=int(cfg["num_threads"]) or None,
    )
    if verbose:
        print(f"model: {asr.precision} on {asr.provider} ({asr.num_threads} threads)")
        print(f"warm-up: {asr.warmup():.2f}s")

    failures: list[str] = []
    results: list[tuple[Path, str]] = []
    for wav in wavs:
        audio = load_wav(wav)
        res = asr.transcribe(audio, lang=str(cfg["lang"]), mode=str(cfg["mode"]))
        results.append((wav, res.text))
        mark = "ok " if res.text and _is_thai(res.text) else "BAD"
        if mark == "BAD":
            failures.append(f"{wav.name}: expected Thai text, got {res.text!r}")
        if verbose:
            print(
                f"  [{mark}] {wav.name} {res.duration_s:5.2f}s "
                f"enc={res.encode_s:.2f}s dec={res.decode_s:.2f}s rtf={res.rtf:.2f} bias={res.bias:+.1f}"
            )
            print(f"        {res.text}")

    first = load_wav(wavs[0])
    again = asr.transcribe(first, lang=str(cfg["lang"]), mode=str(cfg["mode"]))
    if again.text != results[0][1]:
        failures.append("decoding is not deterministic")
    elif verbose:
        print("  [ok ] deterministic decoding")

    if len(results) >= 2:
        a, b = load_wav(wavs[0]), load_wav(wavs[1])
        gap = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
        glued = np.concatenate([a, gap, b])
        text, segments = asr.transcribe_long(glued, lang=str(cfg["lang"]), mode=str(cfg["mode"]))
        expected = results[0][1] + results[1][1]
        score = _similarity(text.replace(" ", ""), expected.replace(" ", ""))
        if len(segments) < 2:
            failures.append(f"pause splitting produced {len(segments)} segment(s) for 2 utterances")
        if score < 0.8:
            failures.append(f"glued-recording text only {score:.0%} similar to the individual clips")
        if verbose:
            print(
                f"  [{'ok ' if score >= 0.8 and len(segments) >= 2 else 'BAD'}] "
                f"pause splitting: {len(segments)} segments, {score:.0%} similarity"
            )
            print(f"        {text}")

    if failures:
        print("\nSELFTEST FAILED:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nSELFTEST PASSED")
    return 0
