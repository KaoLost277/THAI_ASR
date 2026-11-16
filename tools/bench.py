"""Measure how fast this machine decodes, and how VRAM/RAM-free it stays.

    python tools/bench.py [--precision fp32] [--device cpu] [--repeat 3]

Prints load time, warm-up and, per example clip, the audio duration, the
encoder/decoder split and the real-time factor (lower is faster).  Use it to
decide whether live partial text is worth keeping on (see ``partials``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thaiasr.asr import OnnxAsr  # noqa: E402
from thaiasr.audio import load_wav  # noqa: E402
from thaiasr.config import Config  # noqa: E402
from thaiasr.features import SAMPLE_RATE  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ThaiASR benchmark")
    parser.add_argument("--model-dir")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--lang", default="th-TH", choices=["th-TH", "th-mix"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    cfg = Config.load(None, overrides={"model_dir": args.model_dir} if args.model_dir else None)
    model_dir = cfg.model_dir()

    t0 = time.perf_counter()
    asr = OnnxAsr(model_dir, precision=args.precision, device=args.device)
    load_s = time.perf_counter() - t0
    warm_s = asr.warmup()

    clips = sorted((ROOT / "examples").glob("*.wav"))
    rows = []
    for clip in clips:
        audio = load_wav(clip)
        best = None
        for _ in range(max(1, args.repeat)):
            res = asr.transcribe(audio, lang=args.lang)
            row = {
                "file": clip.name,
                "duration_s": round(res.duration_s, 3),
                "encode_s": round(res.encode_s, 4),
                "decode_s": round(res.decode_s, 4),
                "rtf": round(res.rtf, 4),
                "text": res.text,
            }
            if best is None or row["rtf"] < best["rtf"]:
                best = row
        rows.append(best)

    summary = {
        "precision": asr.precision,
        "provider": asr.provider,
        "threads": asr.num_threads,
        "load_s": round(load_s, 3),
        "warmup_s": round(warm_s, 3),
        "clips": rows,
    }
    realtime = 1.0 / max((r["rtf"] for r in rows), default=float("nan")) if rows else float("nan")
    summary["realtime_factor_x"] = round(realtime, 2)
    summary["note"] = (
        "rtf < 1 means faster than real time; the app re-decodes the whole "
        "utterance for each live partial, so keep partials only when rtf is "
        "comfortably below ~0.5 for your typical utterance length"
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print(f"model: {summary['precision']} on {summary['provider']} ({summary['threads']} threads)")
    print(f"load {load_s:.2f}s, warm-up {warm_s:.2f}s")
    print(f"{'file':<30}{'audio s':>9}{'encode s':>10}{'decode s':>10}{'rtf':>8}")
    for row in rows:
        print(
            f"{row['file']:<30}{row['duration_s']:>9.2f}{row['encode_s']:>10.2f}"
            f"{row['decode_s']:>10.2f}{row['rtf']:>8.2f}"
        )
    print(f"\nabout {realtime:.1f}x faster than real time on this machine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
