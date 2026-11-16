"""Fetch the example Thai clips used by ``--selftest``.

    python tools/fetch_examples.py

They come from the model authors' Hugging Face Space and are only for testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

SPACE = "hugging-apps/openthai2-asr-streaming"
FILES = [
    "examples/10021525843523202225.wav",
    "examples/10363030037687949251.wav",
    "examples/10464708231492140124.wav",
    "examples/220102323854312396.wav",
]


def main() -> int:
    from huggingface_hub import hf_hub_download

    target = Path(__file__).resolve().parents[1] / "examples"
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = Path(hf_hub_download(SPACE, name, repo_type="space", local_dir=str(target)))
        if path.parent != target:  # hf_hub_download keeps the repo folder structure
            path.replace(target / path.name)
        print(f"{path.name}: {(target / path.name).stat().st_size} bytes")
    print(f"examples in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
