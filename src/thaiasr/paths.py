"""Where ThaiASR keeps its files.

Two modes:

* **source checkout** -- everything stays inside the repository, exactly as
  before (``F:\\ASR_Streaming\\models``, ``logs``, ``history.jsonl``).
* **packaged .exe** (``sys.frozen``) -- the program files are read-only-ish
  inside ``dist``, so the model, logs and history move to
  ``%LOCALAPPDATA%\\ThaiASR``.

``THAIASR_HOME`` overrides both, which is also how the tests (and a fresh build)
can point at an existing model folder instead of downloading 2.6 GB again.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "app_home",
    "is_frozen",
    "logs_dir",
    "log_path",
    "models_dir",
    "history_path",
    "config_path",
    "exe_dir",
]


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def exe_dir() -> Path | None:
    """Folder holding the running .exe (``None`` when run from source)."""
    if not is_frozen():
        return None
    return Path(sys.executable).resolve().parent


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def app_home() -> Path:
    """Folder for models, logs, history and the user's ``config.toml``."""
    env = os.environ.get("THAIASR_HOME")
    if env:
        return Path(env).expanduser()
    if is_frozen():
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / ".local" / "share"
        return root / "ThaiASR"
    return _source_root()


def logs_dir() -> Path:
    return app_home() / "logs"


def log_path() -> Path:
    return logs_dir() / "thaiasr.log"


def models_dir() -> Path:
    return app_home() / "models" / "openthai2.0-asr-streaming-onnx"


def history_path() -> Path:
    return app_home() / "history.jsonl"


def config_path() -> Path:
    return app_home() / "config.toml"
