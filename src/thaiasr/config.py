"""Configuration: defaults, a TOML file, and command-line overrides."""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

from .model_store import model_ready

__all__ = ["Config", "DEFAULTS", "default_config_path", "project_root", "user_config_dir"]

DEFAULTS: dict[str, Any] = {
    # --- hotkeys (see hotkey.py for the accepted names) ---------------------
    "hotkey": "ctrl_r",
    # hold   = push-to-talk (press and keep holding while you speak)
    # toggle = tap once to start, tap again to stop
    "hotkey_mode": "hold",
    "quit_hotkey": "ctrl+alt+q",
    "cancel_hotkey": "esc",
    # Show/hide the overlay (same as the "— ย่อ" button on the window).
    "overlay_hotkey": "ctrl+alt+h",
    # --- model --------------------------------------------------------------
    "model_dir": "models/openthai2.0-asr-streaming-onnx",
    "precision": "fp32",  # fp32 | int8
    "device": "cpu",  # cpu | cuda (needs onnxruntime-gpu)
    "num_threads": 0,  # 0 = pick automatically (min(8, cpus))
    "preload": True,  # load the model at start-up in the background
    "lang": "th-TH",  # th-TH | th-mix
    "mode": "standard",  # standard | conversation
    # --- recording ----------------------------------------------------------
    "max_utterance_s": 120.0,
    "min_utterance_s": 0.3,
    "trim_silence": True,
    "min_silence_s": 0.7,
    # Utterances longer than this are cut at pauses before decoding (this model
    # drops words after a quiet pause when a long stream is decoded in one go).
    "split_over_s": 12.0,
    # --- live partial text --------------------------------------------------
    "partials": True,
    "partial_interval_ms": 700,
    "partial_max_utterance_s": 15.0,
    # --- output -------------------------------------------------------------
    "output": "paste",  # paste | type | clipboard | none
    # Keep the recognised text on the clipboard (recommended): if the target app
    # ignores the synthetic Ctrl+V, you can still paste it by hand.
    "restore_clipboard": False,
    "paste_delay_ms": 500,
    "save_history": True,
    "history_path": "history.jsonl",
    # --- overlay / tray -----------------------------------------------------
    "show_overlay": True,
    "tray": True,  # icon in the system tray (needs pystray + pillow)
    "single_instance": True,  # a second launch just says "already running"
    "start_minimized": False,  # start with the overlay hidden (tray only)
    "overlay": {
        "width": 560,
        "height": 168,
        "bottom_margin": 90,
        "opacity": 0.94,
        "hide_after_s": 6.0,
    },
}


def project_root() -> Path:
    """Repository root (the directory that holds ``src/``)."""
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return user_config_dir_or_home() / "config.toml"


def user_config_dir_or_home() -> Path:
    """Folder that owns ``config.toml``, models, logs and history."""
    from .paths import app_home

    return app_home()


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _nearby_model_dirs(folder_name: str) -> list[Path]:
    """Model folders worth checking next to a packaged .exe."""
    from .paths import exe_dir

    base = exe_dir()
    if base is None:
        return []
    return [
        base / "models" / folder_name,
        base.parent / "models" / folder_name,
        base.parent.parent / "models" / folder_name,
    ]


class Config:
    """Layered settings with dotted lookup and path resolution."""

    def __init__(self, data: dict[str, Any] | None = None, *, path: Path | None = None):
        self.data = _merge(DEFAULTS, data or {})
        self.path = path
        self.root = path.parent if path else user_config_dir_or_home()

    # -- construction
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> "Config":
        """Load ``path`` (or the default) and apply non-``None`` overrides."""
        cfg_path: Path | None = None
        if path is not None:
            cfg_path = Path(path).expanduser()
            if not cfg_path.exists():
                raise FileNotFoundError(f"config file not found: {cfg_path}")
        else:
            candidate = default_config_path()
            if candidate.exists():
                cfg_path = candidate
        data: dict[str, Any] = {}
        if cfg_path is not None:
            with cfg_path.open("rb") as fh:
                data = tomllib.load(fh)
        cfg = cls(data, path=cfg_path)
        for key, value in (overrides or {}).items():
            if value is not None:
                cfg.data[key] = value
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.data["precision"] not in ("fp32", "int8"):
            raise ValueError("precision must be 'fp32' or 'int8'")
        if self.data["output"] not in ("paste", "type", "clipboard", "none"):
            raise ValueError("output must be paste|type|clipboard|none")
        if self.data["mode"] not in ("standard", "conversation", "meeting"):
            raise ValueError("mode must be standard|conversation|meeting")
        if self.data["hotkey_mode"] not in ("hold", "toggle"):
            raise ValueError("hotkey_mode must be 'hold' or 'toggle'")

    # -- access
    def __getitem__(self, key: str) -> Any:
        node: Any = self.data
        for part in key.split("."):
            node = node[part]
        return node

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def path_for(self, key: str) -> Path:
        """Resolve a path setting relative to the config file (or repo root)."""
        value = Path(str(self[key])).expanduser()
        return value if value.is_absolute() else (self.root / value)

    def model_dir(self) -> Path:
        """Where the ONNX files really are.

        The packaged .exe keeps the model in ``%LOCALAPPDATA%\\ThaiASR``, but if a
        complete model already sits next to the .exe (for example the checkout's
        own ``models`` folder when running from ``dist``), use that instead of
        asking the user to download 2.6 GB again.
        """
        primary = self.path_for("model_dir")
        precision = str(self["precision"])
        if model_ready(primary, precision):
            return primary
        for candidate in _nearby_model_dirs(primary.name):
            if candidate != primary and model_ready(candidate, precision):
                return candidate
        return primary

    def save_setting(self, key: str, value: Any) -> Path:
        """Persist one *top-level* setting, leaving the rest of the file alone.

        Used by the tray menu (changing hold/toggle), so the choice survives a
        restart.  Nested settings (``overlay.*``) are out of scope.
        """
        if "." in key:
            raise ValueError("save_setting only handles top-level keys")
        target = self.path or default_config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
        rendered = f"{key} = {_format_toml_value(value)}"

        first_section = next(
            (i for i, line in enumerate(lines) if line.strip().startswith("[")), len(lines)
        )
        for i in range(first_section):
            stripped = lines[i].strip()
            if stripped.startswith(f"{key} =") or stripped.startswith(f"{key}="):
                lines[i] = rendered
                break
        else:
            lines.insert(first_section, rendered)
        target.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
        self.data[key] = value
        return target

    def describe(self) -> str:
        where = str(self.path) if self.path else "<defaults>"
        return (
            f"config={where}\n"
            f"  model_dir={self.path_for('model_dir')} precision={self['precision']} "
            f"device={self['device']}\n"
            f"  hotkey={self['hotkey']} ({self['hotkey_mode']}) lang={self['lang']} "
            f"output={self['output']} tray={self['tray']}"
        )


def user_config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "ThaiASR"
