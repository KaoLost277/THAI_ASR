"""Where the recognised text goes: clipboard, paste, typing, history file."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import pyperclip
from pynput.keyboard import Controller, Key

__all__ = ["TextOutput", "HistoryWriter", "foreground_window_info", "clipboard_text"]

_KEY_HOLD_S = 0.03
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def foreground_window_info() -> str:
    """``"chrome.exe — (4) Instagram"`` for the window that has focus right now.

    Used in the history log: it tells you which window a paste was aimed at,
    which is the first thing to check when text "does not appear".
    """
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return "<none>"
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(max(1, length + 1))
        user32.GetWindowTextW(hwnd, buf, len(buf))
        title = buf.value or "<untitled>"

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        exe = "<unknown>"
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if handle:
            try:
                size = wintypes.DWORD(1024)
                path = ctypes.create_unicode_buffer(size.value)
                if kernel32.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                    exe = path.value.rsplit("\\", 1)[-1]
            finally:
                kernel32.CloseHandle(handle)
        return f"{exe} — {title}"
    except Exception:  # noqa: BLE001 - diagnostics must never break a paste
        return "<unavailable>"


def clipboard_text() -> str | None:
    """Current clipboard text, or ``None`` when it holds something else."""
    try:
        value = pyperclip.paste()
    except Exception:  # noqa: BLE001 - clipboard may hold a non-text format
        return None
    return value if isinstance(value, str) and value else None


class TextOutput:
    """Delivers text to the focused window and/or the clipboard.

    ``paste`` (default) copies the text and sends Ctrl+V into the focused window.
    The text is **left on the clipboard**, so if the target app ignores the
    synthetic keystroke the user can still press Ctrl+V by hand -- set
    ``restore_clipboard = true`` if you would rather have the old clipboard back
    (that also risks racing an app that reads the clipboard asynchronously, and
    it wipes the only copy of the text if the paste did not land).

    ``type`` sends Unicode keystrokes instead (slower, works in more apps but not
    all of them). ``clipboard`` only copies, ``none`` only shows it on screen.
    """

    def __init__(
        self,
        mode: str = "paste",
        *,
        restore_clipboard: bool = False,
        paste_delay_ms: int = 500,
        clipboard_settle_ms: int = 60,
    ):
        self.mode = mode
        self.restore_clipboard = restore_clipboard
        self.paste_delay_ms = paste_delay_ms
        self.clipboard_settle_ms = clipboard_settle_ms
        self._keyboard = Controller()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ pieces
    def _ctrl_v(self) -> None:
        # Give the clipboard update a moment to reach other processes, then hold
        # the keys long enough that slow apps do not miss the chord.
        time.sleep(max(0, self.clipboard_settle_ms) / 1000.0)
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("v")
            time.sleep(_KEY_HOLD_S)
            self._keyboard.release("v")

    def _type(self, text: str) -> None:
        self._keyboard.type(text)

    # ------------------------------------------------------------------- send
    def send(self, text: str) -> str:
        """Deliver ``text``; returns a short description of what was done."""
        if not text:
            return "empty"
        target = foreground_window_info()
        if self.mode == "none":
            return "shown"
        with self._lock:
            previous = (
                clipboard_text()
                if (self.restore_clipboard and self.mode == "paste")
                else None
            )
            try:
                pyperclip.copy(text)
            except Exception as exc:  # noqa: BLE001
                return f"clipboard-failed: {exc}"
            if self.mode == "clipboard":
                return f"copied [target: {target}]"
            if self.mode == "type":
                try:
                    self._type(text)
                except Exception as exc:  # noqa: BLE001
                    return f"type-failed: {exc} (text is on the clipboard)"
                return f"typed [target: {target}]"
            # mode == "paste"
            try:
                self._ctrl_v()
            except Exception as exc:  # noqa: BLE001
                try:  # one fallback: Unicode keystrokes
                    self._type(text)
                except Exception:  # noqa: BLE001
                    return f"paste-failed: {exc} (text is on the clipboard)"
                return f"typed-after-paste-failed [target: {target}]"
            if previous is not None and previous != text:
                time.sleep(max(0, self.paste_delay_ms) / 1000.0)
                try:
                    pyperclip.copy(previous)
                except Exception:  # noqa: BLE001 - restoring is best effort
                    pass
                return f"pasted+clipboard-restored [target: {target}]"
            return f"pasted [target: {target}]"


class HistoryWriter:
    """Append-only JSONL log of everything that was recognised."""

    def __init__(self, path: str | Path | None, enabled: bool = True):
        self.path = Path(path) if path else None
        self.enabled = enabled and self.path is not None
        self._lock = threading.Lock()

    def append(self, text: str, **fields) -> None:
        if not self.enabled or not text:
            return
        record = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "text": text,
            **{k: v for k, v in fields.items() if v is not None},
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
