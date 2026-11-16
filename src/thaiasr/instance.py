"""Single-instance guard: running ``run.bat`` twice must not start two apps.

A Windows named mutex is the reliable way (it disappears with the process, so a
crash never leaves a stale lock).  On other platforms a lock file is used.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["InstanceGuard", "already_running_message"]

_MUTEX_NAME = "Local\\ThaiASR_SingleInstance"
_ERROR_ALREADY_EXISTS = 183


class InstanceGuard:
    """Context-manager-ish guard; ``acquire()`` returns False if already running."""

    def __init__(self, name: str = _MUTEX_NAME, lock_file: Path | None = None):
        self.name = name
        if lock_file is None:
            from .paths import app_home

            lock_file = app_home() / "thaiasr.lock"
        self.lock_file = Path(lock_file)
        self._handle = None
        self._fh = None

    def acquire(self) -> bool:
        try:
            return self._acquire_windows()
        except Exception:  # noqa: BLE001 - fall back to a lock file
            return self._acquire_lockfile()

    def _acquire_windows(self) -> bool:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateMutexW(None, True, self.name)
        if not handle:
            raise OSError("CreateMutexW failed")
        if kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._handle = handle
        return True

    def _acquire_lockfile(self) -> bool:
        import msvcrt

        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        fh = self.lock_file.open("a+")
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._handle is not None:
            try:
                import ctypes

                ctypes.windll.kernel32.ReleaseMutex(self._handle)
                ctypes.windll.kernel32.CloseHandle(self._handle)
            except Exception:  # noqa: BLE001
                pass
            self._handle = None
        if self._fh is not None:
            try:
                import msvcrt

                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:  # noqa: BLE001
                pass
            self._fh.close()
            self._fh = None


def already_running_message(title: str = "ThaiASR") -> None:
    """Tell the user, even when the app was started without a console.

    The box closes itself after a few seconds (``MessageBoxTimeoutW``) so a
    forgotten second launch can never leave a modal dialog stuck on screen.
    """
    text = (
        "ThaiASR เปิดอยู่แล้ว\n\n"
        "ดูไอคอนที่มุมขวาล่างของจอ (system tray) หรือ overlay มุมล่างกลางจอ\n"
        "ถ้าหาไม่เจอ ให้ปิดโปรแกรมเดิมก่อนด้วย stop.bat แล้วเปิดใหม่"
    )
    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            user32.MessageBoxTimeoutW(None, text, title, 0x40, 0, 5000)
        except AttributeError:  # older Windows: plain modal box
            user32.MessageBoxW(None, text, title, 0x40)
    except Exception:  # noqa: BLE001 - non-Windows or no GUI
        print(text, file=os.sys.stderr)
