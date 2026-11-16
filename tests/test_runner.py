"""Shutdown wiring: the overlay's close button (and Ctrl+Alt+Q) must really exit.

The bug this guards against: stopping the recognition loop left the Tk window on
screen, so the program looked impossible to close.
"""

from __future__ import annotations

import os
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thaiasr import runner  # noqa: E402
from thaiasr.config import Config  # noqa: E402
from thaiasr.ui import Overlay  # noqa: E402


class _FakeApp:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.quit_requested = False

    def request_quit(self) -> None:
        self.quit_requested = True
        if self.fail:
            raise RuntimeError("app shutdown blew up")


class _FakeOverlay:
    def __init__(self):
        self.on_quit = None
        self.closed = False

    def close(self) -> None:
        self.closed = True


class TestWireOverlay(unittest.TestCase):
    def test_close_button_shuts_down_app_and_window(self):
        app, overlay = _FakeApp(), _FakeOverlay()
        handler = runner.wire_overlay(app, overlay)
        self.assertIs(overlay.on_quit, handler)
        overlay.on_quit()
        self.assertTrue(app.quit_requested, "the app must be told to stop")
        self.assertTrue(overlay.closed, "the overlay window must be closed too")

    def test_window_closes_even_if_the_app_raises(self):
        app, overlay = _FakeApp(fail=True), _FakeOverlay()
        runner.wire_overlay(app, overlay)
        with self.assertRaises(RuntimeError):
            overlay.on_quit()
        self.assertTrue(overlay.closed)

    def test_quit_hotkey_uses_the_same_handler(self):
        """Ctrl+Alt+Q must close the window, not just stop the recognition loop."""
        from pynput.keyboard import Key, KeyCode

        app, overlay = _FakeApp(), _FakeOverlay()
        handler = runner.wire_overlay(app, overlay)
        cfg = Config.load(None)
        manager = runner._make_manager(app, cfg, handler)
        manager._on_press(Key.ctrl_l)
        manager._on_press(Key.alt_l)
        manager._on_press(KeyCode.from_char("q"))
        self.assertTrue(overlay.closed, "the quit hotkey left the window open")
        self.assertTrue(app.quit_requested)


class TestOverlayMainloopExits(unittest.TestCase):
    def test_close_ends_the_tk_main_loop(self):
        cfg = Config.load(None)
        overlay = Overlay(cfg)
        overlay._root = overlay._build()
        root = overlay._root

        watchdog = threading.Timer(
            8.0,
            lambda: (print("watchdog: main loop did not exit", file=sys.stderr), os._exit(3)),
        )
        watchdog.daemon = True
        watchdog.start()
        try:
            root.after(40, overlay._drain)
            root.after(150, overlay.close)
            root.mainloop()  # must return once the close request is drained
        finally:
            watchdog.cancel()
        try:
            exists = bool(root.winfo_exists())
        except Exception:  # noqa: BLE001 - a destroyed interpreter
            exists = False
        self.assertFalse(exists, "the overlay window is still alive after close()")


if __name__ == "__main__":
    unittest.main()
