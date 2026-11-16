"""Tests for the 'easy to use' additions: tray, toggle mode, single instance."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pynput.keyboard import Key, KeyCode

from thaiasr.config import Config
from thaiasr.hotkey import HotkeyManager
from thaiasr.instance import InstanceGuard
from thaiasr.paths import app_home, is_frozen, log_path


class TestPaths(unittest.TestCase):
    def test_source_mode_uses_repo_root(self):
        self.assertFalse(is_frozen())
        self.assertEqual(app_home(), Path(__file__).resolve().parents[1])

    def test_env_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"THAIASR_HOME": tmp}):
                self.assertEqual(app_home(), Path(tmp))
                self.assertEqual(log_path(), Path(tmp) / "logs" / "thaiasr.log")

    def test_config_root_follows_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"THAIASR_HOME": tmp}):
                cfg = Config()
                self.assertEqual(cfg.root, Path(tmp))
                self.assertEqual(cfg.path_for("model_dir"), Path(tmp) / "models" / "openthai2.0-asr-streaming-onnx")


class TestSingleInstance(unittest.TestCase):
    def test_second_guard_is_refused_then_freed(self):
        first = InstanceGuard(name="Local\\ThaiASR_TestGuard")
        second = InstanceGuard(name="Local\\ThaiASR_TestGuard")
        self.assertTrue(first.acquire())
        try:
            self.assertFalse(second.acquire())
        finally:
            first.release()
        again = InstanceGuard(name="Local\\ThaiASR_TestGuard")
        self.assertTrue(again.acquire())
        again.release()

    def test_guard_falls_back_to_lockfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "thaiasr.lock"
            guard = InstanceGuard(lock_file=lock)
            with mock.patch.object(guard, "_acquire_windows", side_effect=OSError("no win32")):
                self.assertTrue(guard.acquire())
                other = InstanceGuard(lock_file=lock)
                with mock.patch.object(other, "_acquire_windows", side_effect=OSError("no win32")):
                    self.assertFalse(other.acquire())
            guard.release()


class TestToggleMode(unittest.TestCase):
    def _manager(self, **kwargs):
        events = []
        manager = HotkeyManager(
            hold="ctrl_r",
            cancel="esc",
            on_hold_start=lambda: events.append("start"),
            on_hold_stop=lambda: events.append("stop"),
            on_cancel=lambda: events.append("cancel"),
            **kwargs,
        )
        return manager, events

    def test_hold_mode_unchanged(self):
        manager, events = self._manager(hold_mode="hold")
        manager._on_press(Key.ctrl_r)
        self.assertEqual(events, ["start"])
        manager._on_release(Key.ctrl_r)
        self.assertEqual(events, ["start", "stop"])

    def test_toggle_starts_and_stops_on_taps(self):
        manager, events = self._manager(hold_mode="toggle")
        manager._on_press(Key.ctrl_r)
        manager._on_release(Key.ctrl_r)  # release is ignored in toggle mode
        self.assertEqual(events, ["start"])
        manager._on_press(Key.ctrl_r)
        manager._on_release(Key.ctrl_r)
        self.assertEqual(events, ["start", "stop"])

    def test_toggle_ignores_auto_repeat(self):
        manager, events = self._manager(hold_mode="toggle")
        manager._on_press(Key.ctrl_r)
        manager._on_press(Key.ctrl_r)  # key repeat while held
        self.assertEqual(events, ["start"])

    def test_sync_active_recovers_after_auto_stop(self):
        manager, events = self._manager(hold_mode="toggle")
        manager._on_press(Key.ctrl_r)
        manager._on_release(Key.ctrl_r)
        manager.sync_active(False)  # the app stopped on its own (Esc / timeout)
        manager._on_press(Key.ctrl_r)
        self.assertEqual(events, ["start", "start"])

    def test_cancel_only_while_recording(self):
        manager, events = self._manager(hold_mode="toggle")
        manager._on_press(Key.esc)
        manager._on_release(Key.esc)
        self.assertEqual(events, [])
        manager._on_press(Key.ctrl_r)
        self.assertEqual(events, ["start"])
        manager._on_press(Key.esc)
        self.assertEqual(events, ["start", "cancel"])

    def test_held_esc_does_not_cancel_the_start(self):
        """Starting while Esc happens to be down must not cancel immediately."""
        manager, events = self._manager(hold_mode="hold")
        manager._on_press(Key.esc)
        manager._on_press(Key.ctrl_r)
        self.assertEqual(events, ["start"])

    def test_other_keys_do_not_retrigger_the_talk_key(self):
        """A keystroke while the talk key is still down must not count as a tap."""
        manager, events = self._manager(hold_mode="toggle")
        manager._on_press(KeyCode.from_char("a"))
        manager._on_press(Key.ctrl_r)
        manager._on_press(KeyCode.from_char("a"))
        manager._on_release(KeyCode.from_char("a"))
        self.assertEqual(events, ["start"])

    def test_bad_mode_rejected(self):
        with self.assertRaises(Exception):
            HotkeyManager(hold="ctrl_r", hold_mode="press")


class TestTrayHelpers(unittest.TestCase):
    def test_dependencies_available(self):
        from thaiasr.tray import tray_available

        self.assertTrue(tray_available(), "pystray + pillow should be installed")

    def test_icon_is_drawn_without_asset_files(self):
        from thaiasr.tray import _icon_image

        idle = _icon_image(False)
        recording = _icon_image(True)
        self.assertEqual(idle.size, (64, 64))
        self.assertNotEqual(idle.tobytes(), recording.tobytes())


class TestAppMessages(unittest.TestCase):
    def test_missing_model_points_at_the_tray(self):
        from thaiasr.app import missing_model_hint, _short_progress

        self.assertIn("tray", missing_model_hint())
        self.assertIn("โมเดล", _short_progress("downloading fp32 model"))
        self.assertEqual(_short_progress("   "), "")

    def test_download_async_guards_against_double_start(self):
        from thaiasr.app import DictationApp

        cfg = Config({"precision": "int8"})
        app = DictationApp(cfg)
        app._downloading = True
        with mock.patch.object(app, "on_status") as status:
            app.download_model_async()
            self.assertTrue(any("อยู่แล้ว" in str(call) for call in status.call_args_list))


class TestConfigPersistence(unittest.TestCase):
    def test_save_setting_keeps_sections_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('hotkey = "f9"\n\n[overlay]\nwidth = 560\n', encoding="utf-8")
            cfg = Config.load(path)
            cfg.save_setting("hotkey_mode", "toggle")
            text = path.read_text(encoding="utf-8")
            self.assertIn('hotkey_mode = "toggle"', text)
            self.assertLess(text.index("hotkey_mode"), text.index("[overlay]"))
            self.assertIn("width = 560", text)

            cfg.save_setting("hotkey_mode", "hold")  # replaces, never duplicates
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("hotkey_mode"), 1)
            self.assertEqual(Config.load(path)["hotkey_mode"], "hold")

    def test_save_setting_formats_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(path=Path(tmp) / "config.toml")
            cfg.save_setting("tray", False)
            cfg.save_setting("num_threads", 4)
            text = (Path(tmp) / "config.toml").read_text(encoding="utf-8")
            self.assertIn("tray = false", text)
            self.assertIn("num_threads = 4", text)

    def test_save_setting_rejects_nested_keys(self):
        cfg = Config()
        with self.assertRaises(ValueError):
            cfg.save_setting("overlay.width", 100)

    def test_validate_hotkey_mode(self):
        with self.assertRaises(ValueError):
            Config({"hotkey_mode": "sometimes"}).validate()
        Config({"hotkey_mode": "toggle"}).validate()


class TestModelDirFallback(unittest.TestCase):
    """The .exe should reuse a model that already sits next to it."""

    def _cfg(self, tmp):
        home = Path(tmp) / "home"
        exe = Path(tmp) / "dist" / "ThaiASR"
        exe.mkdir(parents=True)
        nearby = Path(tmp) / "models" / "openthai2.0-asr-streaming-onnx"
        nearby.mkdir(parents=True)
        env = mock.patch.dict(os.environ, {"THAIASR_HOME": str(home)})
        freeze = mock.patch("thaiasr.paths.exe_dir", return_value=exe)
        return home, nearby, env, freeze

    def test_uses_the_copy_next_to_the_exe(self):
        with tempfile.TemporaryDirectory() as tmp:
            home, nearby, env, freeze = self._cfg(tmp)
            with env, freeze, mock.patch(
                "thaiasr.config.model_ready", side_effect=lambda path, _p: Path(path) == nearby
            ):
                cfg = Config()
                self.assertEqual(cfg.model_dir(), nearby)
                self.assertEqual(
                    cfg.path_for("model_dir"),
                    home / "models" / "openthai2.0-asr-streaming-onnx",
                )

    def test_configured_location_wins_when_it_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            home, _nearby, env, freeze = self._cfg(tmp)
            with env, freeze, mock.patch("thaiasr.config.model_ready", return_value=True):
                cfg = Config()
                self.assertEqual(
                    cfg.model_dir(), home / "models" / "openthai2.0-asr-streaming-onnx"
                )

    def test_reports_the_configured_path_when_nothing_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            home, _nearby, env, freeze = self._cfg(tmp)
            with env, freeze, mock.patch("thaiasr.config.model_ready", return_value=False):
                cfg = Config()
                self.assertEqual(
                    cfg.model_dir(), home / "models" / "openthai2.0-asr-streaming-onnx"
                )


class TestConsoleEncoding(unittest.TestCase):
    def test_thai_text_survives_a_legacy_console(self):
        """The frozen console starts as cp437; Thai must not raise."""
        import io
        import sys

        from thaiasr.__main__ import _configure_console

        raw = io.BytesIO()
        fake = io.TextIOWrapper(raw, encoding="cp437", errors="strict")
        with mock.patch.object(sys, "stdout", fake), mock.patch.object(sys, "stderr", fake):
            _configure_console()
            fake.write("ทดสอบภาษาไทย")
            fake.flush()
        self.assertIn("ทดสอบภาษาไทย", raw.getvalue().decode("utf-8"))

    def test_missing_streams_are_replaced(self):
        import sys

        from thaiasr.__main__ import _ensure_streams

        with mock.patch.object(sys, "stdout", None), mock.patch.object(sys, "stderr", None):
            _ensure_streams()
            self.assertIsNotNone(sys.stdout)
            self.assertIsNotNone(sys.stderr)
            sys.stdout.close()  # the devnull handle


class TestOverlayMinimize(unittest.TestCase):
    def test_minimize_button_prefers_its_callback(self):
        from thaiasr.ui import Overlay

        overlay = Overlay(Config())
        calls = []
        overlay.on_minimize = lambda: calls.append(1)
        overlay._on_minimize()
        self.assertEqual(calls, [1])
        self.assertTrue(overlay.visible, "the callback decides, not the widget")

    def test_minimize_button_hides_the_window_without_a_callback(self):
        from thaiasr.ui import Overlay

        overlay = Overlay(Config())
        overlay._on_minimize()
        self.assertFalse(overlay.visible)

    def test_visibility_toggles(self):
        from thaiasr.ui import Overlay

        overlay = Overlay(Config())
        self.assertTrue(overlay.visible)
        self.assertFalse(overlay.toggle_visible())
        self.assertFalse(overlay.visible)
        self.assertTrue(overlay.toggle_visible())
        self.assertTrue(overlay.visible)

    def test_overlay_hotkey_toggles_and_ignores_repeats(self):
        events = []
        manager = HotkeyManager(
            hold="ctrl_r",
            quit_combo="ctrl+alt+q",
            overlay_combo="ctrl+alt+h",
            on_overlay_toggle=lambda: events.append("overlay"),
        )
        manager._on_press(Key.ctrl_l)
        manager._on_press(Key.alt_l)
        manager._on_press(KeyCode.from_char("h"))
        self.assertEqual(events, ["overlay"])
        manager._on_press(KeyCode.from_char("h"))  # key repeat while held
        self.assertEqual(events, ["overlay"])
        manager._on_release(KeyCode.from_char("h"))
        manager._on_press(KeyCode.from_char("h"))
        self.assertEqual(events, ["overlay", "overlay"])

    def test_overlay_hotkey_does_not_fire_the_quit_combo(self):
        events = []
        manager = HotkeyManager(
            hold="ctrl_r",
            quit_combo="ctrl+alt+q",
            overlay_combo="ctrl+alt+h",
            on_quit=lambda: events.append("quit"),
            on_overlay_toggle=lambda: events.append("overlay"),
        )
        manager._on_press(Key.ctrl_l)
        manager._on_press(Key.alt_l)
        manager._on_press(KeyCode.from_char("h"))
        manager._on_press(KeyCode.from_char("q"))
        self.assertEqual(events, ["overlay", "quit"])

    def test_config_has_the_overlay_hotkey(self):
        cfg = Config()
        self.assertEqual(cfg["overlay_hotkey"], "ctrl+alt+h")
        self.assertEqual(cfg["start_minimized"], False)


class TestRunnerWiring(unittest.TestCase):
    def test_manager_gets_the_configured_mode(self):
        from thaiasr.app import DictationApp
        from thaiasr.runner import _make_manager

        app = DictationApp(Config({"hotkey_mode": "toggle"}))
        manager = _make_manager(app, Config({"hotkey_mode": "toggle"}))
        self.assertEqual(manager.hold_mode, "toggle")
        manager._on_press(KeyCode.from_char("q"))  # unrelated key must be harmless

    def test_pretty_hotkey_labels(self):
        from thaiasr.runner import pretty_hotkey

        self.assertEqual(pretty_hotkey("ctrl_r"), "Right Ctrl")


if __name__ == "__main__":
    unittest.main()
