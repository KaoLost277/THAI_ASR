"""Output layer: clipboard handling, paste bookkeeping and target reporting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thaiasr.output import HistoryWriter, TextOutput, clipboard_text, foreground_window_info  # noqa: E402


class TestForegroundInfo(unittest.TestCase):
    def test_reports_something(self):
        info = foreground_window_info()
        self.assertIsInstance(info, str)
        self.assertTrue(info)

    def test_clipboard_roundtrip(self):
        import pyperclip

        if sys.platform != "win32":
            self.skipTest("clipboard test is Windows-only")
        saved = clipboard_text()
        try:
            pyperclip.copy("ThaiASR clipboard test ๑๒๓")
            self.assertEqual(clipboard_text(), "ThaiASR clipboard test ๑๒๓")
        finally:
            pyperclip.copy(saved or "")


class TestSendModes(unittest.TestCase):
    def test_none_mode_does_not_touch_the_clipboard(self):
        import pyperclip

        if sys.platform != "win32":
            self.skipTest("clipboard test is Windows-only")
        pyperclip.copy("sentinel")
        out = TextOutput("none")
        self.assertEqual(out.send("ข้อความ"), "shown")
        self.assertEqual(clipboard_text(), "sentinel")

    def test_clipboard_mode_copies_and_reports_target(self):
        if sys.platform != "win32":
            self.skipTest("clipboard test is Windows-only")
        out = TextOutput("clipboard")
        note = out.send("ข้อความทดสอบ")
        self.assertTrue(note.startswith("copied"), note)
        self.assertIn("target:", note)
        self.assertEqual(clipboard_text(), "ข้อความทดสอบ")

    def test_empty_text_is_a_no_op(self):
        self.assertEqual(TextOutput("paste").send(""), "empty")

    def test_paste_keeps_the_text_on_the_clipboard_by_default(self):
        if sys.platform != "win32":
            self.skipTest("clipboard test is Windows-only")
        out = TextOutput("paste")
        self.assertFalse(out.restore_clipboard)
        note = out.send("ข้อความ")
        self.assertIn("pasted", note)
        # the whole point: the text is still there for a manual Ctrl+V
        self.assertEqual(clipboard_text(), "ข้อความ")


class TestHistory(unittest.TestCase):
    def test_appends_jsonl(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.jsonl"
            writer = HistoryWriter(path)
            writer.append("สวัสดี", duration_s=1.5, output="pasted [target: x]")
            writer.append("", duration_s=2.0)  # empty text is skipped
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["text"], "สวัสดี")
            self.assertEqual(record["duration_s"], 1.5)

    def test_disabled_writer_writes_nothing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.jsonl"
            HistoryWriter(path, enabled=False).append("x")
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
