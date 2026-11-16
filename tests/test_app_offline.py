"""Integration test of the app layer: decode -> output -> history.

Needs the real model, so it skips itself when the ONNX files are absent.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thaiasr.app import DictationApp  # noqa: E402
from thaiasr.audio import load_wav  # noqa: E402
from thaiasr.config import Config  # noqa: E402
from thaiasr.model_store import model_ready  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class TestAppFinalize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = sorted((ROOT / "examples").glob("*.wav"))
        if not cls.examples:
            raise unittest.SkipTest("no example wav files (run tools/fetch_examples.py)")
        if not model_ready(ROOT / "models" / "openthai2.0-asr-streaming-onnx", "fp32"):
            raise unittest.SkipTest("ONNX model not downloaded")

    def _config(self, history: Path) -> Config:
        return Config.load(
            None,
            overrides={
                "model_dir": str(ROOT / "models" / "openthai2.0-asr-streaming-onnx"),
                "precision": "fp32",
                "output": "none",
                "save_history": True,
                "history_path": str(history),
                "partials": False,
                "preload": False,
                "show_overlay": False,
            },
        )

    def test_finalize_writes_text_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "history.jsonl"
            events: list[tuple[str, str]] = []
            app = DictationApp(
                self._config(history), on_status=lambda text, kind="info", value=None: events.append((kind, text))
            )
            self.assertTrue(app.load_model(), app.asr_error)
            app._finalize(load_wav(self.examples[0]))

            finals = [text for kind, text in events if kind == "final"]
            self.assertTrue(finals, f"no final text; events={events}")
            self.assertTrue(any("\u0e00" <= ch <= "\u0e7f" for ch in finals[0]))
            self.assertEqual(app.state, DictationApp.IDLE)

            content = history.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(content), 1)
            import json

            record = json.loads(content[0])
            self.assertEqual(record["text"], finals[0])
            self.assertEqual(record["lang"], "th-TH")
            self.assertIn("rtf", record)

    def test_too_short_utterance_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            events: list[tuple[str, str]] = []
            app = DictationApp(
                self._config(Path(tmp) / "h.jsonl"),
                on_status=lambda text, kind="info", value=None: events.append((kind, text)),
            )
            import numpy as np

            self.assertTrue(app.load_model(), app.asr_error)
            app._finalize(np.zeros(1600, dtype=np.float32))
            self.assertTrue(any(kind == "warn" for kind, _ in events), events)
            self.assertFalse([t for k, t in events if k == "final"])


if __name__ == "__main__":
    unittest.main()
