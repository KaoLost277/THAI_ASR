"""Energy endpointer, audio helpers, hotkey parsing and model-file checks."""

from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thaiasr import endpointer as EP  # noqa: E402
from thaiasr.audio import load_wav, resample  # noqa: E402
from thaiasr.features import SAMPLE_RATE  # noqa: E402
from thaiasr.hotkey import HotkeyError, HotkeyManager, parse_spec  # noqa: E402
from thaiasr.model_store import model_ready, model_size_hint, verify_model  # noqa: E402


def _speech(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    tone = 0.25 * np.sin(2 * np.pi * 180 * t)
    return (tone + 0.05 * rng.standard_normal(n)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


class TestLevelsAndMask(unittest.TestCase):
    def test_levels_are_dbfs(self):
        levels = EP.frame_levels(_silence(0.5))
        self.assertTrue(np.all(levels < -100.0))
        loud = EP.frame_levels(np.ones(SAMPLE_RATE, dtype=np.float32) * 0.5)
        self.assertAlmostEqual(float(np.median(loud)), 20 * np.log10(0.5), places=1)

    def test_mask_marks_speech(self):
        audio = np.concatenate([_silence(1.0), _speech(1.5), _silence(1.5)])
        mask = EP.speech_mask(EP.frame_levels(audio))
        self.assertTrue(mask.any())
        self.assertFalse(mask.all())
        hop = int(SAMPLE_RATE * EP.HOP_MS / 1000)
        speech_region = int(1.2 * SAMPLE_RATE / hop)
        self.assertTrue(mask[speech_region : speech_region + 10].all())

    def test_mask_empty(self):
        self.assertEqual(EP.speech_mask(np.empty(0, dtype=np.float32)).size, 0)


class TestSplitAtPauses(unittest.TestCase):
    def test_two_utterances_are_separated(self):
        audio = np.concatenate([_speech(1.5), _silence(1.2), _speech(1.5, seed=1)])
        pieces = EP.split_at_pauses(audio, min_silence_s=0.7)
        self.assertEqual(len(pieces), 2)
        self.assertLess(pieces[0][0], pieces[0][1])
        self.assertLessEqual(pieces[0][1], pieces[1][0])
        for start, end in pieces:
            self.assertGreater(end - start, int(0.5 * SAMPLE_RATE))

    def test_short_gap_is_not_a_split(self):
        audio = np.concatenate([_speech(1.0), _silence(0.2), _speech(1.0, seed=2)])
        pieces = EP.split_at_pauses(audio, min_silence_s=0.7)
        self.assertEqual(len(pieces), 1)

    def test_long_piece_is_force_split(self):
        audio = _speech(12.0)
        pieces = EP.split_at_pauses(audio, min_silence_s=0.7, max_piece_s=5.0)
        self.assertGreater(len(pieces), 1)
        for start, end in pieces:
            self.assertLessEqual(end - start, 5.5 * SAMPLE_RATE + 1)

    def test_all_silence_keeps_everything(self):
        audio = _silence(2.0)
        self.assertEqual(EP.split_at_pauses(audio), [(0, audio.size)])

    def test_empty(self):
        self.assertEqual(EP.split_at_pauses(np.zeros(0, dtype=np.float32)), [])


class TestTrimSilence(unittest.TestCase):
    def test_trims_both_ends(self):
        audio = np.concatenate([_silence(1.0), _speech(1.0), _silence(1.0)])
        out = EP.trim_silence(audio)
        self.assertLess(out.size, audio.size)
        self.assertGreater(out.size, int(0.9 * SAMPLE_RATE))
        self.assertGreater(float(np.abs(out).max()), 0.05)

    def test_silence_untouched(self):
        audio = _silence(1.0)
        np.testing.assert_array_equal(EP.trim_silence(audio), audio)


class TestAudio(unittest.TestCase):
    def test_resample_identity(self):
        x = _speech(0.2)
        np.testing.assert_array_equal(resample(x, SAMPLE_RATE, SAMPLE_RATE), x)

    def test_resample_length_and_range(self):
        x = _speech(0.5)
        down = resample(x, 48000, 16000)
        self.assertAlmostEqual(down.size, x.size / 3, delta=2)
        self.assertLess(float(np.abs(down).max()), 1.0)

    def test_long_signal_resample_is_chunk_free(self):
        x = _speech(3.0)
        up = resample(x, 16000, 22050)
        self.assertAlmostEqual(up.size, int(round(x.size * 22050 / 16000)), delta=2)

    def test_wav_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            audio = (_speech(0.4) * 32767).astype("<i2")
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(audio.tobytes())
            out = load_wav(path)
            self.assertEqual(out.dtype, np.float32)
            self.assertEqual(out.size, audio.size)
            np.testing.assert_allclose(out, audio.astype(np.float32) / 32768.0, atol=1e-6)

    def test_stereo_is_mixed_and_resampled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "st.wav"
            stereo = np.stack([np.ones(4800, dtype="<i2") * 16384] * 2, axis=1)
            with wave.open(str(path), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(48000)
                w.writeframes(stereo.tobytes())
            out = load_wav(path)
            self.assertAlmostEqual(out.size, 1600, delta=4)
            self.assertGreater(float(out.mean()), 0.4)


class TestHotkeys(unittest.TestCase):
    def test_parse_spec(self):
        self.assertEqual(parse_spec("ctrl_r"), ("ctrl_r",))
        self.assertEqual(parse_spec("Ctrl+Alt+Q"), ("ctrl", "alt", "q"))
        self.assertEqual(parse_spec("f9"), ("f9",))
        self.assertEqual(parse_spec("control+shift+space"), ("ctrl", "shift", "space"))

    def test_bad_specs(self):
        for spec in ("", "   ", "ctrl+nope", "f25", "ctrl+"):
            with self.assertRaises(HotkeyError):
                parse_spec(spec)

    def test_hold_key_must_be_single(self):
        with self.assertRaises(HotkeyError):
            HotkeyManager(hold="ctrl+alt")

    def test_names_for_modifier(self):
        from pynput.keyboard import Key, KeyCode

        from thaiasr.hotkey import _names_for

        self.assertIn("ctrl", _names_for(Key.ctrl_r))
        self.assertIn("ctrl_r", _names_for(Key.ctrl_r))
        self.assertIn("q", _names_for(KeyCode.from_char("q")))
        self.assertNotIn("ctrl", _names_for(KeyCode.from_char("q")))

    def test_names_for_virtual_key_no_char(self):
        """A non-Latin layout reports the physical Q key without a 'q' char."""
        from pynput.keyboard import KeyCode

        from thaiasr.hotkey import _names_for

        names = _names_for(KeyCode(vk=0x51))  # physical Q
        self.assertIn("q", names)
        self.assertIn("vk81", names)
        self.assertIn("q", _names_for(KeyCode(vk=0x51, char="ๆ")))

    def test_quit_combo_fires_with_thai_layout_keys(self):
        from pynput.keyboard import Key, KeyCode

        fired = []
        manager = HotkeyManager(
            hold="ctrl_r", quit_combo="ctrl+alt+q", on_quit=lambda: fired.append(1)
        )
        manager._on_press(Key.ctrl_l)
        manager._on_press(Key.alt_l)
        manager._on_press(KeyCode(vk=0x51))  # Q on a Thai keyboard
        self.assertEqual(fired, [1])


class TestModelStore(unittest.TestCase):
    def test_missing_files_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems = verify_model(tmp, "fp32")
            self.assertTrue(any("encoder.onnx" in p for p in problems))
            self.assertFalse(model_ready(tmp, "fp32"))

    def test_size_hint(self):
        fp32 = float(model_size_hint("fp32").split()[0])
        int8 = float(model_size_hint("int8").split()[0])
        self.assertGreater(fp32, 2.5)
        self.assertLess(int8, 1.1)
        self.assertGreater(fp32, int8)

    def test_unknown_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                verify_model(tmp, "int4")


if __name__ == "__main__":
    unittest.main()
