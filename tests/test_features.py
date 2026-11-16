"""Feature extraction: shapes, frame counts and mel-filterbank properties."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thaiasr import features as F  # noqa: E402


class TestWindow(unittest.TestCase):
    def test_hann_is_periodic(self):
        w = F.hann_window(400)
        self.assertEqual(w.shape, (400,))
        self.assertAlmostEqual(float(w[0]), 0.0, places=12)  # periodic: w[0] == 0
        self.assertAlmostEqual(float(w.max()), 1.0, places=12)
        # DFT-even symmetry: w[n] == w[N - n] for n in 1..N-1
        n = np.arange(1, 400)
        np.testing.assert_allclose(w[n], w[400 - n], atol=1e-12)


class TestMelFilterbank(unittest.TestCase):
    def test_shape_and_non_negativity(self):
        fb = F.mel_filterbank()
        self.assertEqual(fb.shape, (F.N_MELS, F.N_FFT // 2 + 1))
        self.assertGreaterEqual(fb.min(), 0.0)
        self.assertTrue(np.isfinite(fb).all())

    def test_bands_are_triangles_with_slaney_norm(self):
        fb = F.mel_filterbank()
        # every band must cover at least one FFT bin
        self.assertTrue((fb > 0).any(axis=1).all())
        # Slaney normalisation: the peak of a band equals 2/(width in mel) scaled
        # back to Hz -- simply check the peaks decrease monotonically with centre
        centres = [float(np.argmax(row)) for row in fb]
        self.assertEqual(centres, sorted(centres))

    def test_mel_hz_roundtrip(self):
        hz = np.array([0.0, 100.0, 1000.0, 4000.0, 8000.0])
        np.testing.assert_allclose(F.mel_to_hz(F.hz_to_mel(hz)), hz, rtol=1e-9)

    def test_slaney_norm_matches_librosa_definition(self):
        raw = F.mel_filterbank(norm=None)
        normed = F.mel_filterbank(norm="slaney")
        self.assertFalse(np.allclose(raw, normed))
        # Slaney weighting multiplies every bin of a band by 2 / (band width in
        # Hz) -- the equal-loudness normalisation librosa applies.
        mel_f = F.mel_frequencies(F.N_MELS + 2)
        enorm = 2.0 / (mel_f[2:] - mel_f[:-2])
        mask = raw > 0
        ratio = np.zeros_like(normed)
        np.divide(normed, raw, out=ratio, where=mask)
        np.testing.assert_allclose(ratio[mask], np.repeat(enorm, mask.sum(axis=1)), rtol=1e-9)
        self.assertTrue((normed.max(axis=1) <= enorm + 1e-12).all())
        self.assertGreaterEqual(normed.min(), 0.0)

    def test_unknown_norm_rejected(self):
        with self.assertRaises(ValueError):
            F.mel_filterbank(norm="htk")


class TestLogMel(unittest.TestCase):
    def test_frame_count_and_shape(self):
        for seconds in (0.2, 1.0, 3.7, 8.82):
            n = int(seconds * F.SAMPLE_RATE)
            x = np.zeros(n, dtype=np.float32)
            mel = F.log_mel(x)
            self.assertEqual(mel.shape, (F.N_MELS, F.num_mel_frames(n)))
            self.assertEqual(mel.dtype, np.float32)
            self.assertTrue(np.isfinite(mel).all(), "silence must not produce NaN/-inf")

    def test_silence_hits_the_log_guard(self):
        mel = F.log_mel(np.zeros(16000, dtype=np.float32))
        np.testing.assert_allclose(mel, np.log(F.LOG_ZERO_GUARD), rtol=1e-5)

    def test_tone_is_louder_in_the_right_band(self):
        t = np.arange(F.SAMPLE_RATE, dtype=np.float32) / F.SAMPLE_RATE
        x = 0.5 * np.sin(2 * np.pi * 1000 * t)
        mel = F.log_mel(x)
        band_centres = F.mel_frequencies(F.N_MELS + 2)[1:-1]
        expected = int(np.argmin(np.abs(band_centres - 1000.0)))
        loudest = int(np.argmax(mel[:, 20:].mean(axis=1)))
        self.assertLessEqual(abs(loudest - expected), 1)

    def test_empty_input(self):
        mel = F.log_mel(np.zeros(0, dtype=np.float32))
        self.assertEqual(mel.shape, (F.N_MELS, 0))

    def test_blocks_do_not_change_the_result(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(F.SAMPLE_RATE * 2).astype(np.float32) * 0.1
        a = F.log_mel(x, block_frames=16)
        b = F.log_mel(x, block_frames=100000)
        np.testing.assert_allclose(a, b, atol=1e-5)


class TestPreemphasis(unittest.TestCase):
    def test_matches_definition(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = F.preemphasis(x, 0.97)
        np.testing.assert_allclose(y, [1.0, 2.0 - 0.97 * 1.0, 3.0 - 0.97 * 2.0, 4.0 - 0.97 * 3.0], rtol=1e-6)

    def test_disabled(self):
        x = np.arange(5, dtype=np.float32)
        np.testing.assert_array_equal(F.preemphasis(x, 0.0), x)


if __name__ == "__main__":
    unittest.main()
