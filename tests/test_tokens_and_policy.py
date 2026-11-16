"""Tokenizer id mapping and detokenisation, plus the blank-bias policy."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thaiasr.asr import (  # noqa: E402
    BLANK_BIAS,
    EMPTY_RETRY_BIAS,
    SHORT_CLIP_BIAS,
    SHORT_CLIP_S,
    DecodeResult,
    OnnxAsr,
)
from thaiasr.tokens import PROMPT_INDEX, Tokenizer, prompt_index  # noqa: E402


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tokens.txt"
        # one piece per line; id == line number - 1
        self.path.write_text("<unk>\n\u2581\n\u0e01\n\u0e32\nab\n", encoding="utf-8")
        self.tok = Tokenizer(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_id_mapping(self):
        self.assertEqual(self.tok.piece(0), "<unk>")
        self.assertEqual(self.tok.piece(1), "\u2581")
        self.assertEqual(self.tok.piece(2), "\u0e01")
        self.assertIsNone(self.tok.piece(99))

    def test_blank_is_one_past_the_last_piece(self):
        self.assertEqual(len(self.tok), 5)
        self.assertEqual(self.tok.blank_id, 5)
        self.assertEqual(self.tok.vocab_size, 6)

    def test_decode_word_boundary_and_specials(self):
        text = self.tok.decode([1, 2, 3, 1, 4])
        self.assertEqual(text, "\u0e01\u0e32 ab")

    def test_lang_tags_are_stripped(self):
        path = Path(self.tmp.name) / "tags.txt"
        path.write_text("<unk>\n\u2581\n<th-TH>\n\u0e01\n", encoding="utf-8")
        tok = Tokenizer(path)
        # ids: 0=<unk>, 1=space, 2=<th-TH>, 3=ก
        self.assertEqual(tok.decode([2, 3, 2]), "\u0e01")
        self.assertEqual(tok.decode([0, 0]), "")

    def test_out_of_range_ids_are_ignored(self):
        self.assertEqual(self.tok.decode([5, 99, 2]), "\u0e01")

    def test_prompt_index(self):
        self.assertEqual(prompt_index("th-TH"), 32)
        self.assertEqual(prompt_index("th-mix"), 63)
        self.assertEqual(PROMPT_INDEX["th-TH"], 32)
        with self.assertRaises(ValueError):
            prompt_index("en-US")


class _FakeAsr(OnnxAsr):
    """Runs the real policy code without loading ONNX."""

    def __init__(self, texts_by_bias=None):
        self.calls: list[float] = []
        self.texts_by_bias = texts_by_bias or {}

    def decode_audio(self, audio, *, lang="th-TH", bias=BLANK_BIAS["standard"]):
        self.calls.append(bias)
        text = self.texts_by_bias.get(bias, "ข้อความ")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return DecodeResult(text=text, bias=bias, lang=lang, duration_s=audio.size / 16000)


class TestBlankBiasPolicy(unittest.TestCase):
    def test_standard_uses_minus_one(self):
        asr = _FakeAsr()
        res = asr.transcribe(np.zeros(16000 * 5, dtype=np.float32))
        self.assertEqual(asr.calls, [BLANK_BIAS["standard"]])
        self.assertEqual(res.text, "ข้อความ")
        self.assertEqual(res.attempts, 1)

    def test_short_clip_uses_minus_two_point_five(self):
        asr = _FakeAsr()
        asr.transcribe(np.zeros(int(SHORT_CLIP_S * 16000) - 160, dtype=np.float32))
        self.assertEqual(asr.calls, [SHORT_CLIP_BIAS])

    def test_conversation_uses_minus_two(self):
        asr = _FakeAsr()
        asr.transcribe(np.zeros(16000 * 5, dtype=np.float32), mode="conversation")
        self.assertEqual(asr.calls, [BLANK_BIAS["conversation"]])

    def test_empty_result_is_retried_at_minus_four(self):
        asr = _FakeAsr(texts_by_bias={-1.0: "", -4.0: "กู้คืนได้"})
        res = asr.transcribe(np.zeros(16000 * 5, dtype=np.float32))
        self.assertEqual(asr.calls, [-1.0, EMPTY_RETRY_BIAS])
        self.assertEqual(res.text, "กู้คืนได้")
        self.assertEqual(res.attempts, 2)

    def test_still_empty_after_retry(self):
        asr = _FakeAsr(texts_by_bias={-1.0: "", -4.0: ""})
        res = asr.transcribe(np.zeros(16000 * 5, dtype=np.float32))
        self.assertEqual(res.text, "")
        self.assertEqual(res.attempts, 2)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            _FakeAsr().transcribe(np.zeros(16000, dtype=np.float32), mode="nope")


if __name__ == "__main__":
    unittest.main()
