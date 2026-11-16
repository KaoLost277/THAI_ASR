"""Token vocabulary for OpenThai2.0-ASR-Streaming.

``tokens.txt`` holds one SentencePiece piece per line: a piece's id is its line
number minus one (that off-by-one is what the ONNX model card documents).  The
blank symbol of the RNN-T joint network sits one past the last piece.

``prompt_index`` selects the language prompt baked into the encoder:
32 is Thai (everything in Thai script), 63 is Thai mixed with English.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["Tokenizer", "PROMPT_INDEX", "prompt_index"]

PROMPT_INDEX = {
    "th-TH": 32,
    "th-mix": 63,
}

_SPACE = "\u2581"  # SentencePiece's word-boundary marker
_SPECIAL = re.compile(r"^<[^>]*>$")


def prompt_index(lang: str) -> int:
    """Map a language name to the encoder's ``prompt_index``."""
    try:
        return PROMPT_INDEX[lang]
    except KeyError:
        raise ValueError(
            f"unknown lang {lang!r}; expected one of {sorted(PROMPT_INDEX)}"
        ) from None


class Tokenizer:
    """Piece table with NeMo-compatible detokenisation."""

    def __init__(self, tokens_file: str | Path):
        text = Path(tokens_file).read_text(encoding="utf-8")
        self.pieces = [line for line in text.splitlines()]
        if not self.pieces:
            raise ValueError(f"empty token file: {tokens_file}")
        self.path = Path(tokens_file)

    def __len__(self) -> int:
        return len(self.pieces)

    @property
    def blank_id(self) -> int:
        return len(self.pieces)

    @property
    def vocab_size(self) -> int:
        """Number of joint-network outputs (pieces plus blank)."""
        return len(self.pieces) + 1

    def piece(self, token_id: int) -> str | None:
        """Return the piece text for an id (``None`` for blank / out of range)."""
        if 0 <= token_id < len(self.pieces):
            return self.pieces[token_id]
        return None

    def decode(self, token_ids) -> str:
        """Detokenise ids the way NeMo does: pieces joined, ``▁`` -> space.

        Special tokens such as ``<unk>`` or the ``<th-TH>`` prompt tag are
        dropped (NeMo's ``strip_lang_tags``), and the result is stripped.
        """
        out = []
        for tid in token_ids:
            piece = self.piece(tid)
            if piece is None or _SPECIAL.match(piece):
                continue
            out.append(piece.replace(_SPACE, " "))
        return "".join(out).strip()
