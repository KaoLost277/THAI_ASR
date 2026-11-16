"""ONNX Runtime backend: FastConformer encoder + greedy RNN-T decoder.

Contract read off the exported graphs (``iapp/OpenThai2.0-ASR-Streaming-ONNX``):

    encoder.onnx
        in  audio_signal   float32 [B, 128, T]   (log-mel, 16 kHz mono)
            length         int64   [B]           (mel frame count)
            prompt_index   int64   [B]           (32 = th-TH, 63 = th-mix)
        out encoded        float32 [B, T/8, 1024]
            encoded_length int64   [B]

    decoder_joint.onnx            (one greedy RNN-T step)
        in  encoder_outputs float32 [B, 1024, 1]
            targets         int32   [B, 1]
            target_length   int32   [B]
            input_states_1  float32 [2, B, 640]
            input_states_2  float32 [2, B, 640]
        out outputs         float32 [B, 1, 1, 14989]   (last index = blank)
            prednet_lengths int32   [B]
            output_states_1 float32 [2, B, 640]
            output_states_2 float32 [2, B, 640]

The encoder carries no cache tensors: it is a whole-utterance graph, which is
why long recordings are cut at pauses and decoded piece by piece (see
``transcribe_long``), exactly as the model cards require.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features import SAMPLE_RATE, log_mel
from .tokens import Tokenizer, prompt_index

__all__ = ["OnnxAsr", "DecodeResult", "BLANK_BIAS", "model_files", "ModelFilesMissing"]

# Blank-score bias per mode (the exported ONNX file has no bias baked in, so we
# apply it here -- see the model card's "Add a bias to the blank's score").
BLANK_BIAS = {"standard": -1.0, "conversation": -2.0, "meeting": -1.0}
SHORT_CLIP_S = 2.0
SHORT_CLIP_BIAS = -2.5
EMPTY_RETRY_BIAS = -4.0
MAX_SYMBOLS_PER_STEP = 2

ENCODER_NAMES = {"fp32": ("encoder.onnx", "encoder.onnx.data"), "int8": ("encoder.int8.onnx", "encoder.int8.onnx.data")}


class ModelFilesMissing(FileNotFoundError):
    """Raised when the ONNX model files are not on disk yet."""


@dataclass
class ModelFiles:
    encoder: Path
    decoder: Path
    tokens: Path
    data: Path | None = None

    def missing(self) -> list[Path]:
        out = [p for p in (self.encoder, self.decoder, self.tokens) if not p.exists()]
        if self.data is not None and not self.data.exists():
            out.append(self.data)
        return out


def model_files(model_dir: str | Path, precision: str = "fp32") -> ModelFiles:
    """Locate the ONNX files for a precision, without touching them."""
    try:
        enc_name, data_name = ENCODER_NAMES[precision]
    except KeyError:
        raise ValueError(f"unknown precision {precision!r}; use 'fp32' or 'int8'") from None
    d = Path(model_dir)
    return ModelFiles(
        encoder=d / enc_name,
        decoder=d / "decoder_joint.onnx",
        tokens=d / "tokens.txt",
        data=d / data_name,
    )


@dataclass
class DecodeResult:
    text: str
    tokens: list[int] = field(default_factory=list)
    bias: float = 0.0
    lang: str = "th-TH"
    duration_s: float = 0.0
    encode_s: float = 0.0
    decode_s: float = 0.0
    attempts: int = 1
    segments: list[dict] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        return self.encode_s + self.decode_s

    @property
    def rtf(self) -> float:
        """Real-time factor: ``total_s / duration_s`` (lower is faster)."""
        return self.total_s / self.duration_s if self.duration_s > 0 else float("nan")


class OnnxAsr:
    """Loads the model once and transcribes 16 kHz mono float32 audio."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        precision: str = "fp32",
        device: str = "cpu",
        num_threads: int | None = None,
        verbose: bool = False,
    ):
        files = model_files(model_dir, precision)
        missing = files.missing()
        if missing:
            raise ModelFilesMissing(
                "model files not found:\n  "
                + "\n  ".join(str(p) for p in missing)
                + "\nrun:  python -m thaiasr --download"
            )
        self.files = files
        self.precision = precision
        self.device = device

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if num_threads is None:
            num_threads = max(1, min(8, os.cpu_count() or 4))
        self.num_threads = num_threads
        opts.intra_op_num_threads = num_threads
        opts.log_severity_level = 2 if verbose else 3

        providers = self._providers(device)
        self.encoder = ort.InferenceSession(str(files.encoder), opts, providers=providers)
        self.decoder = ort.InferenceSession(str(files.decoder), opts, providers=providers)
        self.provider = self.encoder.get_providers()[0]

        self.tokens = Tokenizer(files.tokens)
        self.blank_id = self.tokens.blank_id

        self._dec_out = [o.name for o in self.decoder.get_outputs()]
        # Index by name: the export's output order is not contractual.
        self._i_scores = self._dec_out.index("outputs")
        self._i_state1 = self._dec_out.index("output_states_1")
        self._i_state2 = self._dec_out.index("output_states_2")
        self._state_shape = (2, 1, 640)
        self._zeros = np.zeros(self._state_shape, dtype=np.float32)

    @staticmethod
    def _providers(device: str):
        available = ort.get_available_providers()
        if device == "cuda":
            if "CUDAExecutionProvider" in available:
                return [
                    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                    "CPUExecutionProvider",
                ]
            raise RuntimeError(
                "device='cuda' needs onnxruntime-gpu (available providers: "
                + ", ".join(available)
                + ")"
            )
        return ["CPUExecutionProvider"]

    # ------------------------------------------------------------------ encode
    def encode(self, mel: np.ndarray, lang: str = "th-TH") -> tuple[np.ndarray, int]:
        """Run the encoder on log-mel ``[128, T]`` -> ``([1, 1024, Tenc], Tenc)``."""
        if mel.ndim != 2:
            raise ValueError(f"expected mel [128, T], got shape {mel.shape}")
        t = mel.shape[1]
        signal = np.ascontiguousarray(mel[None, :, :], dtype=np.float32)
        out = self.encoder.run(
            None,
            {
                "audio_signal": signal,
                "length": np.array([t], dtype=np.int64),
                "prompt_index": np.array([prompt_index(lang)], dtype=np.int64),
            },
        )
        encoded = out[0].transpose(0, 2, 1)  # [1, 1024, Tenc]
        encoded_len = int(np.asarray(out[1]).reshape(-1)[0])
        return encoded, encoded_len

    # ------------------------------------------------------------------ decode
    def greedy(self, encoded: np.ndarray, encoded_len: int, bias: float) -> list[int]:
        """Greedy RNN-T decoding with ``bias`` added to the blank score."""
        blank = self.blank_id
        state1 = self._zeros.copy()
        state2 = self._zeros.copy()
        emitted: list[int] = []
        run = self.decoder.run
        feed = {
            "targets": np.zeros((1, 1), dtype=np.int32),
            "target_length": np.ones((1,), dtype=np.int32),
            "input_states_1": state1,
            "input_states_2": state2,
        }
        for t in range(encoded_len):
            frame = np.ascontiguousarray(encoded[:, :, t : t + 1])
            feed["encoder_outputs"] = frame
            for _ in range(MAX_SYMBOLS_PER_STEP):
                outs = run(self._dec_out, feed)
                logits = outs[self._i_scores].reshape(-1)
                if logits.shape[0] <= blank:
                    raise RuntimeError(
                        f"joint output has {logits.shape[0]} scores, expected > {blank}"
                    )
                logits = logits.astype(np.float32, copy=True)
                logits[blank] += np.float32(bias)
                idx = int(np.argmax(logits))
                if idx == blank:
                    break
                emitted.append(idx)
                last = idx
                state1, state2 = outs[self._i_state1], outs[self._i_state2]
                feed["targets"][0, 0] = idx
                feed["input_states_1"] = state1
                feed["input_states_2"] = state2
        return emitted

    def decode_audio(
        self,
        audio: np.ndarray,
        *,
        lang: str = "th-TH",
        bias: float = BLANK_BIAS["standard"],
    ) -> DecodeResult:
        """One pass: mel -> encoder -> greedy -> text (no retry policy)."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        t0 = time.perf_counter()
        mel = log_mel(audio)
        encoded, encoded_len = self.encode(mel, lang)
        t1 = time.perf_counter()
        ids = self.greedy(encoded, encoded_len, bias)
        t2 = time.perf_counter()
        return DecodeResult(
            text=self.tokens.decode(ids),
            tokens=ids,
            bias=bias,
            lang=lang,
            duration_s=audio.size / SAMPLE_RATE,
            encode_s=t1 - t0,
            decode_s=t2 - t1,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        lang: str = "th-TH",
        mode: str = "standard",
    ) -> DecodeResult:
        """Utterance decoding with the reference service's blank-bias policy.

        ``standard``: bias -1.0; clips under two seconds are decoded at -2.5
        (short clips otherwise tend to come out empty); an empty result is
        retried once at -4.0.
        """
        if mode not in BLANK_BIAS:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(BLANK_BIAS)}")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = audio.size / SAMPLE_RATE
        bias = BLANK_BIAS[mode]
        if mode == "standard" and duration < SHORT_CLIP_S:
            bias = SHORT_CLIP_BIAS
        result = self.decode_audio(audio, lang=lang, bias=bias)
        attempts = 1
        if not result.text:
            retry = self.decode_audio(audio, lang=lang, bias=EMPTY_RETRY_BIAS)
            attempts = 2
            retry.attempts = attempts
            retry.encode_s += result.encode_s
            retry.decode_s += result.decode_s
            if retry.text:
                return retry
            result.attempts = attempts
            return result
        return result

    def transcribe_recording(
        self,
        audio: np.ndarray,
        *,
        lang: str = "th-TH",
        mode: str = "standard",
        trim: bool = True,
        split_over_s: float = 12.0,
        min_silence_s: float = 0.7,
        max_piece_s: float = 30.0,
    ) -> DecodeResult:
        """Decode an utterance or a recording, splitting at pauses when needed.

        Feeding this model a long stream makes it drop words after quiet pauses,
        so anything longer than ``split_over_s`` is cut where nobody speaks for
        ``min_silence_s`` and decoded piece by piece.  When no such pause exists
        the audio is decoded in one pass, exactly like a short utterance.
        """
        from .endpointer import split_at_pauses

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = audio.size / SAMPLE_RATE
        if duration <= max(0.0, split_over_s):
            return self.transcribe(audio, lang=lang, mode=mode)

        pieces = split_at_pauses(
            audio,
            SAMPLE_RATE,
            min_silence_s=min_silence_s,
            trim=trim,
            max_piece_s=max_piece_s,
        )
        if len(pieces) <= 1:
            return self.transcribe(audio, lang=lang, mode=mode)

        total = DecodeResult(
            text="", bias=BLANK_BIAS[mode], lang=lang, duration_s=duration
        )
        for start, end in pieces:
            res = self.transcribe(audio[start:end], lang=lang, mode=mode)
            total.tokens.extend(res.tokens)
            total.encode_s += res.encode_s
            total.decode_s += res.decode_s
            total.attempts += res.attempts
            total.segments.append(
                {
                    "start": round(start / SAMPLE_RATE, 3),
                    "end": round(end / SAMPLE_RATE, 3),
                    "text": res.text,
                }
            )
        total.text = " ".join(s["text"] for s in total.segments if s["text"]).strip()
        return total

    def transcribe_long(
        self,
        audio: np.ndarray,
        *,
        lang: str = "th-TH",
        mode: str = "standard",
        min_silence_s: float = 0.7,
        trim: bool = True,
    ) -> tuple[str, list[dict]]:
        """Decode a recording piece by piece; returns ``(text, segments)``.

        Each segment is ``{"start": s, "end": s, "text": str}`` in seconds.
        """
        result = self.transcribe_recording(
            audio,
            lang=lang,
            mode=mode,
            trim=trim,
            split_over_s=0.0,
            min_silence_s=min_silence_s,
        )
        return result.text, result.segments

    # ------------------------------------------------------------------ extras
    def warmup(self) -> float:
        """Run a short silent clip; returns the wall time in seconds."""
        t0 = time.perf_counter()
        self.encode(np.zeros((128, 60), dtype=np.float32))
        return time.perf_counter() - t0
