"""The dictation app: hotkey -> microphone -> ONNX -> focused window.

Threading model:

* **Tk main thread** -- the overlay (only when ``show_overlay``).
* **pynput listener thread** -- hotkey events; it only puts events on a queue.
* **event thread** -- consumes that queue and drives the state machine.
* **partial thread** -- while recording, re-decodes the utterance so far and
  shows it live (adapts its interval to how fast this machine actually is).
* **final thread** -- one decode per utterance, then paste + history.

All ONNX calls are serialised by ``_asr_lock``; the two worker threads never run
the model at the same time.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .asr import DecodeResult, ModelFilesMissing, OnnxAsr
from .audio import AudioError, MicStream
from .config import Config
from .endpointer import trim_silence
from .features import SAMPLE_RATE
from .output import HistoryWriter, TextOutput

__all__ = ["DictationApp", "AppError", "missing_model_hint"]


class AppError(RuntimeError):
    pass


def missing_model_hint() -> str:
    """What to do when the model is not on disk yet."""
    from .paths import is_frozen

    if is_frozen():
        return "ยังไม่มีโมเดล — คลิกขวาที่ไอคอน tray แล้วเลือก “ดาวน์โหลดโมเดล”"
    return "ยังไม่มีโมเดล — คลิกขวาที่ไอคอน tray เลือก “ดาวน์โหลดโมเดล” หรือรัน python -m thaiasr --download"


def _short_progress(message: str) -> str:
    """Turn huggingface_hub's English progress lines into a short Thai status."""
    text = message.strip()
    if not text:
        return ""
    if "already present" in text:
        return "โมเดลพร้อมอยู่แล้ว"
    if text.startswith("downloading"):
        return "กำลังดาวน์โหลดโมเดล… (ไฟล์ใหญ่มาก ใช้เวลาสักครู่)"
    if "extracting" in text:
        return "กำลังแตกไฟล์โมเดล…"
    if text.startswith("ready"):
        return "ดาวน์โหลดครบแล้ว"
    return text[:80]




@dataclass
class _Event:
    kind: str
    detail: str = ""


class DictationApp:
    IDLE, RECORDING, BUSY = "idle", "recording", "busy"

    def __init__(self, config: Config, *, on_status=None, on_state_change=None):
        """``on_status(text, kind, value)`` is called from worker threads.

        ``kind`` is one of ``idle | recording | busy | partial | final | warn |
        error | ok``; for ``level`` the third argument is the microphone RMS.
        ``on_state_change(state)`` fires whenever the state machine moves, which
        is how the tray icon and the toggle hotkey stay in sync.
        """
        self.cfg = config
        self.on_status = on_status or (lambda *_: None)
        self.on_state_change = on_state_change or (lambda *_: None)
        self.state = self.IDLE
        self._asr: OnnxAsr | None = None
        self._asr_error: str | None = None
        self._asr_lock = threading.Lock()
        self._mic = MicStream()
        self._output = TextOutput(
            str(config["output"]),
            restore_clipboard=bool(config["restore_clipboard"]),
            paste_delay_ms=int(config["paste_delay_ms"]),
        )
        self._events: queue.Queue[_Event] = queue.Queue()
        self._mic_ok = True
        self._stop_partials = threading.Event()
        self._partial_thread: threading.Thread | None = None
        self._quit = threading.Event()
        self._downloading = False
        self._last_level_s = 0.0
        self._started_at = 0.0

    # ------------------------------------------------------------------ model
    @property
    def asr(self) -> OnnxAsr | None:
        return self._asr

    @property
    def asr_error(self) -> str | None:
        return self._asr_error

    def load_model(self) -> bool:
        """Load the ONNX sessions (idempotent, thread-safe)."""
        with self._asr_lock:
            if self._asr is not None:
                return True
        threads = int(self.cfg["num_threads"]) or None
        device = str(self.cfg["device"])
        try:
            asr = OnnxAsr(
                self.cfg.model_dir(),
                precision=str(self.cfg["precision"]),
                device=device,
                num_threads=threads,
            )
        except ModelFilesMissing as exc:
            self._asr_error = str(exc)
            self.on_status(missing_model_hint(), "error")
            return False
        except RuntimeError as exc:
            if device != "cpu":
                self.on_status(f"{exc} — ใช้ CPU แทน", "warn")
                asr = OnnxAsr(
                    self.cfg.model_dir(),
                    precision=str(self.cfg["precision"]),
                    device="cpu",
                    num_threads=threads,
                )
            else:
                raise
        with self._asr_lock:
            self._asr = asr
            self._asr_error = None
        self.on_status(
            f"โมเดลพร้อม ({asr.precision}, {asr.provider.split('ExecutionProvider')[0]})",
            "ok",
        )
        return True

    def preload_async(self) -> None:
        def work():
            try:
                self.load_model()
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                self._asr_error = str(exc)
                self.on_status(f"โหลดโมเดลไม่สำเร็จ: {exc}", "error")

        threading.Thread(target=work, name="thaiasr-preload", daemon=True).start()

    def download_model_async(self, *, on_done=None, on_error=None) -> None:
        """Fetch the model on demand (tray menu), showing progress on the overlay."""
        if self._downloading:
            self.on_status("กำลังดาวน์โหลดอยู่แล้ว…", "busy")
            return
        self._downloading = True

        def progress(message: str) -> None:
            short = _short_progress(message)
            if short:
                self.on_status(short, "busy")

        def work():
            from .model_store import download_model, model_size_hint

            try:
                download_model(
                    self.cfg.model_dir(),
                    precision=str(self.cfg["precision"]),
                    force=False,
                    progress=progress,
                )
            except Exception as exc:  # noqa: BLE001 - report to the user
                self.on_status(f"ดาวน์โหลดไม่สำเร็จ: {exc}", "error")
                if on_error:
                    on_error(exc)
                return
            finally:
                self._downloading = False
            self.on_status("ดาวน์โหลดเสร็จ — กำลังโหลดโมเดล…", "busy")
            try:
                self.load_model()
            except Exception as exc:  # noqa: BLE001
                self.on_status(f"โหลดโมเดลไม่สำเร็จ: {exc}", "error")
                return
            self.on_status(f"โมเดลพร้อม ({model_size_hint(str(self.cfg['precision']))})", "ok")
            if on_done:
                on_done()

        threading.Thread(target=work, name="thaiasr-download", daemon=True).start()

    # ----------------------------------------------------------------- events
    def post(self, kind: str, detail: str = "") -> None:
        self._events.put(_Event(kind, detail))

    def request_quit(self) -> None:
        self.post("quit")

    # ------------------------------------------------------------- lifecycle
    def run(self, *, block: bool = True) -> None:
        if bool(self.cfg["preload"]):
            self.preload_async()
        else:
            self.on_status("โหลดโมเดลเมื่อเริ่มพูดครั้งแรก", "idle")
        if block:
            self._event_loop()

    def stop(self) -> None:
        self._quit.set()
        self.post("quit")

    # ------------------------------------------------------------ state machine
    def _set_state(self, state: str) -> None:
        if state == self.state:
            return
        self.state = state
        try:
            self.on_state_change(state)
        except Exception:  # noqa: BLE001 - listeners must not break the app
            pass

    def toggle_recording(self) -> None:
        """Start or stop from the tray (same path as the hotkey)."""
        self.post("stop" if self.state == self.RECORDING else "start")

    def _event_loop(self) -> None:
        while not self._quit.is_set():
            try:
                event = self._events.get(timeout=0.2)
            except queue.Empty:
                self._tick_level()
                continue
            try:
                self._handle(event)
            except Exception as exc:  # noqa: BLE001 - one bad utterance must not stop the app
                self._fail(exc)
        self._cleanup()

    def _handle(self, event: _Event) -> None:
        if event.kind == "start":
            self._start_recording()
        elif event.kind == "stop":
            self._finish_recording()
        elif event.kind == "cancel":
            self._cancel_recording()
        elif event.kind == "quit":
            self._quit.set()

    def _tick_level(self) -> None:
        if self.state != self.RECORDING:
            return
        now = time.monotonic()
        if now - self._last_level_s < 0.08:
            return
        self._last_level_s = now
        self.on_status("", "level", self._mic.level)

    # -------------------------------------------------------------- recording
    def _start_recording(self) -> None:
        if self.state != self.IDLE:
            return
        if self._asr is None and not self.load_model():
            return
        try:
            self._mic.start()
        except AudioError as exc:
            self.on_status(str(exc), "error")
            return
        self._set_state(self.RECORDING)
        self._started_at = time.monotonic()
        self.on_status("กำลังฟัง…", "recording")
        if bool(self.cfg["partials"]):
            self._stop_partials.clear()
            self._partial_thread = threading.Thread(
                target=self._partial_loop, name="thaiasr-partials", daemon=True
            )
            self._partial_thread.start()

    def _finish_recording(self) -> None:
        if self.state != self.RECORDING:
            return
        self._stop_partials.set()
        audio = self._mic.stop()
        self._set_state(self.BUSY)
        self.on_status("กำลังถอดเสียง…", "busy")
        threading.Thread(
            target=self._finalize, args=(audio,), name="thaiasr-final", daemon=True
        ).start()

    def _cancel_recording(self) -> None:
        if self.state != self.RECORDING:
            return
        self._stop_partials.set()
        self._mic.cancel()
        self._set_state(self.IDLE)
        self.on_status("ยกเลิกแล้ว", "idle")

    # ------------------------------------------------------------------ decode
    def _partial_loop(self) -> None:
        interval = max(0.15, float(self.cfg["partial_interval_ms"]) / 1000.0)
        max_utterance = float(self.cfg["partial_max_utterance_s"])
        while not self._stop_partials.wait(interval):
            if self.state != self.RECORDING:
                return
            elapsed = time.monotonic() - self._started_at
            if elapsed > float(self.cfg["max_utterance_s"]):
                self.post("stop")
                return
            if elapsed > max_utterance:
                continue
            audio = self._mic.snapshot()
            if audio.size < int(0.25 * SAMPLE_RATE):
                continue
            if not self._asr_lock.acquire(blocking=False):
                continue
            try:
                asr = self._asr
                if asr is None:
                    continue
                started = time.monotonic()
                result = asr.transcribe(audio, lang=str(self.cfg["lang"]), mode=str(self.cfg["mode"]))
                cost = time.monotonic() - started
            except Exception as exc:  # noqa: BLE001 - partials are best effort
                self._stop_partials.set()
                self.on_status(f"partial ล้มเหลว: {exc}", "error")
                return
            finally:
                self._asr_lock.release()
            if result.text:
                self.on_status(result.text, "partial")
            # Back off if decoding can no longer keep up with the interval.
            if cost > interval * 0.5:
                interval = min(interval * 2.0, 5.0)

    def _finalize(self, audio: np.ndarray) -> None:
        try:
            asr = self._asr
            if asr is None:
                raise AppError("โมเดลยังไม่พร้อม")
            duration = audio.size / SAMPLE_RATE
            if duration < float(self.cfg["min_utterance_s"]):
                self.on_status("สั้นเกินไป — กดค้างนานขึ้นอีกนิด", "warn")
                return
            if bool(self.cfg["trim_silence"]):
                audio = trim_silence(audio, SAMPLE_RATE, window_s=float(self.cfg["min_silence_s"]) + 1.0)
                if audio.size < int(0.2 * SAMPLE_RATE):
                    self.on_status("ไม่ได้ยินเสียงพูด", "warn")
                    return
            with self._asr_lock:
                result = asr.transcribe_recording(
                    audio,
                    lang=str(self.cfg["lang"]),
                    mode=str(self.cfg["mode"]),
                    split_over_s=float(self.cfg["split_over_s"]),
                    min_silence_s=float(self.cfg["min_silence_s"]),
                    trim=False,  # already trimmed above
                )
            if not result.text:
                self.on_status("ถอดเสียงไม่ได้ (ไม่มีข้อความ)", "warn")
                return
            note = self._output.send(result.text)
            self._history(result, note)
            self.on_status(result.text, "final")
        except Exception as exc:  # noqa: BLE001 - report instead of dying
            self._fail(exc)
        finally:
            self._set_state(self.IDLE)
            self.on_status("", "recording", False)

    def _fail(self, exc: Exception) -> None:
        self._set_state(self.IDLE)
        try:
            self._mic.cancel()
        except Exception:  # noqa: BLE001
            pass
        self.on_status(f"ผิดพลาด: {exc}", "error")

    # ------------------------------------------------------------------ extras
    def _history(self, result: DecodeResult, note: str) -> None:
        if not bool(self.cfg["save_history"]):
            return
        writer = HistoryWriter(self.cfg.path_for("history_path"))
        writer.append(
            result.text,
            duration_s=round(result.duration_s, 3),
            lang=result.lang,
            mode=str(self.cfg["mode"]),
            bias=result.bias,
            attempts=result.attempts,
            rtf=round(result.rtf, 3) if result.duration_s else None,
            output=note,
        )

    def _cleanup(self) -> None:
        self._stop_partials.set()
        try:
            self._mic.cancel()
        except Exception:  # noqa: BLE001
            pass
