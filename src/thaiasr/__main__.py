"""Command line entry point: ``python -m thaiasr``.

    python -m thaiasr                      # start the dictation overlay
    python -m thaiasr --download           # fetch the ONNX model (~2.6 GB, fp32)
    python -m thaiasr --transcribe a.wav   # transcribe a file
    python -m thaiasr --selftest           # check the whole pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thaiasr",
        description="พูดแล้วออกมาเป็นข้อความ ด้วย OpenThai2.0-ASR-Streaming (ONNX Runtime)",
    )
    parser.add_argument("--config", metavar="TOML", help="config file (default: ./config.toml)")
    parser.add_argument("--model-dir", help="folder holding the ONNX model files")
    parser.add_argument("--precision", choices=["fp32", "int8"], help="fp32 = most accurate (~2.6 GB), int8 = smallest (~1 GB)")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="cpu (default) or cuda")
    parser.add_argument("--threads", type=int, help="ONNX Runtime threads (0 = auto)")
    parser.add_argument("--lang", choices=["th-TH", "th-mix"], help="th-TH = Thai script only, th-mix = English words in Latin")
    parser.add_argument("--mode", choices=["standard", "conversation"], help="blank-bias policy")
    parser.add_argument("--hotkey", help="push-to-talk key, e.g. ctrl_r, f9, space")
    parser.add_argument("--toggle", action="store_true", help="tap the key once to start and once to stop")
    parser.add_argument("--hold", action="store_true", help="force push-to-talk (hold the key while speaking)")
    parser.add_argument("--no-tray", action="store_true", help="do not show the tray icon")
    parser.add_argument("--minimized", action="store_true", help="start with the overlay hidden (tray only)")
    parser.add_argument("--allow-multiple", action="store_true", help="do not refuse to start a second copy")
    parser.add_argument("--quit-hotkey", help="combo that exits, e.g. ctrl+alt+q")
    parser.add_argument("--output", choices=["paste", "type", "clipboard", "none"], help="where the text goes")
    parser.add_argument("--no-ui", action="store_true", help="run without the overlay")
    parser.add_argument("--no-partials", action="store_true", help="do not decode while speaking")
    parser.add_argument("--no-preload", action="store_true", help="load the model on first use instead of at start-up")

    parser.add_argument("--download", action="store_true", help="download the model and exit")
    parser.add_argument("--force-download", action="store_true", help="re-download even if the files look complete")
    parser.add_argument("--keep-zip", action="store_true", help="keep the download zip after extracting")
    parser.add_argument("--transcribe", metavar="WAV", help="transcribe a wav file and exit")
    parser.add_argument("--json", action="store_true", help="machine-readable output for --transcribe")
    parser.add_argument(
        "--paste-test",
        action="store_true",
        help="count down, then paste a Thai test string into the focused window",
    )
    parser.add_argument("--text", help="text used by --paste-test")
    parser.add_argument("--selftest", action="store_true", help="run the end-to-end checks")
    parser.add_argument("--doctor", action="store_true", help="check imports, model, microphone and output settings")
    parser.add_argument("--examples-dir", help="where --selftest looks for wav files")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    hotkey_mode = "toggle" if getattr(args, "toggle", False) else None
    if getattr(args, "hold", False):
        hotkey_mode = "hold"
    overrides = {
        "precision": args.precision,
        "device": args.device,
        "num_threads": args.threads,
        "lang": args.lang,
        "mode": args.mode,
        "hotkey": args.hotkey,
        "hotkey_mode": hotkey_mode,
        "quit_hotkey": args.quit_hotkey,
        "output": args.output,
        "partials": False if args.no_partials else None,
        "show_overlay": False if args.no_ui else None,
        "preload": False if args.no_preload else None,
        "tray": False if getattr(args, "no_tray", False) else None,
        "start_minimized": True if getattr(args, "minimized", False) else None,
        "single_instance": False if getattr(args, "allow_multiple", False) else None,
    }
    if args.model_dir:
        overrides["model_dir"] = args.model_dir
    return Config.load(args.config, overrides)


def _cmd_download(cfg: Config, args: argparse.Namespace) -> int:
    from .model_store import download_model, model_size_hint

    target = cfg.model_dir()
    print(f"precision={cfg['precision']} ({model_size_hint(str(cfg['precision']))}) -> {target}")
    if args.keep_zip:
        print("keeping the zip file (--keep-zip)")
    download_model(
        target,
        precision=str(cfg["precision"]),
        keep_zip=bool(args.keep_zip),
        force=bool(args.force_download),
    )
    return 0


def _cmd_transcribe(cfg: Config, args: argparse.Namespace) -> int:
    from .asr import OnnxAsr
    from .audio import AudioError, load_wav

    wav = Path(args.transcribe)
    try:
        audio = load_wav(wav)
    except AudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    asr = OnnxAsr(
        cfg.model_dir(),
        precision=str(cfg["precision"]),
        device=str(cfg["device"]),
        num_threads=int(cfg["num_threads"]) or None,
    )
    res = asr.transcribe_recording(
        audio,
        lang=str(cfg["lang"]),
        mode=str(cfg["mode"]),
        split_over_s=float(cfg["split_over_s"]),
        min_silence_s=float(cfg["min_silence_s"]),
    )
    if args.json:
        print(
            json.dumps(
                {
                    "file": str(wav),
                    "text": res.text,
                    "segments": res.segments,
                    "duration_s": round(res.duration_s, 3),
                    "bias": res.bias,
                    "attempts": res.attempts,
                    "encode_s": round(res.encode_s, 3),
                    "decode_s": round(res.decode_s, 3),
                    "rtf": round(res.rtf, 3),
                },
                ensure_ascii=False,
            )
        )
        return 0
    print(res.text)
    for seg in res.segments:
        print(f"  [{seg['start']:7.2f} - {seg['end']:7.2f}] {seg['text']}", file=sys.stderr)
    print(
        f"\n{res.duration_s:.2f}s audio in {len(res.segments) or 1} piece(s), "
        f"encode {res.encode_s:.2f}s + decode {res.decode_s:.2f}s "
        f"(rtf {res.rtf:.2f}), bias {res.bias:+.1f}, {res.attempts} attempt(s)",
        file=sys.stderr,
    )
    return 0


def _cmd_paste_test(cfg: Config, args: argparse.Namespace) -> int:
    """Check the output half of the app without involving the model."""
    import time

    from .output import TextOutput, clipboard_text, foreground_window_info

    sample = args.text or "ทดสอบการวางข้อความจาก ThaiASR ๑๒๓ abc"
    print(f"โหมด output = {cfg['output']}  (restore_clipboard = {cfg['restore_clipboard']})")
    print("คลิกไปที่ช่องข้อความปลายทาง (เช่น ช่องคอมเมนต์ Facebook) ภายใน 5 วินาที …")
    for remaining in range(5, 0, -1):
        print(f"  {remaining}…", flush=True)
        time.sleep(1.0)
    target = foreground_window_info()
    print(f"หน้าต่างเป้าหมาย: {target}")
    out = TextOutput(
        str(cfg["output"]),
        restore_clipboard=bool(cfg["restore_clipboard"]),
        paste_delay_ms=int(cfg["paste_delay_ms"]),
    )
    note = out.send(sample)
    print(f"ผลลัพธ์: {note}")
    print(f"ข้อความ: {sample}")
    print(f"คลิปบอร์ดตอนนี้: {clipboard_text()!r}")
    print(
        "\nถ้าข้อความไม่ขึ้นในช่องปลายทาง แต่คลิปบอร์ดมีข้อความอยู่ -> กด Ctrl+V เองได้เลย\n"
        "ถ้าอยากให้ลองพิมพ์แทนการวาง:  python -m thaiasr --paste-test --output type"
    )
    return 0


def _cmd_doctor(cfg: Config, args: argparse.Namespace) -> int:
    """Check every moving part, one step at a time (useful in the .exe too)."""
    import platform

    from .paths import app_home, is_frozen, log_path

    def say(text: str = "") -> None:
        print(text, flush=True)

    problems = 0
    say(f"thaiasr {__version__}")
    say(f"python      {platform.python_version()} ({platform.machine()})")
    say(f"frozen exe  {is_frozen()}")
    say(f"home        {app_home()}")
    say(f"config      {cfg.path or '(defaults)'}")
    say(f"log         {log_path()}")
    say(cfg.describe())

    say("\nimports:")
    for name in ("numpy", "onnxruntime", "sounddevice", "pynput", "pyperclip", "pystray", "PIL"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "")
            say(f"  ok    {name} {version}")
        except Exception as exc:  # noqa: BLE001 - report, keep going
            problems += 1
            say(f"  FAIL  {name}: {exc!r}")
            if name in ("numpy", "onnxruntime"):
                say("        (the model cannot run without this)")

    say("\nmodel:")
    from .model_store import model_size_hint, verify_model

    model_dir = cfg.model_dir()
    issues = verify_model(model_dir, str(cfg["precision"]))
    say(f"  {model_dir} [{cfg['precision']}, {model_size_hint(str(cfg['precision']))}]")
    if issues:
        problems += 1
        for issue in issues:
            say(f"  missing/bad: {issue}")
        say("  -> ใช้เมนู tray 'ดาวน์โหลดโมเดล' หรือ --download")
    else:
        say("  files look complete")

    say("\nmicrophone:")
    try:
        import sounddevice as sd

        devices = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        if not devices:
            problems += 1
            say("  no input device found")
        for device in devices[:4]:
            say(f"  [{device['index']}] {device['name']} ({device['max_input_channels']} ch)")
    except Exception as exc:  # noqa: BLE001
        problems += 1
        say(f"  FAIL: {exc!r}")

    say("\noutput:")
    from .output import foreground_window_info

    say(f"  mode={cfg['output']} restore_clipboard={cfg['restore_clipboard']}")
    say(f"  focused window now: {foreground_window_info()}")

    say(f"\n{problems} problem(s)" if problems else "\nno problems found")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(f"thaiasr {__version__}")
        return 0
    try:
        cfg = _config_from_args(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.paste_test:
        return _cmd_paste_test(cfg, args)
    if args.doctor:
        return _cmd_doctor(cfg, args)
    if args.download:
        return _cmd_download(cfg, args)
    if args.transcribe:
        return _cmd_transcribe(cfg, args)
    if args.selftest:
        from . import selftest

        return selftest.run_selftest(cfg, examples_dir=args.examples_dir)

    from . import runner

    return runner.run(cfg)


def _log_crash(exc: BaseException) -> Path:
    """Write a traceback to logs/thaiasr.log (the GUI runs without a console)."""
    import datetime
    import traceback

    from .paths import log_path

    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== {datetime.datetime.now().isoformat(timespec='seconds')} ===\n")
            fh.write("".join(traceback.format_exception(exc)))
    except Exception:  # noqa: BLE001 - logging must never raise
        pass
    return path


def _ensure_streams() -> None:
    """A windowed .exe has no stdout; keep ``print`` from crashing the app."""
    import io
    import os

    if sys.stdout is not None and sys.stderr is not None:
        return
    devnull = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - lives for the process
    if sys.stdout is None:
        sys.stdout = devnull
    if sys.stderr is None:
        sys.stderr = devnull
    sys.stdin = sys.stdin or io.StringIO()


def _configure_console() -> None:
    """Let Thai text reach the console.

    A frozen ``--console`` build starts with the legacy OEM code page (cp437),
    so ``--help`` and the Thai error messages used to die with
    ``UnicodeEncodeError``.  Switch our own streams to UTF-8 and put the console
    back the way we found it on exit.
    """
    import atexit
    import os

    try:
        for stream in (sys.stdout, sys.stderr):
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - a console is a nice-to-have
        pass

    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        previous = kernel32.GetConsoleOutputCP()
        if previous and previous != 65001:
            kernel32.SetConsoleOutputCP(65001)
            atexit.register(kernel32.SetConsoleOutputCP, previous)
    except Exception:  # noqa: BLE001 - no console attached
        pass



def entry() -> int:
    """``main`` with crash logging, used by ``python -m thaiasr`` and the .exe."""
    _ensure_streams()
    _configure_console()
    try:
        return main()
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - always leave a trace behind
        path = _log_crash(exc)
        try:
            print(f"ThaiASR ล้มเหลว: {exc}\nรายละเอียดอยู่ใน {path}", file=sys.stderr)
            _notify_failure(path)
        except Exception:  # noqa: BLE001
            pass
        return 1


def _notify_failure(path) -> None:
    """In a windowed build nobody sees stderr -> show a dialog."""
    from .paths import is_frozen

    if not is_frozen():
        return
    import ctypes

    ctypes.windll.user32.MessageBoxW(
        None,
        f"ThaiASR เปิดไม่สำเร็จ\n\nรายละเอียดอยู่ใน:\n{path}",
        "ThaiASR",
        0x10,  # MB_ICONERROR
    )


if __name__ == "__main__":
    raise SystemExit(entry())
