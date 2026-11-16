"""Show the overlay for a few seconds without recording anything.

    python tools/smoke_ui.py [--seconds 3]

Handy for checking the Thai font, the position and that the window never takes
focus from the app underneath.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thaiasr.config import Config  # noqa: E402
from thaiasr.runner import pretty_hotkey  # noqa: E402
from thaiasr.ui import Overlay  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="overlay smoke test")
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args(argv)

    cfg = Config.load(None)
    overlay = Overlay(
        cfg,
        hotkey_label=pretty_hotkey(str(cfg["hotkey"])),
        quit_hotkey_label=pretty_hotkey(str(cfg["quit_hotkey"])),
    )

    def script():
        time.sleep(0.4)
        overlay.set_status("กำลังฟัง…", "#ff9a9a")
        overlay.set_recording(True)
        overlay.set_level(0.08)
        overlay.set_text("กำลังถอดเสียง… สวัสดีครับ นี่คือตัวอย่างข้อความ")
        time.sleep(1.0)
        overlay.set_recording(False)
        overlay.set_level(0.0)
        overlay.set_text("สวัสดีครับ นี่คือข้อความสุดท้ายที่ถอดได้", final=True)
        overlay.set_status("วางข้อความแล้ว", "#8ef0c8")
        time.sleep(max(0.2, args.seconds - 1.4))
        overlay.close()

    threading.Thread(target=script, daemon=True).start()
    overlay.run()
    print("overlay opened and closed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
