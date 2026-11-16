"""Generate ``packaging/thaiasr.ico`` (the .exe icon) from the tray drawing.

Run once; the result is committed so a build needs no extra steps::

    python packaging/make_icon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thaiasr.tray import _icon_image  # noqa: E402

SIZES = (16, 24, 32, 48, 64, 128, 256)


def main() -> int:
    target = ROOT / "packaging" / "thaiasr.ico"
    base = _icon_image(False).resize((256, 256))
    base.save(target, format="ICO", sizes=[(s, s) for s in SIZES])
    print(f"wrote {target} ({target.stat().st_size} bytes, sizes {SIZES})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
