"""Entry point for the packaged .exe files (PyInstaller).

Two executables share this module:

* ``ThaiASR.exe``     -- no console: the everyday dictation app (tray + overlay).
* ``ThaiASR-cli.exe`` -- console: ``--download``, ``--transcribe``, ``--selftest``.

The ONNX model is *not* bundled (it is 2.6 GB).  On first run the app offers to
download it into ``%LOCALAPPDATA%\\ThaiASR\\models``.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    multiprocessing.freeze_support()
    from thaiasr.__main__ import entry

    return entry()


if __name__ == "__main__":
    sys.exit(main())
