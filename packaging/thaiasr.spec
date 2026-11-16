# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: builds dist/ThaiASR/{ThaiASR.exe, ThaiASR-cli.exe}.

Build with::

    .venv\\Scripts\\python.exe -m PyInstaller packaging\\thaiasr.spec --noconfirm --clean

The 2.6 GB model is deliberately NOT bundled: it is downloaded on first run into
%LOCALAPPDATA%\\ThaiASR (see src/thaiasr/paths.py).
"""

from pathlib import Path
import os

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"

# sounddevice loads PortAudio with ctypes from its own data folder.
datas = collect_data_files("sounddevice")

hiddenimports = [
    "pystray._win32",
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    "PIL.Image",
    "PIL.ImageDraw",
]

excludes = [
    "torch",
    "nemo",
    "librosa",
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "pytest",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "setuptools",
]

a = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "rthook_onnxruntime.py")],
    excludes=excludes,
    noarchive=False,
)

# Never trust the Visual C++ runtime that dependency scanning picked up: it can
# come from any directory on PATH (on this machine an old JRE in Program Files
# holds MSVCP140 14.27) and onnxruntime then crashes with 0xC0000005 inside it.
# Use the current copies from System32, or drop them and let Windows find them.
_SYSTEM32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
_RUNTIME_DLLS = {
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
}
_fixed_binaries = []
for _entry in a.binaries:
    # TOC entries are (dest_name, src_name, typecode)
    _dest, _src, _kind = _entry
    _name = Path(_dest).name
    if _name.lower() in _RUNTIME_DLLS:
        _candidate = _SYSTEM32 / _name
        if _candidate.exists():
            _fixed_binaries.append((_dest, str(_candidate), _kind))
            print(f"spec: {_name} -> {_candidate}")
        else:
            print(f"spec: dropped {_name} (not found in System32)")
        continue
    _fixed_binaries.append(_entry)
a.binaries = _fixed_binaries

pyz = PYZ(a.pure)

exe_gui = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ThaiASR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(ROOT / "packaging" / "thaiasr.ico"),
)

exe_cli = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ThaiASR-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=str(ROOT / "packaging" / "thaiasr.ico"),
)

coll = COLLECT(
    exe_gui,
    exe_cli,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ThaiASR",
)
