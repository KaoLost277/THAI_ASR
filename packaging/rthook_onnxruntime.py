"""Runtime hook: make onnxruntime's own DLL folder visible to the Windows loader.

``onnxruntime.dll`` lives in ``_internal/onnxruntime/capi`` and pulls in
``onnxruntime_providers_shared.dll`` *by name*.  PyInstaller only puts
``_internal`` itself on the DLL search path, so in a frozen build that second
DLL is not found and the import dies with an access violation (0xC0000005).
Adding the folder explicitly fixes it.
"""

import os
import sys

if getattr(sys, "frozen", False):
    _base = getattr(sys, "_MEIPASS", "")
    _dirs = [
        os.path.join(_base, "onnxruntime", "capi"),
        os.path.join(_base, "numpy.libs"),
    ]
    for _path in _dirs:
        if os.path.isdir(_path):
            try:
                os.add_dll_directory(_path)
            except Exception:  # noqa: BLE001 - older Python or no such API
                pass
    _env = os.pathsep.join([p for p in _dirs if os.path.isdir(p)])
    if _env:
        os.environ["PATH"] = _env + os.pathsep + os.environ.get("PATH", "")
