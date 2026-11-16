"""The on-screen overlay: a small always-on-top, never-focused tkinter window.

``WS_EX_NOACTIVATE`` matters: if the overlay took focus, the Ctrl+V that pastes
the text would land in the overlay instead of the app the user was typing into.
All methods are safe to call from other threads; they post to the Tk thread.
"""

from __future__ import annotations

import ctypes
import queue
import tkinter as tk
from tkinter import font as tkfont

__all__ = ["Overlay"]

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080

IDLE_HINT = "กดค้าง Right Ctrl แล้วพูด  •  ปล่อยเพื่อวางข้อความ"
RECORDING_HINT = "กำลังฟัง… ปล่อย Right Ctrl เมื่อพูดจบ  •  Esc = ยกเลิก"
IDLE_HINT_TOGGLE = "กด Right Ctrl ครั้งเดียวเพื่อเริ่มพูด"
RECORDING_HINT_TOGGLE = "กำลังฟัง… กด Right Ctrl อีกครั้งเมื่อพูดจบ  •  Esc = ยกเลิก"


class Overlay:
    def __init__(
        self,
        config,
        *,
        hotkey_label: str = "Right Ctrl",
        quit_hotkey_label: str = "Ctrl+Alt+Q",
        hotkey_mode: str = "hold",
    ):
        self.config = config
        self.hotkey_label = hotkey_label
        self.quit_hotkey_label = quit_hotkey_label
        self.hotkey_mode = hotkey_mode
        self._queue: queue.Queue = queue.Queue()
        self._root: tk.Tk | None = None
        self._hidden_reset = None
        self._visible = True

    # ------------------------------------------------------------------ hints
    def _hint_text(self, recording: bool) -> str:
        if self.hotkey_mode == "toggle":
            base = RECORDING_HINT_TOGGLE if recording else IDLE_HINT_TOGGLE
        else:
            base = RECORDING_HINT if recording else IDLE_HINT
        text = base.replace("Right Ctrl", self.hotkey_label)
        return text if recording else text + f"   •   ออก: {self.quit_hotkey_label}"

    def set_hotkey_mode(self, mode: str) -> None:
        """Switch hold/toggle at run time (tray menu) and refresh the hint."""
        self.hotkey_mode = mode

        def apply():
            self._hint.configure(text=self._hint_text(False))

        self.call(apply)

    # ---------------------------------------------------------------- plumbing
    def call(self, fn, *args, **kwargs) -> None:
        """Schedule ``fn`` on the Tk thread (safe from any thread)."""
        self._queue.put((fn, args, kwargs))

    def _drain(self) -> None:
        root = self._root
        try:
            while True:
                fn, args, kwargs = self._queue.get_nowait()
                try:
                    fn(*args, **kwargs)
                except Exception:  # noqa: BLE001 - a broken update must not kill the UI
                    import traceback

                    traceback.print_exc()
        except queue.Empty:
            pass
        if root is not None:
            try:
                root.after(40, self._drain)
            except tk.TclError:  # the window was destroyed: nothing left to poll
                self._root = None

    # ------------------------------------------------------------------- build
    def _build(self) -> tk.Tk:
        cfg = self.config
        width = int(cfg.get("overlay.width", 560))
        height = int(cfg.get("overlay.height", 168))

        root = tk.Tk()
        root.title("ThaiASR")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", float(cfg.get("overlay.opacity", 0.94)))
        except Exception:  # noqa: BLE001 - alpha is optional
            pass
        root.configure(bg="#101418")

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        margin = int(cfg.get("overlay.bottom_margin", 90))
        x = (screen_w - width) // 2
        y = max(0, screen_h - height - margin)
        root.geometry(f"{width}x{height}+{x}+{y}")

        self._width = width
        family = self._pick_font(root)
        self._f_status = tkfont.Font(family=family, size=10, weight="bold")
        self._f_text = tkfont.Font(family=family, size=13)
        self._f_small = tkfont.Font(family=family, size=9)

        header = tk.Frame(root, bg="#101418")
        header.pack(fill="x", padx=12, pady=(10, 0))

        self._dot = tk.Canvas(header, width=14, height=14, bg="#101418", highlightthickness=0)
        self._dot.pack(side="left", pady=1)
        self._dot_id = self._dot.create_oval(3, 3, 13, 13, fill="#5a6672", outline="")

        self._status = tk.Label(header, text="พร้อมใช้งาน", font=self._f_status, fg="#c9d4e0", bg="#101418")
        self._status.pack(side="left", padx=(6, 0))

        self._quit = tk.Label(
            header,
            text="✕ ปิด",
            font=self._f_status,
            fg="#9aa6b2",
            bg="#1b2027",
            cursor="hand2",
            padx=8,
            pady=1,
        )
        self._quit.pack(side="right")
        self._quit.bind("<Button-1>", lambda _e: self._on_quit())

        self._min = tk.Label(
            header,
            text="— ย่อ",
            font=self._f_status,
            fg="#9aa6b2",
            bg="#1b2027",
            cursor="hand2",
            padx=8,
            pady=1,
        )
        self._min.pack(side="right", padx=(0, 6))
        self._min.bind("<Button-1>", lambda _e: self._on_minimize())

        self._level = tk.Canvas(root, height=6, bg="#1b2027", highlightthickness=0)
        self._level.pack(fill="x", padx=12, pady=(8, 0))
        self._level_bar = self._level.create_rectangle(0, 0, 0, 6, fill="#25c2a0", outline="")

        self._text = tk.Label(
            root,
            text="",
            font=self._f_text,
            fg="#f2f6fa",
            bg="#101418",
            justify="left",
            anchor="w",
            wraplength=width - 24,
        )
        self._text.pack(fill="both", expand=True, padx=12, pady=(8, 0))

        self._hint = tk.Label(
            root,
            text=self._hint_text(False),
            font=self._f_small,
            fg="#66727f",
            bg="#101418",
            anchor="w",
        )
        self._hint.pack(fill="x", padx=12, pady=(0, 10))

        self._make_unfocusable(root)
        self._make_draggable(root, header)
        root.bind("<Escape>", lambda _e: None)
        return root

    @staticmethod
    def _pick_font(root: tk.Tk) -> str:
        available = set(tkfont.families(root))
        for name in ("Leelawadee UI", "Tahoma", "Segoe UI", "Noto Sans Thai"):
            if name in available:
                return name
        return "TkDefaultFont"

    @staticmethod
    def _make_unfocusable(root: tk.Tk) -> None:
        root.update_idletasks()
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        except Exception:  # noqa: BLE001 - non-Windows / restricted session
            pass

    def _make_draggable(self, root: tk.Tk, handle: tk.Widget) -> None:
        state = {"x": 0, "y": 0}

        def press(event):
            state["x"], state["y"] = event.x, event.y

        def drag(event):
            root.geometry(f"+{root.winfo_x() + event.x - state['x']}+{root.winfo_y() + event.y - state['y']}")

        handle.bind("<Button-1>", press)
        handle.bind("<B1-Motion>", drag)

    # ------------------------------------------------------------- public API
    def _on_quit(self) -> None:
        callback = getattr(self, "on_quit", None)
        if callback:
            callback()

    def _on_minimize(self) -> None:
        """The '— ย่อ' button: hide the overlay, keep working in the background."""
        callback = getattr(self, "on_minimize", None)
        if callback:
            callback()
        else:
            self.set_visible(False)

    def set_status(self, text: str, color: str = "#c9d4e0") -> None:
        def apply():
            self._status.configure(text=text, fg=color)

        self.call(apply)

    def set_recording(self, recording: bool) -> None:
        def apply():
            self._dot.itemconfigure(self._dot_id, fill="#ff5b5b" if recording else "#5a6672")
            self._hint.configure(text=self._hint_text(recording))

        self.call(apply)

    def set_level(self, value: float) -> None:
        """``value`` is RMS (0..1); drawn on a log-ish scale."""
        width = getattr(self, "_width", 560) - 24

        def apply():
            import math

            scaled = min(1.0, max(0.0, math.sqrt(max(0.0, value)) * 3.0))
            self._level.coords(self._level_bar, 0, 0, width * scaled, 6)

        self.call(apply)

    def set_text(self, text: str, *, final: bool = False) -> None:
        def apply():
            if self._hidden_reset is not None:
                try:
                    self._root.after_cancel(self._hidden_reset)
                except Exception:  # noqa: BLE001
                    pass
                self._hidden_reset = None
            self._text.configure(text=text, fg="#8ef0c8" if final else "#f2f6fa")
            if final and self.config.get("overlay.hide_after_s"):
                delay = int(float(self.config["overlay.hide_after_s"]) * 1000)

                def clear():
                    self._text.configure(text="", fg="#f2f6fa")
                    self._hidden_reset = None

                self._hidden_reset = self._root.after(delay, clear)

        self.call(apply)

    def set_error(self, message: str) -> None:
        def apply():
            self._text.configure(text=message, fg="#ffb4b4")

        self.call(apply)

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        """Create the window and block in the Tk main loop."""
        self._root = self._build()
        self._root.after(40, self._drain)
        if bool(self.config.get("start_minimized", False)):
            self._visible = False
            self._root.withdraw()
        self._root.mainloop()

    def close(self) -> None:
        root = self._root
        if root is not None:
            self.call(root.destroy)

    @property
    def visible(self) -> bool:
        return self._visible

    def set_visible(self, visible: bool) -> None:
        """Show/hide the overlay without stopping the app (tray menu)."""
        self._visible = bool(visible)

        def apply():
            root = self._root
            if root is None:
                return
            if self._visible:
                root.deiconify()
                root.attributes("-topmost", True)
            else:
                root.withdraw()

        self.call(apply)

    def toggle_visible(self) -> bool:
        self.set_visible(not self._visible)
        return self._visible
