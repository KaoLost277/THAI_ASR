"""System-tray icon: the app is reachable even when the overlay is hidden.

The tray runs its own thread (pystray owns a Windows message loop there) while
tkinter keeps the main thread, so the two never fight.  Everything the menu does
goes through the same handlers the hotkeys use.

``pystray``/``Pillow`` are optional: if they are missing the app logs it and
carries on with the overlay only.
"""

from __future__ import annotations

import threading

__all__ = ["Tray", "tray_available"]


def tray_available() -> bool:
    try:
        import pystray  # noqa: F401
        from PIL import Image, ImageDraw  # noqa: F401
    except Exception:  # noqa: BLE001 - optional dependency
        return False
    return True


def _icon_image(recording: bool):
    """Draw the tray icon (a microphone) so no image file is needed.

    A *filled* circle on purpose: the taskbar is dark, so a dark icon with a thin
    ring all but disappears once Windows scales it down to 16x16.
    """
    from PIL import Image, ImageDraw

    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    body = (255, 91, 91, 255) if recording else (37, 194, 160, 255)
    draw.ellipse((0, 0, size - 1, size - 1), fill=body)
    draw.rounded_rectangle((24, 13, 40, 37), radius=8, fill=(255, 255, 255, 255))
    draw.arc((16, 21, 48, 48), start=0, end=180, fill=(255, 255, 255, 255), width=5)
    draw.line((32, 47, 32, 54), fill=(255, 255, 255, 255), width=4)
    draw.line((23, 54, 41, 54), fill=(255, 255, 255, 255), width=4)
    return img


class Tray:
    """Menu + icon; ``start()`` is non-blocking."""

    def __init__(
        self,
        *,
        on_toggle_recording=lambda: None,
        on_quit=lambda: None,
        on_toggle_overlay=lambda: None,
        on_download_model=lambda: None,
        on_open_logs=lambda: None,
        on_open_config=lambda: None,
        on_set_mode=lambda mode: None,
        is_recording=lambda: False,
        overlay_visible=lambda: True,
        hold_mode=lambda: "hold",
        model_ready=lambda: True,
        status=lambda: "",
    ):
        self._on_toggle_recording = on_toggle_recording
        self._on_quit = on_quit
        self._on_toggle_overlay = on_toggle_overlay
        self._on_download_model = on_download_model
        self._on_open_logs = on_open_logs
        self._on_open_config = on_open_config
        self._on_set_mode = on_set_mode
        self._is_recording = is_recording
        self._overlay_visible = overlay_visible
        self._hold_mode = hold_mode
        self._model_ready = model_ready
        self._status = status
        self._icon = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    # ------------------------------------------------------------------ build
    def _build(self):
        import pystray

        def wrap(fn):
            def handler(icon=None, item=None):
                try:
                    fn()
                except Exception:  # noqa: BLE001 - a menu click must not kill the tray
                    import traceback

                    traceback.print_exc()

            return handler

        def record_label(_item):
            return "หยุดพูด / วางข้อความ" if self._is_recording() else "เริ่มพูด"

        def overlay_label(_item):
            return "ซ่อน overlay" if self._overlay_visible() else "แสดง overlay"

        def download_label(_item):
            return "ดาวน์โหลดโมเดล" if not self._model_ready() else "โมเดลพร้อมแล้ว"

        menu = pystray.Menu(
            pystray.MenuItem(record_label, wrap(self._on_toggle_recording), default=True),
            pystray.MenuItem(overlay_label, wrap(self._on_toggle_overlay)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "โหมดปุ่มลัด",
                pystray.Menu(
                    pystray.MenuItem(
                        "กดค้าง (hold)",
                        wrap(lambda: self._set_mode("hold")),
                        checked=lambda _i: self._hold_mode() == "hold",
                        radio=True,
                    ),
                    pystray.MenuItem(
                        "กดครั้งเดียว (toggle)",
                        wrap(lambda: self._set_mode("toggle")),
                        checked=lambda _i: self._hold_mode() == "toggle",
                        radio=True,
                    ),
                ),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(download_label, wrap(self._on_download_model)),
            pystray.MenuItem("เปิดโฟลเดอร์ log", wrap(self._on_open_logs)),
            pystray.MenuItem("เปิด config.toml", wrap(self._on_open_config)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("ออก (Ctrl+Alt+Q)", wrap(self._on_quit)),
        )
        icon = pystray.Icon(
            "ThaiASR", icon=_icon_image(False), title="ThaiASR", menu=menu
        )
        return icon

    def _set_mode(self, mode: str) -> None:
        self._on_set_mode(mode)
        self.refresh()

    # ------------------------------------------------------------- public API
    def start(self) -> bool:
        """Return False when the tray could not be created."""
        if not tray_available():
            return False
        try:
            self._icon = self._build()
        except Exception:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            return False

        def run():
            def setup(icon):
                # pystray starts hidden: without this the icon never appears.
                icon.visible = True
                self._ready.set()

            try:
                self._icon.run(setup=setup)
            except Exception:  # noqa: BLE001 - tray is a nicety, never fatal
                import traceback

                traceback.print_exc()

        self._thread = threading.Thread(target=run, name="thaiasr-tray", daemon=True)
        self._thread.start()
        # ``setup`` fires inside pystray's own loop; don't block the caller for long.
        self._ready.wait(timeout=3.0)
        return True

    def refresh(self) -> None:
        """Re-read the dynamic labels and switch the recording icon."""
        icon = self._icon
        if icon is None:
            return
        try:
            if not icon.visible:
                icon.visible = True
            icon.icon = _icon_image(bool(self._is_recording()))
            icon.title = f"ThaiASR — {self._status()}" if self._status() else "ThaiASR"
            icon.update_menu()
        except Exception:  # noqa: BLE001
            pass

    def notify(self, message: str, title: str = "ThaiASR") -> None:
        icon = self._icon
        if icon is None:
            return
        try:
            icon.notify(message, title)
        except Exception:  # noqa: BLE001 - notifications are optional
            pass

    def stop(self) -> None:
        icon, self._icon = self._icon, None
        if icon is not None:
            try:
                icon.stop()
            except Exception:  # noqa: BLE001
                pass
