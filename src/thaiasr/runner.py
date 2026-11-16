"""Wiring: overlay + hotkeys + app.  ``python -m thaiasr`` ends up here."""

from __future__ import annotations

import os
import threading

from .app import DictationApp
from .config import Config, default_config_path, project_root
from .hotkey import HotkeyManager
from .paths import logs_dir
from .ui import Overlay

__all__ = ["run_gui", "run_headless", "run"]

_PRETTY = {
    "ctrl_r": "Right Ctrl",
    "ctrl_l": "Left Ctrl",
    "ctrl": "Ctrl",
    "alt_r": "Right Alt",
    "alt_l": "Left Alt",
    "alt": "Alt",
    "shift_r": "Right Shift",
    "shift_l": "Left Shift",
    "shift": "Shift",
    "cmd": "Win",
    "win": "Win",
    "space": "Space",
    "esc": "Esc",
    "enter": "Enter",
    "tab": "Tab",
    "backspace": "Backspace",
    "delete": "Delete",
    "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4", "f5": "F5", "f6": "F6",
    "f7": "F7", "f8": "F8", "f9": "F9", "f10": "F10", "f11": "F11", "f12": "F12",
}


def pretty_hotkey(spec: str) -> str:
    return " + ".join(_PRETTY.get(part.strip().lower(), part.strip().upper()) for part in spec.split("+"))


def _log(message: str) -> None:
    """Append a line to logs/thaiasr.log (the GUI has no console)."""
    import datetime

    from .paths import log_path

    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {message}\n")
    except Exception:  # noqa: BLE001 - logging must never break the app
        pass


def _make_manager(
    app: DictationApp, cfg: Config, quit_handler=None, overlay_handler=None
) -> HotkeyManager:
    """Build the hotkey listener; ``quit_handler`` must close the window too."""
    return HotkeyManager(
        hold=str(cfg["hotkey"]),
        hold_mode=str(cfg["hotkey_mode"]),
        quit_combo=str(cfg["quit_hotkey"]),
        cancel=str(cfg["cancel_hotkey"]),
        overlay_combo=str(cfg["overlay_hotkey"]),
        on_hold_start=lambda: app.post("start"),
        on_hold_stop=lambda: app.post("stop"),
        on_quit=lambda: (quit_handler or app.request_quit)(),
        on_cancel=lambda: app.post("cancel"),
        on_overlay_toggle=overlay_handler,
    )


def wire_overlay(app: DictationApp, overlay: Overlay):
    """Connect the overlay's close button to a *full* shutdown.

    Stopping the app alone leaves the Tk main loop running, so the window used to
    stay on screen forever -- which made the program look impossible to close.
    Returns the handler so the quit hotkey can use exactly the same path.
    """

    def request_quit(*_args) -> None:
        _log("quit requested")
        try:
            app.request_quit()
        finally:
            overlay.close()

    overlay.on_quit = request_quit
    return request_quit


def _open_folder(path) -> None:
    """Open a folder (or a file) with the desktop's default handler."""
    import os
    from pathlib import Path

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    try:
        os.startfile(str(target))  # noqa: S606 - Windows shell open
    except Exception:  # noqa: BLE001
        import subprocess

        subprocess.Popen(["explorer", str(target)])


def _open_config(cfg: Config) -> None:
    """Open config.toml in Notepad, creating it from the example if needed."""
    import subprocess
    from pathlib import Path

    target = cfg.path or Path(str(default_config_path()))
    if not target.exists():
        example = project_root() / "config.example.toml"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if example.exists():
                target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                target.write_text("", encoding="utf-8")
        except Exception:  # noqa: BLE001 - opening is still worth a try
            pass
    try:
        subprocess.Popen(["notepad.exe", str(target)])
    except Exception as exc:  # noqa: BLE001
        _log(f"open config failed: {exc}")


def _start_tray(
    app: DictationApp, cfg: Config, overlay: Overlay, quit_handler, manager, overlay_toggle=None
):
    """Tray icon + menu.  Returns ``None`` when pystray is unavailable."""
    from .model_store import model_ready
    from .tray import Tray, tray_available

    if not tray_available():
        _log("tray disabled: pystray/pillow is not installed")
        return None

    holder: dict = {}
    status_holder = {"text": ""}

    def set_mode(mode: str) -> None:
        if manager is not None:
            manager.hold_mode = mode
        cfg.data["hotkey_mode"] = mode
        overlay.set_hotkey_mode(mode)
        try:
            cfg.save_setting("hotkey_mode", mode)
        except Exception as exc:  # noqa: BLE001 - runtime change still applies
            _log(f"could not save hotkey_mode: {exc}")
        _log(f"hotkey mode -> {mode}")

    def download() -> None:
        if model_ready(cfg.model_dir(), str(cfg["precision"])):
            tray = holder.get("tray")
            if tray:
                tray.notify("โมเดลพร้อมอยู่แล้ว")
            return
        app.download_model_async(
            on_done=lambda: (holder["tray"].refresh(), holder["tray"].notify("โมเดลพร้อมแล้ว")),
            on_error=lambda exc: holder["tray"].notify(f"ดาวน์โหลดไม่สำเร็จ: {exc}"),
        )

    tray = Tray(
        on_toggle_recording=app.toggle_recording,
        on_quit=quit_handler,
        on_toggle_overlay=(
            overlay_toggle
            if overlay_toggle is not None
            else (lambda: overlay.set_visible(not overlay.visible))
        ),
        on_download_model=download,
        on_open_logs=lambda: _open_folder(logs_dir()),
        on_open_config=lambda: _open_config(cfg),
        on_set_mode=set_mode,
        is_recording=lambda: app.state == DictationApp.RECORDING,
        overlay_visible=lambda: overlay.visible,
        hold_mode=lambda: str(cfg["hotkey_mode"]),
        model_ready=lambda: model_ready(cfg.model_dir(), str(cfg["precision"])),
        status=lambda: status_holder["text"],
    )
    holder["tray"] = tray
    if tray.start():
        _log("tray icon started")
        return tray
    _log("tray icon failed to start")
    return None


def run_gui(cfg: Config) -> int:
    """Overlay UI, dictation into whichever window has focus."""
    overlay = Overlay(
        cfg,
        hotkey_label=pretty_hotkey(str(cfg["hotkey"])),
        quit_hotkey_label=pretty_hotkey(str(cfg["quit_hotkey"])),
        hotkey_mode=str(cfg["hotkey_mode"]),
    )
    status_holder = {"text": "พร้อมใช้งาน"}
    tray_holder: dict = {}

    def on_status(text: str, kind: str = "info", value=None) -> None:
        if kind != "level":
            status_holder["text"] = text or status_holder["text"]
        if kind == "level":
            overlay.set_level(float(value or 0.0))
        else:
            tray = tray_holder.get("tray")
            if tray is not None:
                tray.refresh()
        if kind == "recording":
            overlay.set_recording(True if value is None else bool(value))
            if text:
                overlay.set_status(text, "#ff9a9a")
        elif kind == "partial":
            overlay.set_text(text)
        elif kind == "final":
            overlay.set_recording(False)
            overlay.set_text(text, final=True)
            overlay.set_status("วางแล้ว • ข้อความอยู่ในคลิปบอร์ดด้วย", "#8ef0c8")
        elif kind == "busy":
            overlay.set_recording(False)
            overlay.set_status(text, "#ffc46b")
        elif kind == "error":
            overlay.set_recording(False)
            overlay.set_error(text)
            overlay.set_status("ผิดพลาด", "#ffb4b4")
        elif kind == "warn":
            overlay.set_recording(False)
            overlay.set_text(text)
        elif kind == "ok":
            overlay.set_status(text, "#8ef0c8")
        elif text:
            overlay.set_status(text, "#c9d4e0")

    def on_state_change(state: str) -> None:
        if manager is not None:
            # Keep the toggle tap in sync when the app stops on its own.
            manager.sync_active(state == DictationApp.RECORDING)
        tray = tray_holder.get("tray")
        if tray is not None:
            tray.refresh()

    def toggle_overlay(*_args) -> bool:
        """'— ย่อ' button, the tray item and Ctrl+Alt+H all land here."""
        shown = overlay.toggle_visible()
        tray = tray_holder.get("tray")
        if tray is not None:
            tray.refresh()
            if not shown:
                tray.notify(
                    f"ย่อ overlay แล้ว — กด {pretty_hotkey(str(cfg['overlay_hotkey']))} "
                    "หรือคลิกขวาที่ไอคอนนี้เพื่อเปิดกลับ"
                )
        _log(f"overlay {'shown' if shown else 'minimized'}")
        return shown

    app = DictationApp(cfg, on_status=on_status, on_state_change=on_state_change)
    quit_handler = wire_overlay(app, overlay)
    overlay.on_minimize = toggle_overlay

    manager: HotkeyManager | None = None
    try:
        manager = _make_manager(app, cfg, quit_handler, toggle_overlay)
        manager.start()
        _log(
            f"started pid={os.getpid()} hold={cfg['hotkey']} mode={cfg['hotkey_mode']} "
            f"quit={cfg['quit_hotkey']} overlay={cfg['overlay_hotkey']}"
        )
    except Exception as exc:  # noqa: BLE001 - report instead of vanishing
        _log(f"hotkey listener failed: {exc}")
        overlay.set_error(f"ใช้ปุ่มลัดไม่ได้: {exc}")
        overlay.set_status("ปุ่มลัดใช้ไม่ได้", "#ffb4b4")

    if bool(cfg["tray"]):
        tray = _start_tray(app, cfg, overlay, quit_handler, manager, toggle_overlay)
        if tray is not None:
            tray_holder["tray"] = tray

    worker = threading.Thread(target=app.run, name="thaiasr-app", daemon=True)
    worker.start()
    try:
        overlay.run()
    except KeyboardInterrupt:
        pass
    finally:
        _log("shutting down")
        app.stop()
        if manager is not None:
            manager.stop()
        tray = tray_holder.get("tray")
        if tray is not None:
            tray.stop()
        worker.join(timeout=2.0)
    return 0


def run_headless(cfg: Config) -> int:
    """Same dictation loop without the overlay (debugging, scripts)."""

    def on_status(text: str, kind: str = "info", value=None) -> None:
        if kind == "level":
            return
        if text:
            print(f"[{kind}] {text}", flush=True)

    app = DictationApp(cfg, on_status=on_status)
    try:
        manager = _make_manager(app, cfg)
    except Exception as exc:  # noqa: BLE001 - config problem, not a crash
        print(f"error: {exc}", flush=True)
        return 2
    manager.start()
    print(
        f"ThaiASR: กดค้าง {pretty_hotkey(str(cfg['hotkey']))} เพื่อพูด "
        f"(ออก: {pretty_hotkey(str(cfg['quit_hotkey']))})",
        flush=True,
    )
    try:
        app.run(block=True)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        manager.stop()
    return 0


def run(cfg: Config) -> int:
    """Run the app, refusing to start a second copy of itself."""
    guard = None
    if bool(cfg["single_instance"]):
        from .instance import InstanceGuard, already_running_message

        guard = InstanceGuard()
        if not guard.acquire():
            _log("another instance is already running - exiting")
            already_running_message()
            return 0
    try:
        return run_gui(cfg) if bool(cfg["show_overlay"]) else run_headless(cfg)
    finally:
        if guard is not None:
            guard.release()
