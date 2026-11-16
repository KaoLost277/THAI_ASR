"""Global hotkeys with pynput.

Two kinds of binding are supported:

* **hold** (push-to-talk): ``"ctrl_r"`` -- the app starts recording while the key
  is down and stops when it comes back up.
* **toggle**: same key, but a tap starts recording and the next tap stops it
  (``hold_mode="toggle"``).
* **combo**: ``"ctrl+alt+q"`` -- all parts must be down at the same time.

Accepted part names: ``ctrl``/``ctrl_l``/``ctrl_r``, ``alt``/``alt_l``/``alt_r``,
``shift``/``shift_l``/``shift_r``, ``cmd``/``win``, ``space``, ``tab``, ``enter``,
``esc``, ``backspace``, ``delete``, ``insert``, ``home``, ``end``, ``page_up``,
``page_down``, ``up``/``down``/``left``/``right``, ``f1``..``f24``, or a single
character such as ``q``.

Besides push-to-talk there are three optional combos: quit, cancel and
show/hide the overlay (``overlay_combo``).

Callbacks run on the pynput listener thread, so they must be cheap and
thread-safe (the app posts events onto a queue).
"""

from __future__ import annotations

import threading

from pynput import keyboard

__all__ = ["HotkeyManager", "parse_spec", "HotkeyError"]


class HotkeyError(ValueError):
    """Raised for an unparsable hotkey spec."""


_MODIFIERS = {
    "ctrl": ("ctrl_l", "ctrl_r"),
    "control": ("ctrl_l", "ctrl_r"),
    "alt": ("alt_l", "alt_r", "alt_gr"),
    "shift": ("shift", "shift_l", "shift_r"),
    "cmd": ("cmd", "cmd_l", "cmd_r"),
    "win": ("cmd", "cmd_l", "cmd_r"),
    "super": ("cmd", "cmd_l", "cmd_r"),
}
_NAMED = {
    "space", "tab", "enter", "return", "esc", "escape", "backspace", "delete",
    "insert", "home", "end", "page_up", "page_down", "up", "down", "left", "right",
    "caps_lock", "num_lock", "scroll_lock", "print_screen", "pause", "menu",
}
_SPECIFIC = {
    "ctrl_l", "ctrl_r", "alt_l", "alt_r", "alt_gr", "shift_l", "shift_r",
    "cmd_l", "cmd_r",
}


def _canonical(part: str) -> str:
    p = part.strip().lower().replace("control", "ctrl")
    if p in ("return",):
        p = "enter"
    elif p in ("escape",):
        p = "esc"
    return p


def parse_spec(spec: str) -> tuple[str, ...]:
    """Split ``"ctrl+alt+q"`` into canonical part names."""
    if not spec or not spec.strip():
        raise HotkeyError("empty hotkey spec")
    parts = [_canonical(p) for p in spec.split("+")]
    for part in parts:
        if (
            part in _MODIFIERS
            or part in _NAMED
            or part in _SPECIFIC
            or (len(part) == 1 and part.isprintable())
            or (part.startswith("f") and part[1:].isdigit() and 1 <= int(part[1:]) <= 24)
        ):
            continue
        raise HotkeyError(f"unsupported hotkey part: {part!r} (in {spec!r})")
    return tuple(dict.fromkeys(parts))


def _names_for(key) -> set[str]:
    """Every name a physical key answers to.

    A character key is matched by *virtual key code* as well as by character:
    with a Thai (or any non-Latin) keyboard layout the character pynput reports
    for the physical ``Q`` key is not ``"q"``, so ``ctrl+alt+q`` would otherwise
    never fire.
    """
    names: set[str] = set()
    if isinstance(key, keyboard.KeyCode):
        if key.char:
            names.add(key.char.lower())
        if key.vk is not None:
            names.add(f"vk{key.vk}")
            if 0x41 <= key.vk <= 0x5A:  # A-Z
                names.add(chr(key.vk).lower())
            elif 0x30 <= key.vk <= 0x39:  # 0-9
                names.add(chr(key.vk))
        return names
    name = getattr(key, "name", None)
    if name is None:
        return names
    names.add(name)
    for generic, members in _MODIFIERS.items():
        if name in members and generic not in _NAMED:
            names.add(generic)
    return names


class HotkeyManager:
    """Owns one pynput listener and dispatches press/release callbacks."""

    def __init__(
        self,
        *,
        hold: str | None = None,
        hold_mode: str = "hold",
        quit_combo: str | None = None,
        cancel: str | None = None,
        overlay_combo: str | None = None,
        on_hold_start=None,
        on_hold_stop=None,
        on_quit=None,
        on_cancel=None,
        on_overlay_toggle=None,
    ):
        self.hold = parse_spec(hold) if hold else None
        if hold_mode not in ("hold", "toggle"):
            raise HotkeyError("hold_mode must be 'hold' or 'toggle'")
        self.hold_mode = hold_mode
        self.quit_combo = parse_spec(quit_combo) if quit_combo else None
        self.cancel = parse_spec(cancel) if cancel else None
        self.overlay_combo = parse_spec(overlay_combo) if overlay_combo else None
        if self.hold and len(self.hold) > 1:
            raise HotkeyError(
                "the push-to-talk key must be a single key, e.g. 'ctrl_r' or 'f9'"
            )
        self._on_hold_start = on_hold_start
        self._on_hold_stop = on_hold_stop
        self._on_quit = on_quit
        self._on_cancel = on_cancel
        self._on_overlay_toggle = on_overlay_toggle
        self._pressed: set[str] = set()
        self._hold_active = False
        self._lock = threading.Lock()
        self._listener: keyboard.Listener | None = None

    # ------------------------------------------------------------------ state
    @property
    def hold_active(self) -> bool:
        return self._hold_active

    def sync_active(self, active: bool) -> None:
        """Match the manager's tap state to what the app is really doing.

        In toggle mode the app can stop recording on its own (Esc, the
        ``max_utterance_s`` limit, a tray click); without this the next tap
        would "stop" an already stopped recording.
        """
        with self._lock:
            self._hold_active = bool(active)

    def _fire(self, callback, *args) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:  # noqa: BLE001 - never let the listener thread die
            import traceback

            traceback.print_exc()

    def _on_press(self, key) -> None:
        names = _names_for(key)
        with self._lock:
            already = names & self._pressed  # ignore auto-repeat
            self._pressed |= names
            pressed = set(self._pressed)
            hold_names = set(self.hold) if self.hold else set()
            # Only a *fresh press of the talk key itself* counts as a tap: with
            # another key held down the talk key would otherwise look "pressed"
            # again on every unrelated keystroke.
            hold_fresh = (
                bool(hold_names) and bool(names & hold_names) and not (names & already)
            )
            start = stop = False
            if hold_names and hold_names <= pressed and hold_fresh:
                if self.hold_mode == "toggle":
                    # A tap flips recording on/off; the release is ignored.
                    if self._hold_active:
                        self._hold_active = False
                        stop = True
                    else:
                        self._hold_active = True
                        start = True
                elif not self._hold_active:
                    self._hold_active = True
                    start = True
            # A combo/cancel only fires on the press that completes it -- not on
            # an unrelated key that happens to arrive while its keys are held.
            fresh = not (names & already)
            quit_now = (
                bool(self.quit_combo)
                and set(self.quit_combo) <= pressed
                and bool(names & set(self.quit_combo))
                and fresh
            )
            overlay_now = (
                bool(self.overlay_combo)
                and set(self.overlay_combo) <= pressed
                and bool(names & set(self.overlay_combo))
                and fresh
            )
            cancel_now = (
                bool(self.cancel)
                and set(self.cancel) <= pressed
                and bool(names & set(self.cancel))
                and fresh
                and self._hold_active
            )
        if start:
            self._fire(self._on_hold_start)
        if stop:
            self._fire(self._on_hold_stop)
        if quit_now:
            self._fire(self._on_quit)
        if overlay_now:
            self._fire(self._on_overlay_toggle)
        if cancel_now:
            self._fire(self._on_cancel)

    def _on_release(self, key) -> None:
        names = _names_for(key)
        with self._lock:
            stopped = (
                self.hold_mode == "hold"
                and self._hold_active
                and bool(self.hold)
                and bool(set(self.hold) & names)
            )
            if stopped:
                self._hold_active = False
            self._pressed -= names
        if stopped:
            self._fire(self._on_hold_stop)

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()
