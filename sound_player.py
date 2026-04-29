"""
sound_player.py
===============
Cross-platform non-blocking sound player.

macOS:   afplay with system .aiff files (built-in, no install needed)
Windows: winsound with system .wav files (built-in, no install needed)
Linux:   aplay or paplay if available, otherwise silent

All sounds play in background daemon threads — never blocks the camera loop.
Falls back silently if sounds are unavailable on any platform.
"""

import os
import sys
import subprocess
import threading

# ── Platform detection ────────────────────────────────────────────────────────
_PLATFORM = sys.platform   # "darwin" | "win32" | "linux"

# ── macOS — system .aiff files via afplay ─────────────────────────────────────
_MAC_SOUNDS_DIR = "/System/Library/Sounds"
_MAC_SOUND_MAP  = {
    "beat_tick":   "Tink.aiff",
    "shoot":       "Glass.aiff",
    "win":         "Purr.aiff",
    "lose":        "Basso.aiff",
    "draw":        "Pop.aiff",
    "match_win":   "Hero.aiff",
    "match_lose":  "Sosumi.aiff",
    "menu_move":   "Tink.aiff",
    "menu_select": "Bottle.aiff",
}

# ── Windows — winsound MessageBeep constants ──────────────────────────────────
_WIN_BEEP_MAP = {
    "beat_tick":   0x00000000,   # MB_OK
    "shoot":       0x00000030,   # MB_ICONEXCLAMATION
    "win":         0x00000040,   # MB_ICONASTERISK (info)
    "lose":        0x00000010,   # MB_ICONHAND (error)
    "draw":        0x00000000,   # MB_OK
    "match_win":   0x00000040,   # MB_ICONASTERISK
    "match_lose":  0x00000010,   # MB_ICONHAND
    "menu_move":   0xFFFFFFFF,   # simple beep
    "menu_select": 0x00000000,   # MB_OK
}


class SoundPlayer:
    """
    Non-blocking cross-platform sound player.

    Usage:
        player = SoundPlayer()
        player.play("beat_tick")
        player.play("win")

    All sounds are fire-and-forget. If audio is unavailable the call
    silently does nothing — never raises an exception.
    """

    def __init__(self, enabled=True):
        self.enabled    = enabled
        self._available = self._detect_availability()

    def _detect_availability(self):
        if _PLATFORM == "darwin":
            available = os.path.isdir(_MAC_SOUNDS_DIR)
            if not available:
                print("[Sound] macOS system sounds not found — audio disabled.")
            return available

        elif _PLATFORM == "win32":
            try:
                import winsound  # noqa: F401 — built-in on Windows
                return True
            except ImportError:
                print("[Sound] winsound not available — audio disabled.")
                return False

        else:
            # Linux — check for aplay or paplay
            for cmd in ("aplay", "paplay", "pw-play"):
                try:
                    subprocess.run([cmd, "--version"],
                                   capture_output=True, timeout=2)
                    return True
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            print("[Sound] No audio player found — audio disabled.")
            return False

    def play(self, event_name):
        """Play a sound for the given event. Non-blocking."""
        if not self.enabled or not self._available:
            return
        thread = threading.Thread(
            target=self._play_event,
            args=(event_name,),
            daemon=True,
        )
        thread.start()

    def _play_event(self, event_name):
        try:
            if _PLATFORM == "darwin":
                self._play_mac(event_name)
            elif _PLATFORM == "win32":
                self._play_win(event_name)
            else:
                self._play_linux(event_name)
        except Exception:
            pass

    def _play_mac(self, event_name):
        filename = _MAC_SOUND_MAP.get(event_name)
        if not filename:
            return
        filepath = os.path.join(_MAC_SOUNDS_DIR, filename)
        if not os.path.exists(filepath):
            return
        subprocess.run(
            ["afplay", filepath],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        )

    def _play_win(self, event_name):
        import winsound
        beep_type = _WIN_BEEP_MAP.get(event_name, 0xFFFFFFFF)
        winsound.MessageBeep(beep_type)

    def _play_linux(self, event_name):
        # Linux — silent fallback (no universal system sound)
        pass

    def toggle(self):
        """Toggle sound on/off. Returns new state."""
        self.enabled = not self.enabled
        print(f"[Sound] Audio {'ON' if self.enabled else 'OFF'}")
        return self.enabled

    def is_on(self):
        return self.enabled
