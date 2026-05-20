"""
sound_player.py
===============
Cross-platform non-blocking sound player.

macOS:   afplay with system .aiff files (built-in, no install needed)
Windows: winsound with system .wav files (built-in, no install needed)
Linux:   aplay or paplay if available, otherwise silent

All sounds play in background daemon threads so they never block the
camera loop. Falls back silently if audio is unavailable on any platform.
"""

import os
import sys
import subprocess
import threading


# Detect the platform once at import time so every method can check it cheaply.
_PLATFORM = sys.platform  # "darwin" | "win32" | "linux"

# --- macOS: system .aiff files played via the built-in afplay command ---

# The folder where macOS stores its built-in alert sounds.
_MAC_SOUNDS_DIR = "/System/Library/Sounds"

# Map each game event to the macOS system sound file that best fits it.
_MAC_SOUND_MAP = {
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

# --- Windows: MessageBeep constants from the winsound module ---

# Each value is a Windows MessageBeep type constant.
# 0xFFFFFFFF = simple beep (no system sound file needed).
_WIN_BEEP_MAP = {
    "beat_tick":   0x00000000,   # MB_OK
    "shoot":       0x00000030,   # MB_ICONEXCLAMATION
    "win":         0x00000040,   # MB_ICONASTERISK (info sound)
    "lose":        0x00000010,   # MB_ICONHAND (error sound)
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
    silently does nothing -- it never raises an exception.
    """

    def __init__(self, enabled=True):
        # Whether the user has turned sounds on (can be toggled at runtime).
        self.enabled = enabled
        # Check once at startup whether audio is actually usable on this machine.
        self._available = self._detect_availability()

    def _detect_availability(self):
        """
        Check whether audio output is available on the current platform.

        Returns True if we found a way to play sounds, False otherwise.
        """
        if _PLATFORM == "darwin":
            # macOS: we just need the system sounds directory to exist.
            available = os.path.isdir(_MAC_SOUNDS_DIR)
            if not available:
                print("[Sound] macOS system sounds not found -- audio disabled.")
            return available

        elif _PLATFORM == "win32":
            # Windows: winsound is a built-in module but let's confirm it imports.
            try:
                import winsound  # noqa: F401
                return True
            except ImportError:
                print("[Sound] winsound not available -- audio disabled.")
                return False

        else:
            # Linux: look for any of the common command-line audio players.
            for cmd in ("aplay", "paplay", "pw-play"):
                try:
                    subprocess.run([cmd, "--version"], capture_output=True, timeout=2)
                    return True  # found one that works
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue  # try the next player
            print("[Sound] No audio player found -- audio disabled.")
            return False

    def play(self, event_name):
        """
        Play the sound for the given event name. Non-blocking.

        Spawns a daemon thread so the camera loop is never delayed.
        Does nothing if sounds are disabled or unavailable.
        """
        if not self.enabled or not self._available:
            return

        # daemon=True means this thread won't prevent the app from exiting.
        thread = threading.Thread(
            target=self._play_event,
            args=(event_name,),
            daemon=True,
        )
        thread.start()

    def _play_event(self, event_name):
        """
        Internal: called in a background thread to actually play the sound.

        Silently swallows any exception so audio errors never crash the app.
        """
        try:
            if _PLATFORM == "darwin":
                self._play_mac(event_name)
            elif _PLATFORM == "win32":
                self._play_win(event_name)
            else:
                self._play_linux(event_name)
        except Exception:
            pass  # never let a sound error crash the main thread

    def _play_mac(self, event_name):
        """Play a macOS system sound via the afplay command-line tool."""
        filename = _MAC_SOUND_MAP.get(event_name)
        if not filename:
            return  # no sound mapped for this event

        filepath = os.path.join(_MAC_SOUNDS_DIR, filename)
        if not os.path.exists(filepath):
            return  # sound file missing (shouldn't happen on a normal macOS install)

        subprocess.run(
            ["afplay", filepath],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        )

    def _play_win(self, event_name):
        """Play a Windows system sound via winsound.MessageBeep."""
        import winsound
        beep_type = _WIN_BEEP_MAP.get(event_name, 0xFFFFFFFF)  # default: simple beep
        winsound.MessageBeep(beep_type)

    def _play_linux(self, event_name):
        """Linux audio: currently silent (no universal system sound API)."""
        pass

    def toggle(self):
        """Toggle sound on/off and return the new state (True = on)."""
        self.enabled = not self.enabled
        print(f"[Sound] Audio {'ON' if self.enabled else 'OFF'}")
        return self.enabled

    def is_on(self):
        """Return True if sounds are currently enabled."""
        return self.enabled
