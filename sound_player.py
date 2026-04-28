"""
Sound Effects Module

Uses macOS built-in system sounds via the `afplay` command.
All sounds play non-blocking so the camera loop never stalls.

Sound mapping:
    beat_tick    — each countdown pump (1, 2, 3)
    shoot        — SHOOT opens
    win          — player wins a round
    lose         — robot wins / game over
    draw         — draw result
    match_win    — player wins the match
    match_lose   — robot wins the match
    menu_move    — menu navigation
    menu_select  — menu item selected

Falls back silently if sounds are unavailable.
"""

import os
import subprocess
import threading

SOUNDS_DIR = "/System/Library/Sounds"

# Map event names to macOS system sound files.
SOUND_MAP = {
    "beat_tick":    "Tink.aiff",
    "shoot":        "Glass.aiff",
    "win":          "Purr.aiff",
    "lose":         "Basso.aiff",
    "draw":         "Pop.aiff",
    "match_win":    "Hero.aiff",
    "match_lose":   "Sosumi.aiff",
    "menu_move":    "Tink.aiff",
    "menu_select":  "Bottle.aiff",
}


class SoundPlayer:
    """
    Non-blocking sound player using macOS afplay.

    Usage:
        player = SoundPlayer()
        player.play("beat_tick")
        player.play("win")

    Sounds are fire-and-forget. If a sound file doesn't exist
    or afplay is unavailable, it silently does nothing.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._available = os.path.isdir(SOUNDS_DIR)

        if not self._available:
            print("[Sound] macOS system sounds not found — audio disabled.")

    def play(self, event_name):
        """
        Play a sound for the given event.
        Non-blocking: returns immediately.
        """
        if not self.enabled or not self._available:
            return

        filename = SOUND_MAP.get(event_name)
        if filename is None:
            return

        filepath = os.path.join(SOUNDS_DIR, filename)
        if not os.path.exists(filepath):
            return

        # Fire and forget in a background thread.
        thread = threading.Thread(
            target=self._play_file,
            args=(filepath,),
            daemon=True,
        )
        thread.start()

    def toggle(self):
        """Toggle sound on/off. Returns new state."""
        self.enabled = not self.enabled
        state = "ON" if self.enabled else "OFF"
        print(f"[Sound] Audio {state}")
        return self.enabled

    def is_on(self):
        """Return True if sound is currently enabled."""
        return self.enabled

    @staticmethod
    def _play_file(filepath):
        """Play a sound file using afplay (macOS)."""
        try:
            subprocess.run(
                ["afplay", filepath],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3.0,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
