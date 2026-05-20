"""
config_store.py
===============
Loads and saves the app's user settings from/to config.json.

config.json lives in the same directory as this file so the app can
always find it regardless of where it was launched from.

All values are validated on load and save -- if a setting is missing or
out of range, it gets reset to its default so the app never crashes from
a bad config file.

Typical usage:
    from config_store import load_config, save_config
    config = load_config()
    config["player_name"] = "Zac"
    save_config(config)
"""

import json
import os


# The config file lives next to this source file.
CONFIG_FILENAME = "config.json"

# Camera resolutions the app supports, mapped to (width, height) tuples.
SUPPORTED_RESOLUTIONS = {
    "640x480":  (640,  480),
    "800x600":  (800,  600),
    "960x720":  (960,  720),
    "1024x768": (1024, 768),
}

# Valid AI personality names (mirrors the constants in fair_play_ai.py).
_VALID_PERSONALITIES = {
    "Normal", "The Psychologist", "The Gambler",
    "The Mirror", "The Ghost", "The Chaos Agent", "The Hustler",
}

# Default values used when the config file is missing or a key is invalid.
DEFAULT_CONFIG = {
    "default_play_mode":    "FairPlay",  # FairPlay / Cheat / Challenge / Clone
    "default_display_mode": "Game",      # Game / Diagnostic
    "camera_resolution":    "640x480",
    "hand_orientation":     "Side",      # Side / Front
    "player_name":          "",
    "clone_opponent":       "",
    "shoot_window_seconds": 0.90,        # how long the player has to throw
    "rock_assume_seconds":  0.14,        # how long before Rock is assumed
    "beat_cooldown":        0.18,        # minimum time between beats
    "handedness_threshold": 0.80,        # confidence needed to detect handedness
    "ai_difficulty":        "Normal",    # Easy / Normal / Hard
    "ai_personality":       "Normal",    # see _VALID_PERSONALITIES above
    "voice_model":          "US English",# US English / Indian English
    "first_run_complete":   False,       # True after the player sets their name
    "colourblind_mode":     False,       # replace colour-only cues with shapes
    "analytics_consent":    None,        # None=not asked, True=yes, False=no
    "discord_webhook_url":  "",          # optional override for the webhook URL
}


def _config_path():
    """Return the absolute path to config.json (same folder as this file)."""
    return os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)


def _normalise_config(config):
    """
    Merge the given config dict with defaults and validate every value.

    Any key that is missing, the wrong type, or out of range is reset to
    its default. This means a corrupt or outdated config file never crashes
    the app -- it just gets fixed on the next save.

    Returns the fully validated config dict.
    """
    # Start from defaults so any missing keys are filled in.
    merged = DEFAULT_CONFIG.copy()
    if isinstance(config, dict):
        merged.update(config)

    # -- Enum-style string fields -- reset to default if the value isn't valid.

    if merged["default_play_mode"] not in {"Cheat", "FairPlay", "Challenge", "Clone"}:
        merged["default_play_mode"] = DEFAULT_CONFIG["default_play_mode"]

    if merged["default_display_mode"] not in {"Game", "Diagnostic"}:
        merged["default_display_mode"] = DEFAULT_CONFIG["default_display_mode"]

    if merged["camera_resolution"] not in SUPPORTED_RESOLUTIONS:
        merged["camera_resolution"] = DEFAULT_CONFIG["camera_resolution"]

    if merged["hand_orientation"] not in {"Side", "Front"}:
        merged["hand_orientation"] = DEFAULT_CONFIG["hand_orientation"]

    # -- String fields -- must be strings, not None or other types.
    if not isinstance(merged.get("player_name"), str):
        merged["player_name"] = ""
    if not isinstance(merged.get("clone_opponent"), str):
        merged["clone_opponent"] = ""

    # -- More enum-style fields.
    if merged.get("ai_difficulty") not in {"Easy", "Normal", "Hard"}:
        merged["ai_difficulty"] = "Normal"
    if merged.get("ai_personality") not in _VALID_PERSONALITIES:
        merged["ai_personality"] = "Normal"
    if merged.get("voice_model") not in {"US English", "Indian English"}:
        merged["voice_model"] = "US English"

    # -- Boolean fields -- must be actual booleans.
    if not isinstance(merged.get("first_run_complete"), bool):
        merged["first_run_complete"] = False
    if not isinstance(merged.get("colourblind_mode"), bool):
        merged["colourblind_mode"] = False

    # -- Numeric fields -- clamp to their allowed range.
    # Each tuple is (key, min_value, max_value).
    numeric_bounds = [
        ("shoot_window_seconds", 0.35, 2.0),
        ("rock_assume_seconds",  0.08, 0.25),
        ("beat_cooldown",        0.10, 0.35),
        ("handedness_threshold", 0.50, 0.95),
    ]
    for key, lo, hi in numeric_bounds:
        try:
            # Clamp the value between lo and hi.
            merged[key] = max(lo, min(hi, float(merged[key])))
        except (ValueError, TypeError, KeyError):
            # If we can't convert it to a float, fall back to the default.
            merged[key] = DEFAULT_CONFIG[key]

    return merged


def load_config():
    """
    Load config.json from disk and return a validated config dict.

    If the file doesn't exist, a fresh default config is created and saved.
    If the file is corrupt (bad JSON), it is also reset to defaults.
    """
    path = _config_path()

    # First launch: no config file exists yet -- create one with defaults.
    if not os.path.exists(path):
        config = DEFAULT_CONFIG.copy()
        save_config(config)
        return config

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return _normalise_config(raw)
    except Exception:
        # Bad JSON or unexpected structure -- start fresh so the app doesn't crash.
        config = DEFAULT_CONFIG.copy()
        save_config(config)
        return config


def save_config(config):
    """
    Validate and write the given config dict to config.json.

    Always validates before writing so the file on disk is always clean.
    Returns the validated config dict that was actually saved.
    """
    path  = _config_path()
    clean = _normalise_config(config)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)

    return clean


def get_resolution_tuple(config):
    """
    Convert the "camera_resolution" string (e.g. "640x480") to a (width, height) tuple.

    Falls back to the default resolution if the stored value is somehow invalid.
    """
    resolution_name = config.get("camera_resolution", DEFAULT_CONFIG["camera_resolution"])
    return SUPPORTED_RESOLUTIONS.get(resolution_name, SUPPORTED_RESOLUTIONS[DEFAULT_CONFIG["camera_resolution"]])
