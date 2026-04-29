import json
import os


CONFIG_FILENAME = "config.json"

SUPPORTED_RESOLUTIONS = {
    "640x480": (640, 480),
    "800x600": (800, 600),
    "960x720": (960, 720),
    "1024x768": (1024, 768),
}

_VALID_PERSONALITIES = {
    "Normal", "The Psychologist", "The Gambler",
    "The Mirror", "The Ghost", "The Chaos Agent", "The Hustler",
}

DEFAULT_CONFIG = {
    "default_play_mode": "FairPlay",
    "default_display_mode": "Game",
    "camera_resolution": "640x480",
    "hand_orientation": "Side",
    "player_name": "",
    "clone_opponent": "",
    "shoot_window_seconds": 0.90,
    "rock_assume_seconds": 0.14,
    "beat_cooldown": 0.18,
    "handedness_threshold": 0.80,
    "ai_difficulty": "Normal",      # Easy / Normal / Hard
    "ai_personality": "Normal",     # see fair_play_ai.PERSONALITIES
    "voice_model": "US English",    # US English / Indian English
    "first_run_complete": False,    # True after first player name set
    "colourblind_mode": False,      # Replace colour-only indicators with shapes
    "analytics_consent": None,      # None=not asked, True=accepted, False=declined
    "discord_webhook_url": "",      # Override webhook URL (optional, uses hardcoded default if blank)
}


def _config_path():
    return os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)


def _normalise_config(config):
    merged = DEFAULT_CONFIG.copy()
    if isinstance(config, dict):
        merged.update(config)

    if merged["default_play_mode"] not in {"Cheat", "FairPlay", "Challenge", "Clone"}:
        merged["default_play_mode"] = DEFAULT_CONFIG["default_play_mode"]

    if merged["default_display_mode"] not in {"Game", "Diagnostic"}:
        merged["default_display_mode"] = DEFAULT_CONFIG["default_display_mode"]

    if merged["camera_resolution"] not in SUPPORTED_RESOLUTIONS:
        merged["camera_resolution"] = DEFAULT_CONFIG["camera_resolution"]

    if merged["hand_orientation"] not in {"Side", "Front"}:
        merged["hand_orientation"] = DEFAULT_CONFIG["hand_orientation"]

    if not isinstance(merged.get("player_name"), str):
        merged["player_name"] = ""
    if not isinstance(merged.get("clone_opponent"), str):
        merged["clone_opponent"] = ""
    if merged.get("ai_difficulty") not in {"Easy", "Normal", "Hard"}:
        merged["ai_difficulty"] = "Normal"
    if merged.get("ai_personality") not in _VALID_PERSONALITIES:
        merged["ai_personality"] = "Normal"
    if merged.get("voice_model") not in {"US English", "Indian English"}:
        merged["voice_model"] = "US English"
    if not isinstance(merged.get("first_run_complete"), bool):
        merged["first_run_complete"] = False
    if not isinstance(merged.get("colourblind_mode"), bool):
        merged["colourblind_mode"] = False

    merged["shoot_window_seconds"] = max(0.35, min(2.0, float(merged["shoot_window_seconds"])))
    merged["rock_assume_seconds"] = max(0.08, min(0.25, float(merged["rock_assume_seconds"])))
    merged["beat_cooldown"] = max(0.10, min(0.35, float(merged["beat_cooldown"])))
    merged["handedness_threshold"] = max(0.50, min(0.95, float(merged["handedness_threshold"])))

    return merged


def load_config():
    path = _config_path()

    if not os.path.exists(path):
        config = DEFAULT_CONFIG.copy()
        save_config(config)
        return config

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return _normalise_config(raw)
    except Exception:
        config = DEFAULT_CONFIG.copy()
        save_config(config)
        return config


def save_config(config):
    path = _config_path()
    clean = _normalise_config(config)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)

    return clean


def get_resolution_tuple(config):
    resolution_name = config.get("camera_resolution", DEFAULT_CONFIG["camera_resolution"])
    return SUPPORTED_RESOLUTIONS.get(resolution_name, SUPPORTED_RESOLUTIONS[DEFAULT_CONFIG["camera_resolution"]])
