import time
import os
import subprocess
import threading
import queue as _queue
import cv2
import auto_updater
from feedback_store import save_feedback
from privacy_notice import (has_consent, has_declined, needs_consent_prompt,
                             set_consent, get_webhook_url, consent_summary)
from discord_reporter import send_crash_report, send_feedback as discord_send_feedback

from gesture_state import GestureStateTracker
from rps_game_state import RPSGameController
from fair_play_state import FairPlayController
from challenge_mode_state import ChallengeController
from robot_output import RobotOutputBuffer
from challenge_stats_logger import ChallengeStatsLogger
from player_profile_store import PlayerProfileStore
from player_clone_ai import PlayerCloneAI

from hand_landmarks import create_hands_detector, create_nav_detector, process_hand_frame, process_two_hands_frame, create_kalman_wrist_state
from landmark_collector import LandmarkCollector
from emotion_tracker import EmotionTracker
from ui_renderer import (
    draw_top_bar,
    draw_info_panel,
    draw_diagnostic_game_panel,
    draw_game_mode_view,
    draw_menu_screen,
    draw_settings_screen,
    draw_features_screen,
    draw_clone_setup_screen,
    draw_player_stats_screen,
    draw_tutorial_screen,
    draw_emotion_debug,
    draw_gesture_nav_overlay,
    draw_result_flash,
    draw_quality_warnings,
    draw_help_overlay,
    draw_simulation_screen,
    draw_session_summary,
    draw_two_player_view,
    draw_pvpvai_view,
    draw_two_player_diagnostic,
    draw_personality_settings,
    draw_reflex_solo_view,
    draw_reflex_two_player_view,
    draw_bluff_mode_view,
    draw_simon_says_solo_view,
    draw_simon_says_two_player_view,
    draw_squid_game_view,
    draw_squid_game_2p_view,
    draw_prediction_race_view,
    draw_gesture_rehab_view,
    draw_arcade_snake_view,
    draw_rpsls_view,
    draw_game_category_screen,
    draw_simulations_hub_screen,
    draw_rpsls_tutorial_screen,
    draw_rpsls_side_notice,
    draw_login_screen,
    draw_hand_enroll_view,
    draw_hand_login_view,
    draw_hardware_test_view,
    draw_notes_screen,
    draw_consent_screen,
)

from config_store import (
    load_config,
    save_config,
    get_resolution_tuple,
    SUPPORTED_RESOLUTIONS,
)
from sound_player import SoundPlayer
from voice_control import VoiceController, VOSK_AVAILABLE
from gesture_nav import GestureNavController
from two_player_state import TwoPlayerPvPController, PvPvAIController
from reflex_state import ReflexSoloController, ReflexTwoPlayerController
from bluff_mode_state import BluffModeController
from simon_says_state import SimonSaysSoloController, SimonSaysTwoPlayerController
from squid_game_state import SquidGameController, SquidGame2PController
from gesture_fingerprint import FingerprintStore, FingerprintClassifier
from hand_enroll_state import HandEnrollController, HandLoginController, HandDiagController
from rpsls_state import RPSLSController
from fair_play_ai import FairPlayAI, PERSONALITIES, PERSONALITY_NAMES
from prediction_race_state import PredictionRaceController
from gesture_rehab_state import GestureRehabController
from commentary_engine import CommentaryEngine
from arcade_snake_state import ArcadeSnakeController

def _run_report_updater_bg():
    """Background-dispatch the report auto-updater."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from report_updater import update_report
        result = update_report(verbose=False)
        if result:
            print(f"[Report] Updated → {result}")
    except Exception as exc:
        print(f"[Report] Updater error: {exc}")

WINDOW_NAME = "RPS Gesture Recogniser"

KEY_ENTER = {10, 13}
KEY_ESC = 27
KEY_UP = {82, ord("w"), ord("W")}
KEY_DOWN = {84, ord("s"), ord("S")}
KEY_LEFT = {81, ord("a"), ord("A")}
KEY_RIGHT = {83, ord("d"), ord("D")}

SETTINGS_SCHEMA = [
    {
        "key": "player_name",
        "label": "Player Name",
        "type": "text",
        "desc": "Your name for stat tracking and clone mode. Set once - rounds are automatically recorded to your profile.",
    },
    {
        "key": "default_play_mode",
        "label": "Default Mode",
        "type": "choice",
        "options": ["Cheat", "FairPlay", "Challenge"],
        "desc": "Which game mode starts by default. Cheat = robot always wins, Fair Play = best of 3, Challenge = endless streak.",
    },
    {
        "key": "default_display_mode",
        "label": "Default Display",
        "type": "choice",
        "options": ["Game", "Diagnostic"],
        "desc": "Game = arcade-style view. Diagnostic = shows raw gesture data, finger states, and AI internals.",
    },
    {
        "key": "camera_resolution",
        "label": "Resolution",
        "type": "choice",
        "options": list(SUPPORTED_RESOLUTIONS.keys()),
        "desc": "Camera capture resolution. Higher = better detection but slower. 640x480 recommended.",
    },
    {
        "key": "hand_orientation",
        "label": "Hand Orientation",
        "type": "choice",
        "options": ["Side", "Front"],
        "desc": "Side = hand viewed from the side (default). Front = palm facing camera (needs trained model).",
    },
    {
        "key": "shoot_window_seconds",
        "label": "Shoot Window",
        "type": "float",
        "min": 0.35,
        "max": 0.90,
        "step": 0.05,
        "desc": "How long SHOOT stays open for you to throw. Longer = more time to change gesture. Default 0.55s.",
    },
    {
        "key": "rock_assume_seconds",
        "label": "Rock Assume",
        "type": "float",
        "min": 0.08,
        "max": 0.25,
        "step": 0.01,
        "desc": "After SHOOT, if no change is detected in this time, Rock is assumed. Lower = faster but less fair. Default 0.14s.",
    },
    {
        "key": "beat_cooldown",
        "label": "Beat Cooldown",
        "type": "float",
        "min": 0.10,
        "max": 0.35,
        "step": 0.01,
        "desc": "Minimum time between counting pump beats. Prevents double-counting fast pumps. Default 0.18s.",
    },
    {
        "key": "handedness_threshold",
        "label": "Handedness Threshold",
        "type": "float",
        "min": 0.50,
        "max": 0.95,
        "step": 0.05,
        "desc": "How confident MediaPipe must be about left/right hand. Lower = accepts uncertain detections. Default 0.80.",
    },
    {
        "key": "ai_difficulty",
        "label": "AI Difficulty",
        "type": "choice",
        "options": ["Easy", "Normal", "Hard"],
        "desc": "Easy = AI plays near-random for first 20 rounds, limited max skill. Normal = balanced. Hard = adapts fast, higher accuracy.",
    },
    {
        "key": "voice_model",
        "label": "Voice Model",
        "type": "choice",
        "options": ["US English", "Indian English"],
        "desc": "US English = vosk-model-small-en-us-0.15 (default). Indian English = vosk-model-small-en-in-0.4, better for Australian and non-US accents. Download the model from alphacephei.com/vosk/models.",
    },
    {
        "key": "__enroll_fingerprint__",
        "label": "Enroll Fingerprint",
        "type": "action",
        "desc": "Train the system to recognise you by how you move your finger. Launches a Squid Game session for silent biometric data collection, then verifies accuracy before saving.",
    },
    {
        "key": "__switch_player__",
        "label": "Switch Player",
        "type": "action",
        "desc": "Return to the login screen to change the active player.",
    },
    {
        "key": "__privacy__",
        "label": "Privacy Settings",
        "type": "action",
        "desc": "Review or change your consent for sending crash reports and feedback to the developer.",
    },
    {
        "key": "__back__",
        "label": "Back",
        "type": "action",
        "desc": "",
    },
]

# ── Feature toggles (on/off only, separate from program settings) ──────── #
FEATURES_SCHEMA = [
    {
        "key": "input_mode",
        "label": "Input Mode",
        "type": "choice",
        "options": ["Pump", "Voice"],
        "desc": (
            "Pump = physical fist-pump countdown. "
            "Voice = say READY  ONE  TWO  THREE then your throw - for accessibility."
        ),
    },
    {
        "key": "emotion_enabled",
        "label": "Emotion Tracking",
        "desc": (
            "Detect Happy / Surprised / Frustrated from face via FaceMesh. "
            "Runs every 3rd frame. Turn off for best performance."
        ),
    },
    {
        "key": "gesture_nav_enabled",
        "label": "Gesture Navigation",
        "desc": (
            "Navigate menus with your index finger. "
            "Adds hand tracking to menu screens. Hover 2s to select."
        ),
    },
    {
        "key": "face_debug_enabled",
        "label": "Face Debug Overlay",
        "desc": (
            "Show emotion landmark dots and score bars on the camera feed. "
            "Also toggled in-game with the E key."
        ),
    },
    {
        "key": "__personalities__",
        "label": "AI Personalities",
        "type": "action",
        "desc": (
            "Choose how the AI behaves: Normal, The Psychologist, The Gambler, "
            "The Ghost, The Mirror, The Hustler, or The Chaos Agent. "
            "Resets to Normal on restart."
        ),
    },
    {
        "key": "colourblind_mode",
        "label": "Colourblind Mode",
        "desc": (
            "Replace red/green win/lose colours with blue/orange alternatives. "
            "Also adds shape indicators to result screens."
        ),
    },
    {
        "key": "__back__",
        "label": "Back",
        "type": "action",
        "desc": "",
    },
]

# ── Game mode category structure for the 3-level menu ─────────────────────── #
# Each category has a label, icon char, description, and list of (label, action)
# entries shown when the category is opened.
GAME_CATEGORIES = [
    {
        "key":   "rps",
        "label": "Rock Paper Scissors",
        "icon":  "RPS",
        "desc":  "Classic RPS.\nSolo vs AI or two players head-to-head.",
        "modes": [
            ("Fair Play  (vs AI)",   "FairPlay"),
            ("Challenge  (vs AI)",   "Challenge"),
            ("Clone Mode  (vs AI)",  "Clone"),
            ("2 Player  PvP",        "TwoPlayerPvP"),
            ("PvPvAI  (1v1v1)",      "PvPvAI"),
            ("Cheat Mode",           "Cheat"),
        ],
    },
    {
        "key":   "rpsls",
        "label": "RPSLS  (5 Gestures)",
        "icon":  "5G",
        "desc":  "Rock Paper Scissors Lizard Spock.\n5-gesture variant vs AI.",
        "modes": [
            ("RPSLS vs AI",          "RPSLS"),
            ("How to Play RPSLS",    "RPSLSTutorial"),
            ("Gesture Diagnostic",   "RPSLSDiagnostic"),
        ],
    },
    {
        "key":   "reflex",
        "label": "Speed Reflex",
        "icon":  "SPD",
        "desc":  "Match gestures as fast as possible.\n30-second solo sprint or 2-player race.",
        "modes": [
            ("Solo  (30s sprint)",   "ReflexSolo"),
            ("2 Player  Race",       "ReflexTwoPlayer"),
        ],
    },
    {
        "key":   "bluff",
        "label": "Bluff Mode",
        "icon":  "BLF",
        "desc":  "AI announces its move before every round.\nIs it telling the truth?",
        "modes": [
            ("Start Bluff Mode",     "BluffMode"),
        ],
    },
    {
        "key":   "simon",
        "label": "Simon Says",
        "icon":  "SIM",
        "desc":  "Gesture memory game.\nRepeat the growing sequence or lose.",
        "modes": [
            ("Solo",                 "SimonSaysSolo"),
            ("2 Player  Chain",      "SimonSays2P"),
        ],
    },
    {
        "key":   "squid",
        "label": "Red Light Green Light",
        "icon":  "RGL",
        "desc":  "Guide your finger to dots.\nFreeze when the light turns red.",
        "modes": [
            ("Solo  (survival)",     "SquidGame"),
            ("2 Player  (race)",     "SquidGame2P"),
        ],
    },
    {
        "key":   "prediction",
        "label": "Prediction Race",
        "icon":  "PRD",
        "desc":  "The AI shows its prediction before every round.\nYour goal: throw something it DIDN'T predict.",
        "modes": [
            ("Start Prediction Race", "PredictionRace"),
        ],
    },
    {
        "key":   "rehab",
        "label": "Gesture Trainer",
        "icon":  "GYM",
        "desc":  "Guided gesture exercise sessions.\nHold each gesture to build dexterity and accuracy.",
        "modes": [
            ("Start Training",        "GestureRehab"),
        ],
    },
    {
        "key":   "arcade",
        "label": "Gesture Arcade",
        "icon":  "ARC",
        "desc":  "Gesture-controlled Snake.\nRock=straight  Scissors=turn left  Paper=turn right.",
        "modes": [
            ("Gesture Snake",         "ArcadeSnake"),
        ],
    },
]


class _IOWorker:
    """
    Background thread for all disk-bound work (JSON save, Excel write).
    Submitting a callable returns immediately; work runs serially off the
    main thread so round-result I/O never stalls the camera loop.
    """
    def __init__(self):
        self._q      = _queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, fn, *args, **kwargs):
        self._q.put((fn, args, kwargs))

    def flush(self, timeout=3.0):
        """Wait until queue is drained (call on shutdown)."""
        done = threading.Event()
        self._q.put((lambda: done.set(), (), {}))
        done.wait(timeout=timeout)

    def _run(self):
        while True:
            fn, args, kwargs = self._q.get()
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                print(f"[IOWorker] {exc}")


_io_worker = _IOWorker()


class _AsyncChallengeStatsLogger:
    """
    Non-blocking wrapper around ChallengeStatsLogger.
    All methods that touch disk are dispatched to the background IO worker.
    Methods that only touch in-memory state (get_high_score, active_run)
    are called synchronously so the game can read them immediately.
    """
    def __init__(self, inner):
        self._inner = inner

    # --- Passthrough synchronous reads ---
    def get_high_score(self):
        return self._inner.get_high_score()

    def update_context(self, **kwargs):
        self._inner.update_context(**kwargs)

    # --- Async disk writes ---
    def start_run(self):
        # Call synchronously so active_run is set before log_round references it
        return self._inner.start_run()

    def log_round(self, *args, **kwargs):
        _io_worker.submit(self._inner.log_round, *args, **kwargs)

    def finalize_run(self, *args, **kwargs):
        _io_worker.submit(self._inner.finalize_run, *args, **kwargs)

    # Surface active_run for anything that reads it
    @property
    def active_run(self):
        return self._inner.active_run


def build_controllers(robot_output, config, challenge_logger):
    difficulty = config.get("ai_difficulty", "Normal")
    return {
        "cheat_controller": RPSGameController(
            robot_output=robot_output,
            beat_cooldown=config["beat_cooldown"],
            shoot_window_seconds=config["shoot_window_seconds"],
            rock_assume_seconds=config["rock_assume_seconds"],
        ),
        "fair_controller": FairPlayController(
            robot_output=robot_output,
            ai=FairPlayAI(difficulty=difficulty),
            beat_cooldown=config["beat_cooldown"],
            shoot_window_seconds=config["shoot_window_seconds"],
            rock_assume_seconds=config["rock_assume_seconds"],
        ),
        "challenge_controller": ChallengeController(
            robot_output=robot_output,
            stats_logger=challenge_logger,
            beat_cooldown=config["beat_cooldown"],
            shoot_window_seconds=config["shoot_window_seconds"],
            rock_assume_seconds=config["rock_assume_seconds"],
        ),
        "clone_controller": FairPlayController(
            robot_output=robot_output,
            ai=FairPlayAI(difficulty=difficulty),
            win_target=3,
            play_mode_label="Clone",
            beat_cooldown=config["beat_cooldown"],
            shoot_window_seconds=config["shoot_window_seconds"],
            rock_assume_seconds=config["rock_assume_seconds"],
        ),
        "pvp_controller": TwoPlayerPvPController(
            win_target=3,
            beat_cooldown=config["beat_cooldown"],
            shoot_window_seconds=config["shoot_window_seconds"],
            rock_assume_seconds=config["rock_assume_seconds"],
        ),
        "pvpvai_controller": PvPvAIController(
            ai=FairPlayAI(difficulty=difficulty),
            win_target=3,
            beat_cooldown=config["beat_cooldown"],
            shoot_window_seconds=config["shoot_window_seconds"],
            rock_assume_seconds=config["rock_assume_seconds"],
        ),
        "reflex_solo_controller":  ReflexSoloController(),
        "reflex_2p_controller":    ReflexTwoPlayerController(),
        "bluff_controller":        BluffModeController(
            ai=FairPlayAI(difficulty=difficulty,
                          personality=config.get("ai_personality", "Normal")),
        ),
        "simon_solo_controller":   SimonSaysSoloController(),
        "simon_2p_controller":     SimonSaysTwoPlayerController(),
        "squid_controller":        SquidGameController(),
        "squid_2p_controller":     SquidGame2PController(),
        "prediction_race_controller": PredictionRaceController(
            ai=FairPlayAI(difficulty=difficulty,
                          personality=config.get("ai_personality", "Normal")),
            beat_cooldown=config["beat_cooldown"],
            # Prediction Race needs a longer shoot window and guard than FairPlay.
            # After the tracker reset at SHOOT open, the tracker needs ~6 frames
            # (~200ms at 30fps) to re-confirm Scissors from scratch.
            # shoot_change_guard_seconds must be >= tracker confirmation time.
            shoot_window_seconds=max(config["shoot_window_seconds"], 0.90),
            rock_assume_seconds=max(config["rock_assume_seconds"], 0.45),
            shoot_change_guard_seconds=0.20,
        ),
        "gesture_rehab_controller":  GestureRehabController(),
        "arcade_snake_controller":   ArcadeSnakeController(),
        "rpsls_controller":        RPSLSController(
            ai=FairPlayAI(difficulty=difficulty,
                          personality=config.get("ai_personality", "Normal")),
        ),
    }


def update_challenge_logger_context(app_state):
    logger = app_state.get("challenge_logger")
    if logger is None:
        return

    logger.update_context(
        display_mode=app_state["display_mode"],
        camera_resolution=app_state["config"]["camera_resolution"],
    )


def build_app_state():
    config = load_config()
    robot_output = RobotOutputBuffer()
    challenge_logger = _AsyncChallengeStatsLogger(ChallengeStatsLogger())

    app_state = {
        "app_screen": (
            "CONSENT" if needs_consent_prompt(config)
            else "LOGIN" if not config.get("player_name", "").strip()
            else "MENU"
        ),
        "_consent_selected": 0,   # 0=Accept, 1=Decline
        "menu_index": 0,
        "settings_index": 0,
        "features_index": 0,
        "menu_items": [
            ("Game Modes",          "GameModes"),
            ("Player Stats",        "Stats"),
            ("How to Play",         "Tutorial"),
            ("Settings",            "Settings"),
            ("Features",            "Features"),
            ("Simulations",         "Simulations"),
            ("Quit",                "Quit"),
        ],
        # Sub-menu kept for legacy voice-nav compatibility (not shown in new UI)
        "submenu_items": [
            ("< Back",              "BackToMenu"),
        ],
        "in_submenu": False,
        "submenu_index": 0,
        # Simulation state
        "sim_state": {"status": "idle", "progress": 0.0, "progress_text": "",
                      "results": None, "error": None},
        "config": config,
        "target_hand": "Auto",
        "display_mode": config["default_display_mode"],
        "play_mode": config["default_play_mode"],
        "cap": None,
        "tracker": GestureStateTracker(
            history_size=7,
            confirm_frames=3,
            invalid_reset_frames=6
        ),
        "robot_output": robot_output,
        "challenge_logger": challenge_logger,
        "profile_store": PlayerProfileStore(),
        "landmark_collector": LandmarkCollector(),
        "collector_message": "",
        "_last_recorded_round": None,
        "_pending_round_log": None,
        "emotion_tracker": EmotionTracker(),
        "emotion_state": None,
        "voice_controller": VoiceController(),
        "voice_mode_active": False,
        "emotion_debug": config.get("face_debug_enabled", False),
        "gesture_nav": GestureNavController(),
        "_emotion_frame_skip": 0,
        "sound_player":      SoundPlayer(),
        "commentary_engine": CommentaryEngine(enabled=False),
        "fingerprint_store": FingerprintStore(),
        "_fp_enroll_controller":  None,   # built on demand when enrolling
        "_fp_login_controller":   None,   # built on demand for login
        "_login_text":            config.get("player_name", ""),
        "_notes_text":            "",
        "_notes_submitted":       False,
        "_notes_saved_path":      "",
        "_login_mode":            "type",  # "type" | "fingerprint"
        "_snd_last_state":      "",
        "_snd_last_beat_count": 0,
        "_fps_last_t":  time.monotonic(),
        "_fps_val":     0.0,
        "_settings_text_edit": False,
        # Flash effect tracking
        "_flash_info":   {"active": False, "result": "", "frame_idx": 0, "mic_level": 0.0},
        # Win/loss streak tracking for HUD label
        "_streak_count": 0,
        "_streak_type":  "",   # "win" or "lose"
        # Help overlay toggle
        "show_help":     False,
        # EMA wrist smoothing state (shared across frames)
        "_ema_state":    create_kalman_wrist_state(),
        # Two-player: separate Kalman filters for each hand
        "_tp_tracker_p1": GestureStateTracker(),
        "_tp_tracker_p2": GestureStateTracker(),
        "_tp_ema_states": [create_kalman_wrist_state(), create_kalman_wrist_state()],
        # Personality screen state
        "app_screen_personality": False,
        "personality_index":      0,
        # Game category / mode selection screen state
        "game_category_index":    0,
        "game_mode_index":        0,
        "in_game_category":       False,   # True when inside a category's mode list
        # Simulations hub screen state
        "sim_tab_index":          0,       # which simulation is highlighted
        # RPSLS tutorial state
        "rpsls_tutorial_step":    0,
        "_came_from_category":    False,
        # Rounds list for stats history dots
        "stats_rounds":   [],
        "stats_sessions": [],          # match history for History tab
        "stats_filter":   "All",       # mode filter: All / FairPlay / Challenge / Cheat / Clone
        "stats_tab":      "overview",  # overview / history
        # Match summary
        "_match_summary": None,
        # Practice mode flag
        "_practice_mode": False,
    }

    update_challenge_logger_context(app_state)
    app_state.update(build_controllers(robot_output, config, challenge_logger))
    return app_state


def rebuild_controllers(app_state):
    controllers = build_controllers(
        app_state["robot_output"],
        app_state["config"],
        app_state["challenge_logger"],
    )
    app_state.update(controllers)


def _apply_voice_mode(app_state):
    """Start/stop VoiceController and propagate voice_mode flag to all controllers."""
    enabled = app_state["config"].get("input_mode") == "Voice"
    prefer_indian = app_state["config"].get("voice_model") == "Indian English"
    vc = app_state["voice_controller"]

    # Rebuild controller if model preference changed
    if hasattr(vc, "_prefer_indian") and vc._prefer_indian != prefer_indian:
        if vc.is_running():
            vc.stop()
        app_state["voice_controller"] = VoiceController(prefer_indian=prefer_indian)
        vc = app_state["voice_controller"]

    if enabled and not vc.is_running():
        ok = vc.start()
        if not ok:
            print(f"[Voice] Could not start: {vc.get_error()}")
            app_state["config"]["input_mode"] = "Pump"
            save_config(app_state["config"])
            enabled = False

    elif not enabled and vc.is_running():
        vc.stop()

    app_state["voice_mode_active"] = enabled
    for key in ("cheat_controller", "fair_controller", "challenge_controller", "clone_controller"):
        ctrl = app_state.get(key)
        if ctrl and hasattr(ctrl, "set_voice_mode"):
            ctrl.set_voice_mode(enabled)


def apply_camera_resolution(cap, config):
    width, height = get_resolution_tuple(config)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"Requested camera resolution: {width}x{height}")


def finalize_active_challenge_run(app_state, status="abandoned"):
    logger = app_state.get("challenge_logger")
    controller = app_state.get("challenge_controller")

    if logger is None or controller is None:
        return

    logger.finalize_run(
        final_streak=controller.streak,
        status=status
    )


def reset_all_modes(app_state):
    app_state["tracker"].reset()
    app_state["robot_output"].clear_pending_locked()
    app_state["cheat_controller"].reset()
    app_state["fair_controller"].reset()
    app_state["challenge_controller"].reset()
    app_state["clone_controller"].reset()
    app_state["pvp_controller"].reset()
    app_state["pvpvai_controller"].reset()
    app_state["reflex_solo_controller"].reset()
    app_state["reflex_2p_controller"].reset()
    app_state["bluff_controller"].reset()
    app_state["simon_solo_controller"].reset()
    app_state["simon_2p_controller"].reset()
    app_state["squid_controller"].reset()
    app_state["squid_2p_controller"].reset()
    app_state["prediction_race_controller"].reset()
    app_state["gesture_rehab_controller"].reset()
    app_state["arcade_snake_controller"].reset()
    app_state["rpsls_controller"].reset()
    app_state["_tp_tracker_p1"].reset()
    app_state["_tp_tracker_p2"].reset()
    app_state["_tp_ema_states"] = [create_kalman_wrist_state(), create_kalman_wrist_state()]
    # Reset main wrist Kalman
    kf = app_state["_ema_state"].get("kalman")
    if kf:
        kf.reset()
    app_state["_pending_round_log"] = None
    app_state["gesture_nav"].reset()
    app_state["_snd_last_state"]      = ""
    app_state["_snd_last_beat_count"] = 0


def _dispatch_sounds(app_state, game_state):
    """
    Fire sound effects by detecting game-state transitions.
    Called once per frame during GAME mode.
    """
    sp         = app_state["sound_player"]
    cur_state  = game_state.get("state", "")
    beat_count = game_state.get("beat_count", 0)
    prev_state = app_state["_snd_last_state"]
    prev_beats = app_state["_snd_last_beat_count"]
    banner     = game_state.get("result_banner", "").upper()

    # Each pump beat
    if beat_count > prev_beats and cur_state == "COUNTDOWN":
        sp.play("beat_tick")

    # SHOOT window opens
    if cur_state == "SHOOT_WINDOW" and prev_state != "SHOOT_WINDOW":
        sp.play("shoot")

    # Round result (fires exactly once per result)
    if cur_state == "ROUND_RESULT" and prev_state != "ROUND_RESULT":
        if "YOU WIN" in banner or "SURVIVE" in banner:
            sp.play("win")
        elif "DRAW" in banner:
            sp.play("draw")
        else:
            sp.play("lose")

    # Match result
    if cur_state == "MATCH_RESULT" and prev_state != "MATCH_RESULT":
        if "YOU WIN" in banner:
            sp.play("match_win")
        else:
            sp.play("match_lose")

    app_state["_snd_last_state"]      = cur_state
    app_state["_snd_last_beat_count"] = beat_count


def start_game(app_state, mode=None, from_category=False):
    if mode is None:
        mode = app_state["config"]["default_play_mode"]

    if (
        app_state["app_screen"] == "GAME"
        and app_state["play_mode"] == "Challenge"
        and mode != "Challenge"
    ):
        finalize_active_challenge_run(app_state, status="abandoned")

    app_state["play_mode"]          = mode
    app_state["display_mode"]       = app_state["config"]["default_display_mode"]
    app_state["_came_from_category"] = from_category
    update_challenge_logger_context(app_state)
    reset_all_modes(app_state)
    app_state["app_screen"] = "GAME"

    # ── Cross-session AI learning: restore bandit weights for this player ──
    player_name = app_state["config"].get("player_name", "").strip()
    if player_name:
        store = app_state["profile_store"]
        for ctrl_key in ("fair_controller", "challenge_controller",
                         "clone_controller", "bluff_controller"):
            ctrl = app_state.get(ctrl_key)
            if ctrl and hasattr(ctrl, "ai") and hasattr(ctrl.ai, "_bandit"):
                loaded = store.load_ai_state(player_name, ctrl.ai)
                if loaded:
                    print(f"[AI] Restored learned model for {player_name} in {ctrl_key}")
                    break  # only need to load once — they share the same pattern

    print(f"Play mode: {app_state['play_mode']}")
    print(f"Display mode: {app_state['display_mode']}")


def open_menu(app_state):
    if app_state["app_screen"] == "GAME" and app_state["play_mode"] == "Challenge":
        finalize_active_challenge_run(app_state, status="abandoned")

    # ── Cross-session AI learning: persist bandit weights for this player ──
    player_name = app_state["config"].get("player_name", "").strip()
    if player_name and app_state.get("app_screen") == "GAME":
        store = app_state.get("profile_store")
        if store:
            for ctrl_key in ("fair_controller", "challenge_controller"):
                ctrl = app_state.get(ctrl_key)
                if ctrl and hasattr(ctrl, "ai") and hasattr(ctrl.ai, "_bandit"):
                    store.save_ai_state(player_name, ctrl.ai)
                    break

    app_state["app_screen"]          = "MENU"
    app_state["menu_index"]          = 0
    app_state["in_submenu"]          = False
    app_state["submenu_index"]       = 0
    app_state["in_game_category"]    = False
    app_state["game_category_index"] = 0
    app_state["game_mode_index"]     = 0
    reset_all_modes(app_state)

    # First-run onboarding: no player name set → go to Settings with name field focused
    if not app_state["config"].get("first_run_complete") and \
       not app_state["config"].get("player_name", "").strip():
        app_state["app_screen"] = "SETTINGS"
        app_state["settings_index"] = 0  # Player Name is first item
        app_state["_settings_text_edit"] = True


def open_settings(app_state):
    app_state["app_screen"] = "SETTINGS"
    app_state["settings_index"] = 0
    app_state["_settings_text_edit"] = False


def open_features(app_state):
    app_state["app_screen"] = "FEATURES"
    app_state["features_index"] = 0


def apply_feature_toggle(app_state, key, direction=0):
    """
    Toggle a boolean feature or cycle a choice feature, then apply side effects.

    direction: -1 = previous option, +1 = next option, 0 = toggle bool / advance choice
    """
    config = app_state["config"]
    item   = next((s for s in FEATURES_SCHEMA if s.get("key") == key), None)

    if item is None:
        return

    if item.get("type") == "choice":
        options = item["options"]
        current = config.get(key, options[0])
        idx     = options.index(current) if current in options else 0
        if direction == 0:
            direction = 1   # Enter advances forward
        idx = (idx + direction) % len(options)
        config[key] = options[idx]
    else:
        config[key] = not config.get(key, False)

    save_config(config)

    # Side effects
    if key == "face_debug_enabled":
        app_state["emotion_debug"] = config[key]

    elif key == "gesture_nav_enabled":
        if not config[key]:
            app_state["gesture_nav"].reset()

    elif key == "emotion_enabled":
        if not config[key]:
            app_state["emotion_tracker"].reset()
            app_state["emotion_state"] = None

    elif key == "input_mode":
        _apply_voice_mode(app_state)

    elif key == "colourblind_mode":
        print(f"[Features] Colourblind mode {'ON' if config[key] else 'OFF'}")

    print(f"[Features] {key} = {config[key]}")


def handle_features_key(app_state, key):
    """Handle keys on the Features screen."""
    schema = FEATURES_SCHEMA
    if key in KEY_UP:
        app_state["features_index"] = (app_state["features_index"] - 1) % len(schema)
    elif key in KEY_DOWN:
        app_state["features_index"] = (app_state["features_index"] + 1) % len(schema)
    elif key in KEY_LEFT:
        item = schema[app_state["features_index"]]
        if item.get("key") not in ("__back__", "__personalities__"):
            apply_feature_toggle(app_state, item["key"], direction=-1)
    elif key in KEY_RIGHT or key in KEY_ENTER:
        item = schema[app_state["features_index"]]
        if item.get("key") == "__back__":
            open_menu(app_state)
        elif item.get("key") == "__personalities__":
            app_state["app_screen"] = "PERSONALITY_SELECT"
            cur = app_state["config"].get("ai_personality", "Normal")
            app_state["personality_index"] = PERSONALITY_NAMES.index(cur) \
                if cur in PERSONALITY_NAMES else 0
        else:
            apply_feature_toggle(app_state, item["key"], direction=1)
    elif key == KEY_ESC:
        open_menu(app_state)


def open_clone_setup(app_state):
    """Open the Clone Mode setup screen."""
    app_state["app_screen"] = "CLONE_SETUP"
    app_state["clone_step"] = "enter_name"
    app_state["clone_text_buffer"] = app_state["config"].get("player_name", "")
    app_state["clone_opponent_index"] = 0
    app_state["clone_available"] = []
    app_state["clone_message"] = ""


def _start_clone_game(app_state, opponent_name):
    """Load a clone and start the game."""
    store = app_state["profile_store"]
    tables = store.build_pattern_tables(opponent_name)

    if tables is None or tables["round_count"] < 30:
        count = tables["round_count"] if tables else 0
        app_state["clone_message"] = f"'{opponent_name}' has {count} rounds. Need 30+."
        return

    clone_ai = PlayerCloneAI(tables)
    app_state["clone_controller"].ai = clone_ai
    app_state["clone_controller"].play_mode_label = f"vs {opponent_name}"
    app_state["clone_controller"].opponent_label = opponent_name.upper()
    app_state["clone_controller"].win_target = 3

    app_state["config"]["clone_opponent"] = opponent_name
    save_config(app_state["config"])

    print(f"[Clone] Playing vs '{opponent_name}' ({tables['round_count']} rounds)")
    start_game(app_state, "Clone")


def handle_clone_setup_key(app_state, key):
    """Handle keys on the Clone Setup screen."""
    step = app_state.get("clone_step", "enter_name")

    if step == "enter_name":
        buf = app_state.get("clone_text_buffer", "")

        if key in KEY_ENTER and buf.strip():
            # Save player name and move to opponent selection.
            app_state["config"]["player_name"] = buf.strip()
            save_config(app_state["config"])
            print(f"[Clone] Player name: '{buf.strip()}'")

            # Load available opponents.
            store = app_state["profile_store"]
            all_players = store.list_players()

            # Update Excel research sheets.
            # Background-dispatch the Excel report generation - can take 1-3s
            app_state["clone_profiles_updating"] = True
            def _profiles_done():
                store.generate_all_player_reports()
                app_state["clone_profiles_updating"] = False
            _io_worker.submit(_profiles_done)
            playable = [
                (name, count) for name, count in all_players
                if count >= 30
            ]

            if playable:
                app_state["clone_available"] = playable
                app_state["clone_opponent_index"] = 0
                app_state["clone_step"] = "select_opponent"
                app_state["clone_message"] = ""
            else:
                # Show all profiles with their counts.
                app_state["clone_step"] = "no_profiles"
                app_state["clone_all_players"] = all_players
                app_state["clone_message"] = ""

        elif key == KEY_ESC:
            open_menu(app_state)

        elif key == 8 or key == 127:
            app_state["clone_text_buffer"] = buf[:-1]

        elif 32 <= key <= 126:
            app_state["clone_text_buffer"] = buf + chr(key)

    elif step == "select_opponent":
        available = app_state.get("clone_available", [])

        if key in KEY_UP and available:
            app_state["clone_opponent_index"] = (
                (app_state["clone_opponent_index"] - 1) % len(available)
            )
        elif key in KEY_DOWN and available:
            app_state["clone_opponent_index"] = (
                (app_state["clone_opponent_index"] + 1) % len(available)
            )
        elif key in KEY_ENTER and available:
            name, count = available[app_state["clone_opponent_index"]]
            _start_clone_game(app_state, name)
        elif key == KEY_ESC:
            app_state["clone_step"] = "enter_name"

    elif step == "no_profiles":
        if key == KEY_ESC:
            open_menu(app_state)
        elif key in KEY_ENTER:
            # Go back to name entry so they can start playing to build data.
            open_menu(app_state)


def open_player_stats(app_state):
    """Open the Player Stats viewer."""
    store = app_state["profile_store"]
    all_players = store.list_players()

    if not all_players:
        print("[Stats] No player profiles found.")
        return

    app_state["app_screen"] = "PLAYER_STATS"
    app_state["stats_players"] = all_players
    app_state["stats_player_index"] = 0
    app_state["stats_step"] = "select" if len(all_players) > 1 else "view"
    app_state["stats_data"] = None
    app_state["stats_traits"] = []

    if len(all_players) == 1:
        _load_stats_for_player(app_state, all_players[0][0])


def _load_stats_for_player(app_state, name, mode_filter=None):
    """Build pattern tables and traits for viewing, optionally filtered by game mode."""
    store = app_state["profile_store"]

    # Use filtered build when a specific mode is selected
    if mode_filter and mode_filter != "All":
        tables = store.build_pattern_tables_filtered(name, mode_filter)
    else:
        tables = store.build_pattern_tables(name)

    if tables is None:
        app_state["stats_data"]            = None
        app_state["stats_traits"]          = ["No data available"]
        app_state["stats_step"]            = "view"
        app_state["stats_rounds"]          = []
        app_state["stats_sessions"]        = store.get_session_history(name)
        app_state["stats_current_player"]  = name
        return

    profile = store.load_profile(name)
    all_rounds = profile.get("rounds", []) if profile else []

    # Normalise legacy game_mode labels (e.g. "Fair Play" -> "FairPlay")
    _norm = {
        "Fair Play":  "FairPlay",
        "fair play":  "FairPlay",
        "Bluff Mode": "Cheat",
    }
    def _normalise_mode(m):
        return _norm.get(m, m)

    # Filtered round list for history dots
    if mode_filter and mode_filter != "All":
        filtered_rounds = [r for r in all_rounds
                           if _normalise_mode(r.get("game_mode", "")) == mode_filter]
    else:
        filtered_rounds = all_rounds

    # Win/loss/draw from the filtered set
    wins   = sum(1 for r in filtered_rounds if r.get("outcome") == "win")
    losses = sum(1 for r in filtered_rounds if r.get("outcome") == "lose")
    draws  = sum(1 for r in filtered_rounds if r.get("outcome") == "draw")
    total  = max(wins + losses + draws, 1)

    tables["wins"]     = wins
    tables["losses"]   = losses
    tables["draws"]    = draws
    tables["win_pct"]  = wins / total
    tables["loss_pct"] = losses / total
    tables["draw_pct"] = draws / total

    traits = store._compute_traits(tables)

    app_state["stats_data"]     = tables
    app_state["stats_traits"]   = traits
    app_state["stats_step"]     = "view"
    app_state["stats_rounds"]   = filtered_rounds
    app_state["stats_sessions"] = store.get_session_history(name)
    app_state["stats_current_player"] = name


def handle_player_stats_key(app_state, key):
    """Handle keys on the Player Stats screen."""
    step = app_state.get("stats_step", "select")
    _FILTERS = ["All", "FairPlay", "Challenge", "Cheat", "Clone"]
    _TABS    = ["overview", "history"]

    if step == "select":
        players = app_state.get("stats_players", [])
        if key in KEY_UP and players:
            app_state["stats_player_index"] = (app_state["stats_player_index"] - 1) % len(players)
        elif key in KEY_DOWN and players:
            app_state["stats_player_index"] = (app_state["stats_player_index"] + 1) % len(players)
        elif key in KEY_ENTER and players:
            name, _ = players[app_state["stats_player_index"]]
            app_state["stats_filter"] = "All"
            app_state["stats_tab"]    = "overview"
            _load_stats_for_player(app_state, name)
        elif key == KEY_ESC:
            open_menu(app_state)

    elif step == "view":
        data = app_state.get("stats_data")
        # Use stored player name - stays populated even when filtered data is None
        name = app_state.get("stats_current_player", "") or \
               (data.get("player_name", "") if data else "")

        if key == KEY_ESC:
            if len(app_state.get("stats_players", [])) > 1:
                app_state["stats_step"] = "select"
            else:
                open_menu(app_state)

        elif key in (ord("t"), ord("T")):
            # Toggle between overview and history tabs
            tabs = _TABS
            cur  = app_state.get("stats_tab", "overview")
            app_state["stats_tab"] = tabs[(tabs.index(cur) + 1) % len(tabs)]

        elif key in KEY_LEFT:
            # Cycle mode filter backwards
            if name:
                filters = _FILTERS
                cur_idx = filters.index(app_state.get("stats_filter", "All"))
                new_f   = filters[(cur_idx - 1) % len(filters)]
                app_state["stats_filter"] = new_f
                _load_stats_for_player(app_state, name, mode_filter=new_f)

        elif key in KEY_RIGHT:
            # Cycle mode filter forwards
            if name:
                filters = _FILTERS
                cur_idx = filters.index(app_state.get("stats_filter", "All"))
                new_f   = filters[(cur_idx + 1) % len(filters)]
                app_state["stats_filter"] = new_f
                _load_stats_for_player(app_state, name, mode_filter=new_f)

        elif key == ord("x") or key == ord("X"):
            if data:
                path = app_state["profile_store"].export_csv(name)
                if path:
                    print(f"[Stats] Exported to {path}")
                    app_state["collector_message"] = f"Exported: {path}"


TUTORIAL_STEPS = [
    {
        "id": "rock",
        "title": "STEP 1: ROCK",
        "instruction": "Make a FIST",
        "sub": "Close all fingers into a fist shape",
        "target_gesture": "Rock",
        "hold_frames": 10,
    },
    {
        "id": "paper",
        "title": "STEP 2: PAPER",
        "instruction": "OPEN your HAND",
        "sub": "Spread all five fingers wide",
        "target_gesture": "Paper",
        "hold_frames": 10,
    },
    {
        "id": "scissors",
        "title": "STEP 3: SCISSORS",
        "instruction": "Show SCISSORS",
        "sub": "Hold up index and middle finger",
        "target_gesture": "Scissors",
        "hold_frames": 10,
    },
    {
        "id": "pump",
        "title": "STEP 4: THE PUMP",
        "instruction": "Make a FIST and PUMP 4 times",
        "sub": "Move your fist up and down like a countdown",
        "target_gesture": "Rock",
        "hold_frames": 0,
    },
    {
        "id": "shoot",
        "title": "STEP 5: SHOOT!",
        "instruction": "THROW Rock, Paper, or Scissors!",
        "sub": "Change from fist to your throw",
        "target_gesture": None,
        "hold_frames": 0,
    },
    {
        "id": "done",
        "title": "YOU'RE READY!",
        "instruction": "You know the basics",
        "sub": "Press Enter to return to the menu",
        "target_gesture": None,
        "hold_frames": 0,
    },
]

# Voice-mode parallel - same IDs so the renderer can share status panel logic.
TUTORIAL_STEPS_VOICE = [
    {
        "id": "rock",
        "title": "STEP 1: ROCK",
        "instruction": 'Say  "ROCK"',
        "sub": "Speak clearly into your microphone",
        "target_gesture": "Rock",
        "hold_frames": 0,
        "voice_word": "Rock",
    },
    {
        "id": "paper",
        "title": "STEP 2: PAPER",
        "instruction": 'Say  "PAPER"',
        "sub": "Speak clearly into your microphone",
        "target_gesture": "Paper",
        "hold_frames": 0,
        "voice_word": "Paper",
    },
    {
        "id": "scissors",
        "title": "STEP 3: SCISSORS",
        "instruction": 'Say  "SCISSORS"',
        "sub": "Speak clearly into your microphone",
        "target_gesture": "Scissors",
        "hold_frames": 0,
        "voice_word": "Scissors",
    },
    {
        "id": "pump",
        "title": "STEP 4: COUNTDOWN",
        "instruction": 'Say  "ONE"  "TWO"  "THREE"',
        "sub": "Three words open the throw window",
        "target_gesture": None,
        "hold_frames": 0,
    },
    {
        "id": "shoot",
        "title": "STEP 5: THROW!",
        "instruction": "Say your throw",
        "sub": 'Say  "ROCK"  "PAPER"  or  "SCISSORS"',
        "target_gesture": None,
        "hold_frames": 0,
    },
    {
        "id": "done",
        "title": "YOU'RE READY!",
        "instruction": "You know voice controls",
        "sub": 'Say  "SELECT"  to return to menu',
        "target_gesture": None,
        "hold_frames": 0,
    },
]


def open_tutorial(app_state):
    """Open the interactive tutorial."""
    app_state["app_screen"] = "TUTORIAL"
    app_state["tutorial_step"] = 0
    app_state["tutorial_hold_count"] = 0
    app_state["tutorial_complete"] = False
    app_state["tutorial_detected"] = "Unknown"
    # Pump / countdown tracking for step 4.
    app_state["tutorial_pump_count"] = 0
    app_state["tutorial_pump_phase"] = "ready_for_down"
    app_state["tutorial_pump_top_y"] = None
    app_state["tutorial_pump_bot_y"] = None
    # Shoot tracking for step 5.
    app_state["tutorial_shot_gesture"] = None
    app_state["tutorial_shoot_visible_since"] = None
    # Which step list to use - depends on current input mode.
    app_state["tutorial_voice_mode"] = app_state.get("voice_mode_active", False)

    app_state["tracker"].reset()
    print(f"[Tutorial] Started ({'voice' if app_state['tutorial_voice_mode'] else 'physical'})")


def _tutorial_steps(app_state):
    """Return the correct step list for the current tutorial session."""
    if app_state.get("tutorial_voice_mode"):
        return TUTORIAL_STEPS_VOICE
    return TUTORIAL_STEPS


def update_tutorial(app_state, hand_state, tracker_state):
    """
    Update tutorial state based on current hand detection.
    Called every frame when app_screen == TUTORIAL.
    Physical-mode logic only - voice events are routed via
    handle_voice_tutorial_event() in the main loop.
    """
    if app_state.get("tutorial_voice_mode"):
        return   # voice mode advances via handle_voice_tutorial_event

    steps = _tutorial_steps(app_state)
    step_idx = app_state["tutorial_step"]
    if step_idx >= len(steps):
        return

    step = steps[step_idx]
    confirmed = tracker_state.get("confirmed_gesture", "Unknown")
    stable = tracker_state.get("stable_gesture", "Unknown")
    wrist_y = hand_state.get("wrist_y")

    app_state["tutorial_detected"] = confirmed if confirmed != "Unknown" else stable

    # --- Steps 1-3: Hold target gesture ---
    if step["id"] in ("rock", "paper", "scissors"):
        if confirmed == step["target_gesture"] or stable == step["target_gesture"]:
            app_state["tutorial_hold_count"] += 1
        else:
            app_state["tutorial_hold_count"] = max(0, app_state["tutorial_hold_count"] - 1)

        if app_state["tutorial_hold_count"] >= step["hold_frames"]:
            _advance_tutorial(app_state)

    # --- Step 4: Pump counting ---
    elif step["id"] == "pump":
        is_rock = confirmed == "Rock" or stable == "Rock"

        if is_rock and wrist_y is not None:
            phase = app_state["tutorial_pump_phase"]
            top_y = app_state["tutorial_pump_top_y"]
            bot_y = app_state["tutorial_pump_bot_y"]

            if top_y is None:
                app_state["tutorial_pump_top_y"] = wrist_y
                return
            if bot_y is None:
                app_state["tutorial_pump_bot_y"] = wrist_y

            if phase == "ready_for_down":
                app_state["tutorial_pump_top_y"] = min(top_y, wrist_y)
                if (wrist_y - app_state["tutorial_pump_top_y"]) >= 0.04:
                    app_state["tutorial_pump_count"] += 1
                    app_state["tutorial_pump_phase"] = "waiting_for_up"
                    app_state["tutorial_pump_bot_y"] = wrist_y

                    if app_state["tutorial_pump_count"] >= 4:
                        _advance_tutorial(app_state)
                        app_state["tracker"].clear_for_new_throw()

            elif phase == "waiting_for_up":
                app_state["tutorial_pump_bot_y"] = max(bot_y if bot_y else wrist_y, wrist_y)
                if (app_state["tutorial_pump_bot_y"] - wrist_y) >= 0.03:
                    app_state["tutorial_pump_phase"] = "ready_for_down"
                    app_state["tutorial_pump_top_y"] = wrist_y

    # --- Step 5: Throw any gesture ---
    elif step["id"] == "shoot":
        # Give the player time to read "SHOOT!" and react - require the step
        # to be visible for at least 2 seconds before accepting a throw.
        if app_state.get("tutorial_shoot_visible_since") is None:
            app_state["tutorial_shoot_visible_since"] = time.monotonic()
        wait_done = (time.monotonic() - app_state["tutorial_shoot_visible_since"]) >= 2.0

        if wait_done:
            if confirmed in ("Paper", "Scissors"):
                app_state["tutorial_shot_gesture"] = confirmed
                _advance_tutorial(app_state)
            elif stable in ("Paper", "Scissors"):
                app_state["tutorial_shot_gesture"] = stable
                _advance_tutorial(app_state)
            elif confirmed == "Rock" or stable == "Rock":
                app_state["tutorial_hold_count"] += 1
                if app_state["tutorial_hold_count"] >= 15:
                    app_state["tutorial_shot_gesture"] = "Rock"
                    _advance_tutorial(app_state)


def handle_voice_tutorial_event(app_state, event):
    """
    Advance the voice-mode tutorial based on incoming beat/throw events.

    Steps 1-3 (rock/paper/scissors): advance when the correct throw word is spoken.
    Step 4 (countdown):              advance pump_count on one/two/three; auto-advance at 3.
    Step 5 (shoot):                  any throw word completes the step.
    Step 6 (done):                   handled by the nav "select" action in handle_voice_nav.
    """
    steps = TUTORIAL_STEPS_VOICE
    step_idx = app_state.get("tutorial_step", 0)
    if step_idx >= len(steps):
        return

    step = steps[step_idx]
    step_id = step["id"]

    if event["type"] == "throw":
        gesture = event["gesture"]
        app_state["tutorial_detected"] = gesture

        if step_id in ("rock", "paper", "scissors"):
            if gesture == step.get("voice_word") or gesture == step.get("target_gesture"):
                app_state["tutorial_shot_gesture"] = gesture
                _advance_tutorial(app_state)

        elif step_id == "shoot":
            app_state["tutorial_shot_gesture"] = gesture
            _advance_tutorial(app_state)

    elif event["type"] == "beat":
        word = event["word"]

        if step_id == "pump" and word in ("one", "two", "three"):
            # Map one→1, two→2, three→3 so out-of-order speech still advances
            word_to_num = {"one": 1, "two": 2, "three": 3}
            target_count = word_to_num[word]
            # Only advance if this word is the next expected beat
            if target_count == app_state["tutorial_pump_count"] + 1:
                app_state["tutorial_pump_count"] = target_count
                if app_state["tutorial_pump_count"] >= 3:
                    _advance_tutorial(app_state)


def _advance_tutorial(app_state):
    """Move to the next tutorial step."""
    app_state["tutorial_step"] += 1
    app_state["tutorial_hold_count"] = 0
    steps = _tutorial_steps(app_state)
    app_state["tutorial_complete"] = app_state["tutorial_step"] >= len(steps) - 1

    if app_state["tutorial_step"] < len(steps):
        step = steps[app_state["tutorial_step"]]
        print(f"[Tutorial] Step: {step['title']}")


def handle_tutorial_key(app_state, key):
    """Handle keys on the tutorial screen."""
    if key == KEY_ESC:
        open_menu(app_state)
    elif key in KEY_ENTER:
        step_idx = app_state.get("tutorial_step", 0)
        if step_idx >= len(TUTORIAL_STEPS) - 1:
            open_menu(app_state)


def handle_voice_nav(app_state, action):
    """
    Dispatch a voice navigation action to the correct screen handler.

    Called every frame for all voice "nav" events regardless of screen.
    Beat/throw events are handled separately inside the GAME block.

    Actions:
        up / down / left / right  - scroll / change value
        select                    - confirm / enter
        back                      - ESC / cancel
        quit                      - quit the app
        restart                   - restart game (Reflex, Snake, etc.)
        start                     - begin session (Gesture Trainer, Snake)
        next                      - advance tutorial step
        commentary                - toggle commentary overlay
        simulations               - open Simulations hub
        gamemodes                 - open Game Category screen
        snake / squid / simon / bluff / reflex / rehab / race / rpsls
                                  - direct game shortcuts
        cheat / fair / challenge / clone / stats / tutorial / settings / features
                                  - direct menu shortcuts
    """
    screen = app_state["app_screen"]

    # ── Global actions (work everywhere) ──────────────────────────────────
    if action == "quit":
        return "quit"

    if action == "commentary":
        eng = app_state.get("commentary_engine")
        if eng and app_state["app_screen"] == "GAME":
            on = eng.toggle()
            app_state["collector_message"] = f"Commentary {'ON' if on else 'OFF'}"
        return None

    if action == "restart":
        if screen == "GAME":
            mode = app_state["play_mode"]
            if mode == "ReflexSolo":
                app_state["reflex_solo_controller"].reset()
            elif mode == "ArcadeSnake":
                gs = app_state.get("game_state") or {}
                if gs.get("state") == "GAME_OVER":
                    app_state["arcade_snake_controller"].reset()
            elif mode in ("FairPlay", "Challenge", "Clone", "BluffMode",
                          "PredictionRace", "RPSLS"):
                reset_all_modes(app_state)
        return None

    if action == "start":
        if screen == "GAME":
            mode = app_state["play_mode"]
            if mode == "GestureRehab":
                app_state["gesture_rehab_controller"].start_session()
        return None

    if action == "next":
        if screen == "TUTORIAL":
            steps = _tutorial_steps(app_state)
            step  = app_state.get("tutorial_step", 0)
            app_state["tutorial_step"] = min(step + 1, len(steps) - 1)
        elif screen == "RPSLS_TUTORIAL":
            step = app_state.get("rpsls_tutorial_step", 0)
            if step < 5:
                app_state["rpsls_tutorial_step"] = step + 1
            else:
                start_game(app_state, "RPSLS", from_category=True)
        return None

    # ── MENU ──────────────────────────────────────────────────────────────
    if screen == "MENU":
        if action == "up":
            app_state["menu_index"] = (app_state["menu_index"] - 1) % len(app_state["menu_items"])
        elif action == "down":
            app_state["menu_index"] = (app_state["menu_index"] + 1) % len(app_state["menu_items"])
        elif action == "select":
            return activate_menu_item(app_state)
        elif action == "back":
            pass
        elif action == "gamemodes":
            app_state["app_screen"]       = "GAME_CATEGORY"
            app_state["in_game_category"] = False
            app_state["game_category_index"] = 0
            app_state["game_mode_index"]  = 0
        elif action == "simulations":
            app_state["app_screen"] = "SIMULATIONS"
        # Direct mode shortcuts
        elif action == "cheat":
            start_game(app_state, "Cheat")
        elif action == "fair":
            start_game(app_state, "FairPlay")
        elif action == "challenge":
            start_game(app_state, "Challenge")
        elif action == "clone":
            open_clone_setup(app_state)
        elif action == "stats":
            open_player_stats(app_state)
        elif action == "tutorial":
            open_tutorial(app_state)
        elif action == "settings":
            open_settings(app_state)
        elif action == "features":
            open_features(app_state)
        elif action == "simulations":
            app_state["app_screen"] = "SIMULATIONS"
        elif action == "snake":
            start_game(app_state, "ArcadeSnake")
        elif action == "squid":
            start_game(app_state, "SquidGame")
        elif action == "simon":
            start_game(app_state, "SimonSaysSolo")
        elif action == "bluff":
            start_game(app_state, "BluffMode")
        elif action == "reflex":
            start_game(app_state, "ReflexSolo")
        elif action == "rehab":
            start_game(app_state, "GestureRehab")
        elif action == "race":
            start_game(app_state, "PredictionRace")
        elif action == "rpsls":
            app_state["app_screen"] = "RPSLS_SIDE_NOTICE"

    # ── GAME_CATEGORY ──────────────────────────────────────────────────────
    elif screen == "GAME_CATEGORY":
        in_mode = app_state.get("in_game_category", False)
        if action == "back":
            if in_mode:
                app_state["in_game_category"] = False
            else:
                open_menu(app_state)
        elif action == "up":
            if in_mode:
                n = len(GAME_CATEGORIES[app_state["game_category_index"]]["modes"])
                app_state["game_mode_index"] = (app_state["game_mode_index"] - 1) % n
            else:
                n = len(GAME_CATEGORIES)
                app_state["game_category_index"] = (app_state["game_category_index"] - 1) % n
        elif action == "down":
            if in_mode:
                n = len(GAME_CATEGORIES[app_state["game_category_index"]]["modes"])
                app_state["game_mode_index"] = (app_state["game_mode_index"] + 1) % n
            else:
                n = len(GAME_CATEGORIES)
                app_state["game_category_index"] = (app_state["game_category_index"] + 1) % n
        elif action == "select":
            if not in_mode:
                app_state["in_game_category"] = True
                app_state["game_mode_index"]  = 0
            else:
                cat  = GAME_CATEGORIES[app_state["game_category_index"]]
                mode = cat["modes"][app_state["game_mode_index"]]
                if mode.get("key") == "RPSLS":
                    app_state["app_screen"] = "RPSLS_SIDE_NOTICE"
                else:
                    start_game(app_state, mode["key"], from_category=True)
        # Direct game name shortcuts inside category screen too
        elif action in ("snake", "squid", "simon", "bluff",
                        "reflex", "rehab", "race", "rpsls"):
            mode_map = {
                "snake": "ArcadeSnake", "squid": "SquidGame",
                "simon": "SimonSaysSolo", "bluff": "BluffMode",
                "reflex": "ReflexSolo", "rehab": "GestureRehab",
                "race": "PredictionRace",
            }
            if action == "rpsls":
                app_state["app_screen"] = "RPSLS_SIDE_NOTICE"
            elif action in mode_map:
                start_game(app_state, mode_map[action], from_category=True)

    # ── SIMULATIONS ────────────────────────────────────────────────────────
    elif screen == "SIMULATIONS":
        if action == "back":
            open_menu(app_state)
        elif action == "up":
            n = app_state.get("sim_tab_index", 0)
            app_state["sim_tab_index"] = max(0, n - 1)
        elif action == "down":
            n = app_state.get("sim_tab_index", 0)
            app_state["sim_tab_index"] = min(2, n + 1)
        elif action == "select":
            # Same as pressing Enter on the simulation hub
            pass   # handled by keyboard; voice fires the same result

    # ── SIMULATION results ────────────────────────────────────────────────
    elif screen == "SIMULATION":
        if action in ("back", "select"):
            app_state["app_screen"] = "SIMULATIONS"

    # ── FEATURES ──────────────────────────────────────────────────────────
    elif screen == "FEATURES":
        if action == "up":
            app_state["features_index"] = (app_state["features_index"] - 1) % len(FEATURES_SCHEMA)
        elif action == "down":
            app_state["features_index"] = (app_state["features_index"] + 1) % len(FEATURES_SCHEMA)
        elif action == "left":
            item = FEATURES_SCHEMA[app_state["features_index"]]
            if item.get("key") != "__back__":
                apply_feature_toggle(app_state, item["key"], direction=-1)
        elif action in ("select", "right"):
            item = FEATURES_SCHEMA[app_state["features_index"]]
            if item.get("key") == "__back__":
                open_menu(app_state)
            elif item.get("key") == "__personalities__":
                app_state["app_screen"] = "PERSONALITY_SELECT"
                cur = app_state["config"].get("ai_personality", "Normal")
                app_state["personality_index"] = PERSONALITY_NAMES.index(cur) \
                    if cur in PERSONALITY_NAMES else 0
            else:
                apply_feature_toggle(app_state, item["key"], direction=1)
        elif action == "back":
            open_menu(app_state)

    # ── PERSONALITY_SELECT ────────────────────────────────────────────────
    elif screen == "PERSONALITY_SELECT":
        if action == "up":
            app_state["personality_index"] = (app_state["personality_index"] - 1) % len(PERSONALITY_NAMES)
        elif action == "down":
            app_state["personality_index"] = (app_state["personality_index"] + 1) % len(PERSONALITY_NAMES)
        elif action in ("select", "right"):
            chosen = PERSONALITY_NAMES[app_state["personality_index"]]
            app_state["config"]["ai_personality"] = chosen
            for key in ("fair_controller", "challenge_controller",
                        "clone_controller", "bluff_controller"):
                ctrl = app_state.get(key)
                if ctrl and hasattr(ctrl, "ai") and hasattr(ctrl.ai, "set_personality"):
                    ctrl.ai.set_personality(chosen)
            app_state["app_screen"] = "FEATURES"
        elif action == "back":
            app_state["app_screen"] = "FEATURES"

    # ── SETTINGS ──────────────────────────────────────────────────────────
    elif screen == "SETTINGS":
        if action == "up":
            app_state["settings_index"] = (app_state["settings_index"] - 1) % len(SETTINGS_SCHEMA)
        elif action == "down":
            app_state["settings_index"] = (app_state["settings_index"] + 1) % len(SETTINGS_SCHEMA)
        elif action == "left":
            apply_setting_change(app_state, -1)
        elif action == "right":
            apply_setting_change(app_state, 1)
        elif action in ("select", "back"):
            activate_settings_item(app_state)
            if action == "back":
                open_menu(app_state)

    # ── CLONE_SETUP ───────────────────────────────────────────────────────
    elif screen == "CLONE_SETUP":
        step = app_state.get("clone_step", "enter_name")
        if step == "enter_name":
            if action == "select":
                buf = app_state.get("clone_text_buffer", "").strip()
                if buf:
                    handle_clone_setup_key(app_state, 10)
                else:
                    app_state["clone_message"] = "Say your name on the keyboard first, then say SELECT"
            elif action == "back":
                open_menu(app_state)
        elif step == "select_opponent":
            if action == "up":
                available = app_state.get("clone_available", [])
                if available:
                    app_state["clone_opponent_index"] = (
                        (app_state["clone_opponent_index"] - 1) % len(available))
            elif action == "down":
                available = app_state.get("clone_available", [])
                if available:
                    app_state["clone_opponent_index"] = (
                        (app_state["clone_opponent_index"] + 1) % len(available))
            elif action == "select":
                available = app_state.get("clone_available", [])
                if available:
                    name, _ = available[app_state["clone_opponent_index"]]
                    _start_clone_game(app_state, name)
            elif action == "back":
                app_state["clone_step"] = "enter_name"
        elif step == "no_profiles":
            if action in ("select", "back"):
                open_menu(app_state)

    # ── PLAYER_STATS ──────────────────────────────────────────────────────
    elif screen == "PLAYER_STATS":
        step = app_state.get("stats_step", "select")
        if step == "select":
            players = app_state.get("stats_players", [])
            if action == "up" and players:
                app_state["stats_player_index"] = (
                    (app_state["stats_player_index"] - 1) % len(players))
            elif action == "down" and players:
                app_state["stats_player_index"] = (
                    (app_state["stats_player_index"] + 1) % len(players))
            elif action == "select" and players:
                name, _ = players[app_state["stats_player_index"]]
                _load_stats_for_player(app_state, name)
            elif action == "back":
                open_menu(app_state)
        elif step == "view":
            if action == "back":
                players = app_state.get("stats_players", [])
                if len(players) > 1:
                    app_state["stats_step"] = "select"
                else:
                    open_menu(app_state)

    # ── TUTORIAL ──────────────────────────────────────────────────────────
    elif screen == "TUTORIAL":
        if action == "back":
            open_menu(app_state)
        elif action in ("select", "next"):
            steps   = _tutorial_steps(app_state)
            step_idx = app_state.get("tutorial_step", 0)
            if step_idx >= len(steps) - 1:
                open_menu(app_state)
            else:
                app_state["tutorial_step"] = step_idx + 1

    # ── RPSLS_SIDE_NOTICE ─────────────────────────────────────────────────
    elif screen == "RPSLS_SIDE_NOTICE":
        if action in ("select", "next", "start"):
            app_state["app_screen"] = "RPSLS_TUTORIAL"
            app_state["rpsls_tutorial_step"] = 0
        elif action == "back":
            open_menu(app_state)

    # ── RPSLS_TUTORIAL ────────────────────────────────────────────────────
    elif screen == "RPSLS_TUTORIAL":
        if action in ("select", "next"):
            step = app_state.get("rpsls_tutorial_step", 0)
            if step < 5:
                app_state["rpsls_tutorial_step"] = step + 1
            else:
                start_game(app_state, "RPSLS", from_category=True)
        elif action == "back":
            step = app_state.get("rpsls_tutorial_step", 0)
            if step > 0:
                app_state["rpsls_tutorial_step"] = step - 1
            else:
                open_menu(app_state)

    # ── GAME ──────────────────────────────────────────────────────────────
    elif screen == "GAME":
        if action == "back":
            open_menu(app_state)

    return None


def _run_gesture_nav(app_state, hand_state, now, item_count, set_index_fn,
                     content_top=0.44, content_bottom=0.83,
                     adjust_items=None, adjust_fn=None):
    sp     = app_state.get("sound_player")
    result = None
    for ev in app_state["gesture_nav"].update(
        hand_state, now, item_count,
        content_top=content_top, content_bottom=content_bottom,
        adjust_items=adjust_items,
    ):
        if ev["type"] == "hover":
            prev_idx = app_state["gesture_nav"]._last_item_idx
            set_index_fn(ev["item_index"])
            if prev_idx != -1 and ev["item_index"] != prev_idx and sp:
                sp.play("menu_move")
        elif ev["type"] == "select":
            if sp: sp.play("menu_select")
            result = handle_voice_nav(app_state, "select")
        elif ev["type"] == "adjust" and adjust_fn is not None:
            adjust_fn(ev["direction"])
    return result


def toggle_display_mode(app_state):
    app_state["display_mode"] = (
        "Diagnostic" if app_state["display_mode"] == "Game" else "Game"
    )
    update_challenge_logger_context(app_state)
    print(f"Display mode: {app_state['display_mode']}")


def switch_play_mode(app_state, new_mode):
    if new_mode not in {"Cheat", "FairPlay", "Challenge", "Clone"}:
        return

    if new_mode == app_state["play_mode"] and app_state["app_screen"] == "GAME":
        return

    start_game(app_state, new_mode)


def get_active_controller(app_state):
    if app_state["play_mode"] == "FairPlay":
        return app_state["fair_controller"]
    if app_state["play_mode"] == "Challenge":
        return app_state["challenge_controller"]
    if app_state["play_mode"] == "Clone":
        return app_state["clone_controller"]
    if app_state["play_mode"] == "TwoPlayerPvP":
        return app_state["pvp_controller"]
    if app_state["play_mode"] == "PvPvAI":
        return app_state["pvpvai_controller"]
    if app_state["play_mode"] == "ReflexSolo":
        return app_state["reflex_solo_controller"]
    if app_state["play_mode"] == "ReflexTwoPlayer":
        return app_state["reflex_2p_controller"]
    if app_state["play_mode"] == "BluffMode":
        return app_state["bluff_controller"]
    if app_state["play_mode"] == "SimonSaysSolo":
        return app_state["simon_solo_controller"]
    if app_state["play_mode"] == "SimonSays2P":
        return app_state["simon_2p_controller"]
    if app_state["play_mode"] == "SquidGame":
        return app_state["squid_controller"]
    if app_state["play_mode"] == "SquidGame2P":
        return app_state["squid_2p_controller"]
    if app_state["play_mode"] == "PredictionRace":
        return app_state["prediction_race_controller"]
    if app_state["play_mode"] == "GestureRehab":
        return app_state["gesture_rehab_controller"]
    if app_state["play_mode"] == "ArcadeSnake":
        return app_state["arcade_snake_controller"]
    if app_state["play_mode"] == "RPSLS":
        return app_state["rpsls_controller"]
    return app_state["cheat_controller"]


def apply_setting_change(app_state, direction):
    item = SETTINGS_SCHEMA[app_state["settings_index"]]

    if item["type"] in ("action", "text"):
        return

    key = item["key"]
    config = app_state["config"]

    if item["type"] == "choice":
        options = item["options"]
        current_index = options.index(config[key])
        new_index = (current_index + direction) % len(options)
        config[key] = options[new_index]

    elif item["type"] == "float":
        value = config[key] + item["step"] * direction
        value = max(item["min"], min(item["max"], value))
        config[key] = round(value, 2)

    app_state["config"] = save_config(config)
    app_state["display_mode"] = app_state["config"]["default_display_mode"]
    update_challenge_logger_context(app_state)

    if key == "camera_resolution" and app_state.get("cap") is not None:
        apply_camera_resolution(app_state["cap"], app_state["config"])

    rebuild_controllers(app_state)
    _apply_voice_mode(app_state)


def _launch_rpsls(app_state, from_category=False, diagnostic=False):
    """
    Launch RPSLS with a side-mode notice.
    Always forces hand_orientation to 'Side' — front-on detection is unreliable
    for Lizard and Spock.  Shows a one-time notice screen before the game starts.
    """
    # Force Side orientation for RPSLS
    prev_orientation = app_state["config"].get("hand_orientation", "Side")
    app_state["config"]["hand_orientation"] = "Side"
    save_config(app_state["config"])

    # Store whether we switched so the notice can mention it
    was_front = prev_orientation == "Front"
    app_state["_rpsls_was_front_on"] = was_front
    app_state["_rpsls_diagnostic"]   = diagnostic
    app_state["_came_from_category"] = from_category

    # Show the side-mode notice screen first
    app_state["app_screen"]             = "RPSLS_SIDE_NOTICE"
    app_state["_rpsls_notice_until"]    = None
    app_state["_rpsls_ticked"]          = set()
    app_state["_rpsls_dwell_gesture"]   = None
    app_state["_rpsls_dwell_since"]     = 0.0


def activate_menu_item(app_state):
    # ── Level 3: inside a category's mode list ─────────────────────────────
    if app_state.get("in_game_category"):
        cat = GAME_CATEGORIES[app_state["game_category_index"]]
        modes = cat["modes"]
        label, action = modes[app_state["game_mode_index"]]
        app_state["in_game_category"] = False
        app_state["app_screen"] = "MENU"
        if action == "Clone":
            open_clone_setup(app_state)
        elif action in {"Cheat", "FairPlay", "Challenge", "TwoPlayerPvP", "PvPvAI",
                        "ReflexSolo", "ReflexTwoPlayer", "BluffMode",
                        "SimonSaysSolo", "SimonSays2P", "SquidGame", "SquidGame2P",
                        "PredictionRace", "GestureRehab", "ArcadeSnake"}:
            start_game(app_state, action, from_category=True)
        elif action == "RPSLS":
            _launch_rpsls(app_state, from_category=True)
        elif action == "RPSLSTutorial":
            app_state["app_screen"]          = "RPSLS_TUTORIAL"
            app_state["rpsls_tutorial_step"] = 0
            app_state["_came_from_category"] = True
        elif action == "RPSLSDiagnostic":
            _launch_rpsls(app_state, from_category=True, diagnostic=True)
        return None

    # ── Level 2: inside Game Modes category list (select a category) ───────
    if app_state.get("in_submenu"):
        # Legacy path - shouldn't be reached in new UI but kept for voice-nav
        app_state["in_submenu"] = False
        return None

    # ── Level 1: main menu ─────────────────────────────────────────────────
    label, action = app_state["menu_items"][app_state["menu_index"]]

    if action == "GameModes":
        app_state["app_screen"] = "GAME_CATEGORY"
        app_state["game_category_index"] = 0
        app_state["game_mode_index"]     = 0
        app_state["in_game_category"]    = False
        return None
    elif action == "Simulations":
        app_state["app_screen"]    = "SIMULATIONS"
        app_state["sim_tab_index"] = 0
        return None
    elif action == "Stats":
        open_player_stats(app_state)
    elif action == "Tutorial":
        open_tutorial(app_state)
    elif action == "Settings":
        open_settings(app_state)
    elif action == "Features":
        open_features(app_state)
    elif action == "Quit":
        return "quit"

    return None


def _launch_simulation(app_state):
    """
    Run a high-fidelity simulation (~100,000 rounds) in a background thread.
    6 player strategies x 3 AIs x 55 runs x 100 rounds = 99,000 rounds.
    Target runtime ~10 seconds on M-series Mac.
    """
    app_state["app_screen"] = "SIMULATION"
    app_state["sim_state"]  = {
        "status": "running", "progress": 0.0,
        "progress_text": "Initialising...", "results": None, "error": None,
    }

    def _run():
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import simulation_mode as _sm
            from simulation_mode import PLAYER_STRATEGIES, AI_OPPONENTS

            # Scale to ~100k rounds: 6 strategies x 3 AIs x 55 runs x 100 rounds = 99,000
            RUNS   = 55
            ROUNDS = 100
            total  = len(PLAYER_STRATEGIES) * len(AI_OPPONENTS) * RUNS
            done   = [0]

            _orig = _sm.run_single_game
            def _patched(strategy, ai_type, rounds):
                result = _orig(strategy, ai_type, rounds)
                done[0] += 1
                pct = done[0] / max(total, 1)
                app_state["sim_state"].update({
                    "progress":      pct,
                    "progress_text": f"{strategy}  vs  {ai_type}  ({done[0]:,}/{total:,} runs)",
                })
                return result
            _sm.run_single_game = _patched

            results = _sm.run_simulation(
                runs_per_combo=RUNS,
                rounds_per_run=ROUNDS,
                save_excel=True,
            )
            _sm.run_single_game = _orig

            # Enrich results with additional conclusions
            combos = results.get("combo_results", [])
            total_rounds = results.get("total_rounds", 0)

            # Find the most balanced matchup (closest to 50/50)
            if combos:
                most_balanced = min(combos, key=lambda c: abs(c["player_win_rate"] - 0.5))
                results["most_balanced"] = (
                    f"{most_balanced['strategy']} vs {most_balanced['ai']} "
                    f"({most_balanced['player_win_rate']:.1%} player)"
                )

                # AI win rates averaged across all strategies
                ai_avg = {}
                for ai in AI_OPPONENTS:
                    rows = [c for c in combos if c["ai"] == ai]
                    if rows:
                        ai_avg[ai] = sum(c["robot_win_rate"] for c in rows) / len(rows)
                results["ai_win_rates"] = ai_avg

                # Strategy win rates averaged across all AIs
                strat_avg = {}
                for s in PLAYER_STRATEGIES:
                    rows = [c for c in combos if c["strategy"] == s]
                    if rows:
                        strat_avg[s] = sum(c["player_win_rate"] for c in rows) / len(rows)
                results["strategy_win_rates"] = strat_avg
                results["total_rounds_actual"] = total_rounds

            app_state["sim_state"].update({
                "status": "done", "progress": 1.0, "results": results,
            })
            # Auto-update the research report with fresh simulation data
            _run_report_updater_bg()
        except Exception as exc:
            import traceback
            app_state["sim_state"].update({
                "status": "error", "error": f"{exc}\n{traceback.format_exc()[-200:]}",
            })

    _io_worker.submit(_run)


def _launch_pvpvai_simulation(app_state):
    """
    Simulate the 1v1v1 PvPvAI format across all strategy pairings.
    Tests every combination of P1 strategy x P2 strategy x AI type.
    Scoring: beat 1 = +1 pt, beat 2 = +2 pts; first to 5 wins the match.
    """
    app_state["app_screen"] = "SIMULATION"
    app_state["sim_state"]  = {
        "status": "running", "progress": 0.0,
        "progress_text": "Initialising 3-way simulation...",
        "results": None, "error": None,
    }

    def _run():
        try:
            import sys, random
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from simulation_mode import SimulatedPlayer, create_ai_opponent, PLAYER_STRATEGIES, BEATS
            from two_player_state import PvPvAIController

            RUNS        = 30     # runs per combo - fast enough for ~10s
            ROUNDS      = 100    # rounds per run
            WIN_TARGET  = 5

            combos = [(s1, s2, ai)
                      for s1 in PLAYER_STRATEGIES
                      for s2 in PLAYER_STRATEGIES
                      for ai in ("random", "fair_play", "challenge")]
            total  = len(combos) * RUNS
            done   = [0]

            # Results: {(s1,s2,ai): {p1_wins, p2_wins, ai_wins, draws, total_rounds}}
            results = {}

            def _cmp(a, b):
                if a == b:    return "draw"
                if BEATS[a] == b: return "win"
                return "lose"

            def _score_three(g1, g2, g_ai):
                p1 = (1 if _cmp(g1,g2)=="win" else 0) + (1 if _cmp(g1,g_ai)=="win" else 0)
                p2 = (1 if _cmp(g2,g1)=="win" else 0) + (1 if _cmp(g2,g_ai)=="win" else 0)
                pa = (1 if _cmp(g_ai,g1)=="win" else 0) + (1 if _cmp(g_ai,g2)=="win" else 0)
                return p1, p2, pa

            for s1, s2, ai_type in combos:
                key = (s1, s2, ai_type)
                p1_match_wins = p2_match_wins = ai_match_wins = 0

                for _ in range(RUNS):
                    player1 = SimulatedPlayer(strategy=s1)
                    player2 = SimulatedPlayer(strategy=s2)
                    ai_inst, ai_fn = create_ai_opponent(ai_type)
                    if ai_fn is None:
                        continue
                    if ai_inst and hasattr(ai_inst, "reset"):
                        ai_inst.reset()

                    p1_pts = p2_pts = ai_pts = 0
                    p1_last = p2_last = ai_last = None
                    p1_outcome = p2_outcome = None

                    for rn in range(1, ROUNDS + 1):
                        g1  = player1.choose_move(p1_outcome, p1_last, ai_last)
                        g2  = player2.choose_move(p2_outcome, p2_last, ai_last)
                        # AI uses p1 history as a proxy
                        ai_hist = [{"round_number": i, "player_gesture": p1_last,
                                    "player_outcome": p1_outcome}
                                   for i in range(1)] if p1_last else []
                        g_ai = ai_fn(ai_hist, 0, rn)

                        r1, r2, ra = _score_three(g1, g2, g_ai)
                        p1_pts += r1; p2_pts += r2; ai_pts += ra

                        p1_outcome = "win" if r1 > 0 else "lose"
                        p2_outcome = "win" if r2 > 0 else "lose"
                        p1_last = g1; p2_last = g2; ai_last = g_ai

                        if p1_pts >= WIN_TARGET or p2_pts >= WIN_TARGET or ai_pts >= WIN_TARGET:
                            break

                    if   p1_pts >= WIN_TARGET: p1_match_wins += 1
                    elif p2_pts >= WIN_TARGET: p2_match_wins += 1
                    else:                      ai_match_wins += 1

                    done[0] += 1
                    app_state["sim_state"].update({
                        "progress":      done[0] / max(total, 1),
                        "progress_text": f"3-way: {s1} vs {s2} vs {ai_type}  ({done[0]:,}/{total:,})",
                    })

                results[key] = {
                    "p1_strategy": s1, "p2_strategy": s2, "ai_type": ai_type,
                    "p1_win_rate": p1_match_wins / RUNS,
                    "p2_win_rate": p2_match_wins / RUNS,
                    "ai_win_rate": ai_match_wins / RUNS,
                    "runs": RUNS,
                }

            # Summary stats
            all_res = list(results.values())
            ai_avg  = {}
            for ai in ("random","fair_play","challenge"):
                rows = [r for r in all_res if r["ai_type"] == ai]
                ai_avg[ai] = sum(r["ai_win_rate"] for r in rows) / max(len(rows),1)

            strat_avg = {}
            for s in PLAYER_STRATEGIES:
                rows = [r for r in all_res if r["p1_strategy"]==s or r["p2_strategy"]==s]
                strat_avg[s] = sum(
                    (r["p1_win_rate"] if r["p1_strategy"]==s else r["p2_win_rate"])
                    for r in rows) / max(len(rows),1)

            best_ai   = max(ai_avg, key=ai_avg.get)
            best_strat = max(strat_avg, key=strat_avg.get)

            # Most balanced: closest to three-way 33%
            most_balanced_key = min(results, key=lambda k: (
                abs(results[k]["p1_win_rate"]-0.333) +
                abs(results[k]["p2_win_rate"]-0.333) +
                abs(results[k]["ai_win_rate"]-0.333)
            ))
            mb = results[most_balanced_key]

            app_state["sim_state"].update({
                "status": "done", "progress": 1.0,
                "results": {
                    "mode":            "pvpvai",
                    "best_ai":         best_ai,
                    "best_strategy":   best_strat,
                    "ai_win_rates":    ai_avg,
                    "strategy_win_rates": strat_avg,
                    "combo_results": [
                        {"strategy":        r["p1_strategy"],
                         "ai":             r["ai_type"],
                         "player_win_rate": (r["p1_win_rate"]+r["p2_win_rate"])/2,
                         "robot_win_rate":  r["ai_win_rate"],
                         "draw_rate":       0.0,
                         "runs":            r["runs"]}
                        for r in all_res
                    ],
                    "most_balanced": (
                        f"{mb['p1_strategy']} vs {mb['p2_strategy']} vs {mb['ai_type']} "
                        f"(P1:{mb['p1_win_rate']:.0%} P2:{mb['p2_win_rate']:.0%} "
                        f"AI:{mb['ai_win_rate']:.0%})"
                    ),
                    "elapsed_seconds": 0,
                    "total_rounds_actual": total * ROUNDS,
                },
            })
        except Exception as exc:
            import traceback
            app_state["sim_state"].update({
                "status": "error",
                "error": f"{exc}\n{traceback.format_exc()[-300:]}",
            })

    _io_worker.submit(_run)


def _launch_tournament_simulation(app_state):
    """
    Clone Tournament: runs every saved player profile against every other
    in a round-robin. Each matchup plays ROUNDS rounds using PlayerCloneAI
    on both sides. Produces a win-rate leaderboard.
    """
    app_state["app_screen"] = "SIMULATION"
    app_state["sim_state"]  = {
        "status": "running", "progress": 0.0,
        "progress_text": "Loading player profiles...",
        "results": None, "error": None,
    }

    def _run():
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from player_profile_store import PlayerProfileStore
            from player_clone_ai import PlayerCloneAI

            ROUNDS = 200

            BEATS = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
            COUNTER = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}

            def play_match(ai_a, ai_b, rounds):
                hist_a, hist_b = [], []
                wins_a = wins_b = draws = 0
                for rn in range(1, rounds + 1):
                    move_a = ai_a.choose_robot_move(hist_a, rn)
                    move_b = ai_b.choose_robot_move(hist_b, rn)
                    if move_a == move_b:
                        result_a = result_b = "draw"
                        draws += 1
                    elif BEATS[move_a] == move_b:
                        result_a, result_b = "win", "lose"
                        wins_a += 1
                    else:
                        result_a, result_b = "lose", "win"
                        wins_b += 1
                    hist_a.append({"player_gesture": move_a, "robot_gesture": move_b,
                                   "player_outcome": result_a})
                    hist_b.append({"player_gesture": move_b, "robot_gesture": move_a,
                                   "player_outcome": result_b})
                total = wins_a + wins_b + draws
                return wins_a / total, wins_b / total, draws / total

            store   = PlayerProfileStore()
            clones  = store.list_playable_clones()

            if len(clones) < 2:
                app_state["sim_state"].update({
                    "status": "done", "progress": 1.0,
                    "results": {
                        "mode": "tournament",
                        "error_msg": (
                            f"Need at least 2 players with 30+ rounds to run a tournament. "
                            f"Currently have: {len(clones)} eligible player(s)."
                        ),
                        "leaderboard": [],
                    }
                })
                return

            # Build clone AIs
            clone_ais = {}
            for name in clones:
                tables = store.build_pattern_tables(name)
                if tables:
                    clone_ais[name] = PlayerCloneAI(tables, accuracy=0.90)

            n       = len(clone_ais)
            names   = list(clone_ais.keys())
            pairs   = [(a, b) for i, a in enumerate(names)
                               for b in names[i+1:]]
            total_p = len(pairs)

            # Win tracking
            wins   = {n: 0 for n in names}
            played = {n: 0 for n in names}
            match_results = []

            for idx, (name_a, name_b) in enumerate(pairs):
                app_state["sim_state"].update({
                    "progress":      idx / max(total_p, 1),
                    "progress_text": f"{name_a} vs {name_b}...",
                })
                wr_a, wr_b, dr = play_match(
                    clone_ais[name_a], clone_ais[name_b], ROUNDS)
                wins[name_a]   += wr_a
                wins[name_b]   += wr_b
                played[name_a] += 1
                played[name_b] += 1
                match_results.append({
                    "p1": name_a, "p2": name_b,
                    "p1_wr": round(wr_a, 3),
                    "p2_wr": round(wr_b, 3),
                    "draw_rate": round(dr, 3),
                })

            # Build leaderboard sorted by average win rate
            leaderboard = []
            for name in names:
                n_played = played[name]
                avg_wr   = wins[name] / n_played if n_played else 0.0
                rounds_recorded = len(store.load_profile(name).get("rounds", []))
                leaderboard.append({
                    "player":   name,
                    "avg_wr":   round(avg_wr, 3),
                    "played":   n_played,
                    "rounds":   rounds_recorded,
                })
            leaderboard.sort(key=lambda x: -x["avg_wr"])

            app_state["sim_state"].update({
                "status": "done", "progress": 1.0,
                "results": {
                    "mode":           "tournament",
                    "leaderboard":    leaderboard,
                    "match_results":  match_results,
                    "n_players":      n,
                    "rounds_per_match": ROUNDS,
                    "champion":       leaderboard[0]["player"] if leaderboard else "?",
                    "error_msg":      "",
                },
            })

        except Exception as exc:
            import traceback
            app_state["sim_state"].update({
                "status": "error",
                "error":  f"{exc}\n{traceback.format_exc()[-400:]}",
            })

    _io_worker.submit(_run)


def activate_settings_item(app_state):
    """Handle Enter on a settings schema action item."""
    item = SETTINGS_SCHEMA[app_state["settings_index"]]

    if item["key"] == "__back__":
        open_menu(app_state)
    elif item["key"] == "__enroll_fingerprint__":
        player = app_state["config"].get("player_name", "").strip()
        if player:
            store = app_state["fingerprint_store"]
            app_state["_fp_enroll_controller"] = HandEnrollController(
                player_name=player, store=store)
            app_state["app_screen"] = "FP_ENROLL"
        else:
            print("[Fingerprint] No player name set — cannot enroll.")
    elif item["key"] == "__switch_player__":
        app_state["_login_text"] = ""
        app_state["_login_mode"] = "type"
        app_state["app_screen"]  = "LOGIN"
    elif item["key"] == "__hand_diag__":
        store = app_state["fingerprint_store"]
        app_state["_hand_diag_controller"] = HandDiagController(store=store)
        app_state["app_screen"] = "HAND_DIAG"
    elif item["key"] == "__privacy__":
        # Default to Accept highlighted unless they previously declined
        app_state["_consent_selected"] = 1 if has_declined(app_state["config"]) else 0
        app_state["app_screen"] = "CONSENT"


def format_setting_value(app_state, item):
    if item["type"] == "action":
        return ""

    value = app_state["config"][item["key"]]

    if item["type"] == "text":
        display = str(value).strip()
        return display if display else "(not set)"

    if item["type"] == "choice":
        if value == "FairPlay":
            return "Fair Play"
        return str(value)

    return f"{value:.2f}"


def handle_menu_key(app_state, key):
    sp = app_state.get("sound_player")

    # ── Level 3: inside a category's mode list ─────────────────────────────
    if app_state.get("in_game_category"):
        cat   = GAME_CATEGORIES[app_state["game_category_index"]]
        n     = len(cat["modes"])
        if key in KEY_UP:
            app_state["game_mode_index"] = (app_state["game_mode_index"] - 1) % n
            if sp: sp.play("menu_move")
        elif key in KEY_DOWN:
            app_state["game_mode_index"] = (app_state["game_mode_index"] + 1) % n
            if sp: sp.play("menu_move")
        elif key in KEY_ENTER:
            if sp: sp.play("menu_select")
            return activate_menu_item(app_state)
        elif key == KEY_ESC or key in KEY_LEFT:
            app_state["in_game_category"] = False
            if sp: sp.play("menu_move")
        return None

    # ── Level 2: GAME_CATEGORY screen (pick a category) ───────────────────
    if app_state.get("app_screen") == "GAME_CATEGORY":
        n = len(GAME_CATEGORIES)
        if key in KEY_UP:
            app_state["game_category_index"] = (app_state["game_category_index"] - 1) % n
            if sp: sp.play("menu_move")
        elif key in KEY_DOWN:
            app_state["game_category_index"] = (app_state["game_category_index"] + 1) % n
            if sp: sp.play("menu_move")
        elif key in KEY_ENTER or key in KEY_RIGHT:
            # Open the mode list for this category
            app_state["in_game_category"] = True
            app_state["game_mode_index"]  = 0
            if sp: sp.play("menu_select")
        elif key == KEY_ESC:
            app_state["app_screen"] = "MENU"
            if sp: sp.play("menu_move")
        return None

    # ── Level 1: main menu ─────────────────────────────────────────────────
    if key in KEY_UP:
        app_state["menu_index"] = (app_state["menu_index"] - 1) % len(app_state["menu_items"])
        if sp: sp.play("menu_move")
    elif key in KEY_DOWN:
        app_state["menu_index"] = (app_state["menu_index"] + 1) % len(app_state["menu_items"])
        if sp: sp.play("menu_move")
    elif key in KEY_ENTER:
        if sp: sp.play("menu_select")
        return activate_menu_item(app_state)
    return None


def handle_settings_key(app_state, key):
    item = SETTINGS_SCHEMA[app_state["settings_index"]]
    is_text_edit = app_state.get("_settings_text_edit", False)

    if is_text_edit:
        # In text-edit mode: type characters, backspace, Enter to confirm, ESC to cancel
        if key == KEY_ESC:
            app_state["_settings_text_edit"] = False
        elif key in KEY_ENTER:
            val = app_state["config"].get(item["key"], "").strip()
            app_state["config"][item["key"]] = val
            # Mark first run done once any name is confirmed
            if item["key"] == "player_name" and val:
                app_state["config"]["first_run_complete"] = True
                # Ensure a profile exists for the new name
                app_state["profile_store"].get_or_create_profile(val)
                # Refresh player stats so they reflect the new name immediately
                open_player_stats(app_state)
            save_config(app_state["config"])
            app_state["_settings_text_edit"] = False
        elif key == 8 or key == 127:  # backspace
            current = app_state["config"].get(item["key"], "")
            app_state["config"][item["key"]] = current[:-1]
        elif 32 <= key <= 126:  # printable ASCII
            current = app_state["config"].get(item["key"], "")
            if len(current) < 20:
                app_state["config"][item["key"]] = current + chr(key)
        return

    # Normal navigation
    if key in KEY_UP:
        app_state["settings_index"] = (app_state["settings_index"] - 1) % len(SETTINGS_SCHEMA)
    elif key in KEY_DOWN:
        app_state["settings_index"] = (app_state["settings_index"] + 1) % len(SETTINGS_SCHEMA)
    elif key in KEY_LEFT:
        apply_setting_change(app_state, -1)
    elif key in KEY_RIGHT:
        apply_setting_change(app_state, 1)
    elif key in KEY_ENTER:
        if item.get("type") == "text":
            app_state["_settings_text_edit"] = True
        else:
            activate_settings_item(app_state)
    elif key == KEY_ESC:
        open_menu(app_state)


def run():
    app_state = build_app_state()

    # Check for updates silently in background — won't slow startup
    auto_updater.check_in_background()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        raise SystemExit

    app_state["cap"] = cap
    apply_camera_resolution(cap, app_state["config"])

    cv2.namedWindow(WINDOW_NAME)

    with create_hands_detector() as hands, create_nav_detector() as nav_hands:
        _nav_skip_tick = 0   # throttle nav gesture processing on menu screens
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read frame.")
                finalize_active_challenge_run(app_state, status="abandoned")
                break

            # Rolling FPS -- exponential moving average, updated every frame
            _now = time.monotonic()
            _dt  = _now - app_state["_fps_last_t"]
            app_state["_fps_last_t"] = _now
            if _dt > 0:
                app_state["_fps_val"] = 0.9 * app_state["_fps_val"] + 0.1 * (1.0 / _dt)

            # ── Performance: on menu screens skip gesture nav every other frame ──
            # The display and keyboard always run every frame for responsiveness.
            # Only the MediaPipe nav-hand processing is throttled.
            _screen = app_state["app_screen"]
            _throttle_nav = _screen not in ("GAME", "TUTORIAL")
            if _throttle_nav:
                _nav_skip_tick = (_nav_skip_tick + 1) % 2
            else:
                _nav_skip_tick = 0

            if app_state["app_screen"] == "GAME":
                _is_two_player = app_state["play_mode"] in ("TwoPlayerPvP", "PvPvAI",
                                                               "ReflexTwoPlayer",
                                                               "SimonSays2P",
                                                               "SquidGame2P")
                p1_tracker = p2_tracker = None   # set in two-player branch
                show_session_summary = False      # set in single-player path

                if _is_two_player:
                    # ── Two-player path: process both hands simultaneously ──
                    frame, p1_hand, p2_hand, _rgb = process_two_hands_frame(
                        frame=frame,
                        hands=hands,
                        hand_orientation=app_state["config"]["hand_orientation"],
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        ema_states=app_state["_tp_ema_states"],
                        five_gesture_mode=(app_state["play_mode"] == "RPSLS"),
                    )
                    p1_tracker = app_state["_tp_tracker_p1"].update(p1_hand["raw_gesture"])
                    p2_tracker = app_state["_tp_tracker_p2"].update(p2_hand["raw_gesture"])
                    # Stash for diagnostic screen
                    app_state["_tp_last_p1_hand"] = p1_hand
                    app_state["_tp_last_p2_hand"] = p2_hand
                    # Dummy single-player values for shared code that reads them
                    hand_state    = p1_hand
                    tracker_state = p1_tracker
                    _rgb = _rgb  # available for emotion if needed
                    app_state["emotion_state"] = None  # no emotion in two-player

                    controller = get_active_controller(app_state)
                    if app_state["play_mode"] == "ReflexTwoPlayer":
                        game_state = controller.update(
                            p1_tracker=p1_tracker,
                            p2_tracker=p2_tracker,
                            now=time.monotonic(),
                        )
                    elif app_state["play_mode"] == "SimonSays2P":
                        game_state = controller.update(
                            p1_tracker=p1_tracker,
                            p2_tracker=p2_tracker,
                            now=time.monotonic(),
                        )
                        if game_state.get("tracker_reset_requested"):
                            app_state["_tp_tracker_p1"].reset()
                            app_state["_tp_tracker_p2"].reset()
                            controller._tracker_reset_req = False
                    elif app_state["play_mode"] == "SquidGame2P":
                        game_state = controller.update(
                            p1_hand=p1_hand,
                            p2_hand=p2_hand,
                            now=time.monotonic(),
                        )
                    else:
                        game_state = controller.update(
                            p1_tracker_state=p1_tracker,
                            p2_tracker_state=p2_tracker,
                            p1_wrist_y=p1_hand.get("raw_wrist_y") or p1_hand["wrist_y"],
                            p2_wrist_y=p2_hand.get("raw_wrist_y") or p2_hand["wrist_y"],
                            now=time.monotonic(),
                        )
                else:
                    # ── Single-player path (original + reflex + bluff) ──
                    frame, hand_state, _rgb = process_hand_frame(
                        frame=frame,
                        hands=hands,
                        target_hand=app_state["target_hand"],
                        display_mode=app_state["display_mode"],
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        hand_orientation=app_state["config"]["hand_orientation"],
                        _ema_state=app_state["_ema_state"],
                        five_gesture_mode=(app_state["play_mode"] == "RPSLS"),
                    )

                    _mode = app_state["play_mode"]  # needed by emotion gate below

                    # ArcadeSnake uses its own vote-buffer debouncing inside the
                    # controller. Still call tracker.update() so confirmed_gesture
                    # and stable_gesture are populated, but skip emotion entirely.
                    if _mode == "ArcadeSnake":
                        tracker_state = app_state["tracker"].update(hand_state["raw_gesture"])
                    else:
                        if app_state["config"].get("emotion_enabled"):
                            app_state["_emotion_frame_skip"] = (app_state["_emotion_frame_skip"] + 1) % 3
                            if app_state["_emotion_frame_skip"] == 0:
                                emotion_state = app_state["emotion_tracker"].update(_rgb)
                                app_state["emotion_state"] = emotion_state
                        else:
                            app_state["emotion_state"] = None

                        tracker_state = app_state["tracker"].update(hand_state["raw_gesture"])

                    controller = get_active_controller(app_state)

                    if hasattr(controller, "set_emotion_snapshot"):
                        controller.set_emotion_snapshot(
                            app_state["emotion_tracker"].get_round_snapshot()
                        )

                    # Use raw (unsmoothed) wrist Y for all pump beat detection.
                    # The Kalman filter introduces lag that prevents threshold crossing.
                    _pump_y = hand_state.get("raw_wrist_y") or hand_state["wrist_y"]
                    if _mode == "ReflexSolo":
                        game_state = controller.update(
                            tracker_state=tracker_state,
                            now=time.monotonic(),
                            player_name=app_state["config"].get("player_name", ""),
                        )
                    elif _mode == "BluffMode":
                        game_state = controller.update(
                            tracker_state=tracker_state,
                            wrist_y=_pump_y,
                            now=time.monotonic(),
                        )
                    elif _mode == "SimonSaysSolo":
                        game_state = controller.update(
                            tracker_state=tracker_state,
                            now=time.monotonic(),
                            player_name=app_state["config"].get("player_name", ""),
                        )
                    elif _mode == "SquidGame":
                        game_state = controller.update(
                            hand_state=hand_state,
                            now=time.monotonic(),
                        )
                    elif _mode == "PredictionRace":
                        game_state = controller.update(
                            wrist_y=_pump_y,
                            tracker_state=tracker_state,
                            now=time.monotonic(),
                        )
                    elif _mode == "GestureRehab":
                        game_state = controller.update(
                            tracker_state=tracker_state,
                            now=time.monotonic(),
                        )
                    elif _mode == "ArcadeSnake":
                        game_state = controller.update(
                            tracker_state=tracker_state,
                            now=time.monotonic(),
                        )
                    elif _mode == "RPSLS":
                        game_state = controller.update(
                            tracker_state=tracker_state,
                            wrist_y=_pump_y,
                            now=time.monotonic(),
                        )
                    else:
                        game_state = controller.update(
                            wrist_y=_pump_y,
                            tracker_state=tracker_state,
                            now=time.monotonic()
                        )

                # --- Sound effects ---
                _dispatch_sounds(app_state, game_state)

                # Modes that don't use RPS round mechanics — skip all the
                # round-result bookkeeping, profile recording, flash/streak,
                # commentary, voice dispatch, and session summary entirely.
                _is_rps_mode = _mode not in (
                    "ArcadeSnake", "GestureRehab", "SquidGame", "SquidGame2P",
                    "ReflexSolo", "ReflexTwoPlayer", "SimonSaysSolo", "SimonSays2P",
                )

                # --- Commentary engine: fire on round result (RPS modes only) ---
                cur_state  = game_state.get("state", "")
                prev_state = app_state["_snd_last_state"]
                if _is_rps_mode and cur_state == "ROUND_RESULT" and prev_state != "ROUND_RESULT":
                    app_state["commentary_engine"].on_round_result(game_state)

                # --- Flash + streak (RPS modes only) ---
                fi = app_state["_flash_info"]
                if _is_rps_mode:
                    if cur_state == "ROUND_RESULT" and prev_state != "ROUND_RESULT":
                        banner = game_state.get("result_banner", "").upper()
                        if "YOU WIN" in banner or "SURVIVE" in banner:
                            result_type = "win"
                        elif "DRAW" in banner:
                            result_type = "draw"
                        else:
                            result_type = "lose"
                        fi.update({"active": True, "result": result_type, "frame_idx": 0})

                        if result_type == "win":
                            if app_state["_streak_type"] == "win":
                                app_state["_streak_count"] += 1
                            else:
                                app_state["_streak_type"]  = "win"
                                app_state["_streak_count"] = 1
                        elif result_type == "lose":
                            if app_state["_streak_type"] == "lose":
                                app_state["_streak_count"] += 1
                            else:
                                app_state["_streak_type"]  = "lose"
                                app_state["_streak_count"] = 1
                        else:
                            app_state["_streak_count"] = 0
                            app_state["_streak_type"]  = ""

                    if fi["active"]:
                        fi["frame_idx"] += 1
                        if fi["frame_idx"] >= 5:
                            fi["active"] = False

                    fi["mic_level"] = app_state["voice_controller"].get_mic_level() \
                        if app_state.get("voice_mode_active") else 0.0

                    # Replay window
                    if cur_state == "ROUND_RESULT" and prev_state != "ROUND_RESULT":
                        fi["replay_until"] = time.monotonic() + 1.5

                    # Session summary
                    show_session_summary = (
                        cur_state == "MATCH_RESULT"
                        and bool(game_state.get("session_summary"))
                    )

                    # Streak label for HUD
                    streak_n = app_state["_streak_count"]
                    streak_t = app_state["_streak_type"]
                    if streak_n >= 2 and streak_t:
                        label = (f"WIN STREAK  {streak_n}" if streak_t == "win"
                                 else f"LOSE STREAK  {streak_n}")
                        game_state["streak_label"] = label
                    else:
                        game_state["streak_label"] = ""

                    # Voice beat/throw dispatch
                    if app_state["voice_mode_active"]:
                        for event in app_state.pop("_voice_game_events", []):
                            if event["type"] == "beat":
                                if hasattr(controller, "inject_voice_beat"):
                                    controller.inject_voice_beat(event["word"])
                            elif event["type"] == "throw":
                                if hasattr(controller, "inject_voice_throw"):
                                    controller.inject_voice_throw(event["gesture"])

                    # Tracker reset — clears countdown Rock so it doesn't
                    # contaminate the SHOOT throw.
                    if game_state.get("request_tracker_reset"):
                        app_state["tracker"].clear_for_new_throw()
                        app_state["_tp_tracker_p1"].clear_for_new_throw()
                        app_state["_tp_tracker_p2"].clear_for_new_throw()
                        if hasattr(controller, "consume_tracker_reset_request"):
                            controller.consume_tracker_reset_request()

                else:
                    # Non-RPS modes: keep flash inactive, no streak, no summary
                    fi["active"]    = False
                    fi["mic_level"] = 0.0
                    game_state["streak_label"]  = ""
                    show_session_summary        = False

                # --- Record round to player profile (RPS modes only) ---
                # Emotion is captured at the END of the ROUND_RESULT display.
                # Non-RPS modes (Snake, Rehab, Squid, Reflex, Simon) don't
                # produce gesture vs gesture round records — skip entirely.
                player_name = app_state["config"].get("player_name", "").strip()
                if player_name and _is_rps_mode:
                    gs_state = game_state.get("state")
                    # Normalise computer gesture key - different modes use different names
                    _comp_gest = (game_state.get("computer_gesture")
                                  or game_state.get("ai_actual")
                                  or game_state.get("ai_gesture")
                                  or "Unknown")
                    gs_key   = (
                        game_state.get("round_number", 0),
                        game_state.get("player_gesture"),
                        _comp_gest,
                    )

                    # Step 1: When we first enter ROUND_RESULT, store a pending
                    # log entry (gestures + outcome) but do NOT record yet.
                    if (
                        gs_state == "ROUND_RESULT"
                        and game_state.get("player_gesture") not in ("Unknown", "", None)
                        and _comp_gest not in ("Unknown", "", None)
                        and gs_key != app_state.get("_last_recorded_round")
                        and app_state.get("_pending_round_log") is None
                    ):
                        banner      = game_state.get("result_banner", "")
                        last_result = game_state.get("last_round_result", "")
                        banner_up   = banner.upper()
                        if ("YOU WIN" in banner_up or "YOU SURVIVE" in banner_up
                                or last_result == "win"):
                            outcome = "win"
                        elif ("ROBOT" in banner_up or "AI WINS" in banner_up
                              or "GAME OVER" in banner_up or "TAKES THE ROUND" in banner_up
                              or "WINS THE ROUND" in banner_up
                              or last_result == "lose"):
                            outcome = "lose"
                        else:
                            outcome = "draw"

                        # Normalise play_mode_label to match filter keys
                        # (controller labels may have spaces, e.g. "Fair Play")
                        _raw_label = game_state.get("play_mode_label", "")
                        _label_map = {
                            "Fair Play":   "FairPlay",
                            "fair play":   "FairPlay",
                            "Challenge":   "Challenge",
                            "Cheat":       "Cheat",
                            "Clone":       "Clone",
                            "Bluff Mode":  "Cheat",
                        }
                        _mode_label = _label_map.get(_raw_label, _raw_label)

                        app_state["_pending_round_log"] = {
                            "key":            gs_key,
                            "player_gesture": game_state.get("player_gesture", "Unknown"),
                            "robot_gesture":  _comp_gest,
                            "outcome":        outcome,
                            "game_mode":      _mode_label,
                            "round_number":   game_state.get("round_number", 0),
                        }

                    # Step 2: Once we leave ROUND_RESULT (state changed), flush
                    # the pending log with the emotion captured NOW - i.e. after
                    # the player has seen and reacted to the result.
                    pending = app_state.get("_pending_round_log")
                    if pending and gs_state != "ROUND_RESULT":
                        app_state["_last_recorded_round"] = pending["key"]
                        app_state["_pending_round_log"]   = None

                        # Dispatch to background thread - JSON + Excel I/O
                        # would otherwise freeze the frame on round result.
                        _io_worker.submit(
                            app_state["profile_store"].record_round,
                            player_name=player_name,
                            player_gesture=pending["player_gesture"],
                            robot_gesture=pending["robot_gesture"],
                            outcome=pending["outcome"],
                            game_mode=pending["game_mode"],
                            round_number=pending["round_number"],
                            emotion=app_state["emotion_tracker"].get_round_snapshot(),
                        )

                if app_state["display_mode"] == "Diagnostic" and not _is_two_player and _is_rps_mode:
                    # Feed landmarks to collector each frame.
                    app_state["landmark_collector"].update_landmarks(
                        hand_state.get("_landmarks")
                    )

                    collector_status = app_state["landmark_collector"].get_status_text()
                    if collector_status:
                        top_right = collector_status
                    else:
                        top_right = "F Collect | T Train | E Face | 1-3 Mode | ESC Menu"

                    draw_top_bar(
                        frame,
                        f"DIAGNOSTIC | {game_state['play_mode_label'].upper()}",
                        top_right
                    )

                    output_text = app_state.get("collector_message") or app_state["robot_output"].get_latest_summary()

                    draw_info_panel(
                        frame=frame,
                        tracker_state=tracker_state,
                        game_state=game_state,
                        count_text=hand_state["count_text"],
                        status_text=hand_state["status_text"],
                        reason_text=hand_state["reason_text"],
                        ambiguous_count=hand_state["ambiguous_count"],
                        output_summary=output_text,
                        emotion_state=app_state.get("emotion_state"),
                        fps=app_state["_fps_val"],
                    )

                    draw_diagnostic_game_panel(frame, game_state)

                elif _is_two_player:
                    # Two-player renderers
                    cb = app_state["config"].get("colourblind_mode", False)
                    if app_state["display_mode"] == "Diagnostic":
                        draw_two_player_diagnostic(
                            frame, game_state,
                            p1_hand_state=app_state.get("_tp_last_p1_hand"),
                            p2_hand_state=app_state.get("_tp_last_p2_hand"),
                            p1_tracker_state=p1_tracker,
                            p2_tracker_state=p2_tracker,
                            fps=app_state["_fps_val"],
                        )
                    elif app_state["play_mode"] == "TwoPlayerPvP":
                        draw_two_player_view(
                            frame, game_state,
                            p1_tracker_state=p1_tracker,
                            p2_tracker_state=p2_tracker,
                            colourblind=cb,
                        )
                    elif app_state["play_mode"] == "ReflexTwoPlayer":
                        draw_reflex_two_player_view(
                            frame, game_state,
                            p1_tracker_state=p1_tracker,
                            p2_tracker_state=p2_tracker,
                        )
                    elif app_state["play_mode"] == "SimonSays2P":
                        draw_simon_says_two_player_view(
                            frame, game_state,
                            p1_tracker_state=p1_tracker,
                            p2_tracker_state=p2_tracker,
                        )
                    elif app_state["play_mode"] == "SquidGame2P":
                        draw_squid_game_2p_view(
                            frame, game_state,
                            p1_hand=p1_hand,
                            p2_hand=p2_hand,
                        )
                    else:  # PvPvAI
                        draw_pvpvai_view(
                            frame, game_state,
                            p1_tracker_state=p1_tracker,
                            p2_tracker_state=p2_tracker,
                            colourblind=cb,
                        )

                elif app_state["play_mode"] == "ReflexSolo":
                    draw_reflex_solo_view(frame, game_state)

                elif app_state["play_mode"] == "BluffMode":
                    draw_bluff_mode_view(
                        frame, game_state,
                        tracker_state=tracker_state,
                        hand_state=hand_state,
                        flash_info=app_state["_flash_info"],
                    )

                elif app_state["play_mode"] == "SimonSaysSolo":
                    draw_simon_says_solo_view(frame, game_state)

                elif app_state["play_mode"] == "SquidGame":
                    draw_squid_game_view(frame, game_state, hand_state=hand_state)

                elif app_state["play_mode"] == "PredictionRace":
                    draw_prediction_race_view(
                        frame, game_state,
                        tracker_state=tracker_state,
                    )

                elif app_state["play_mode"] == "GestureRehab":
                    draw_gesture_rehab_view(frame, game_state)

                elif app_state["play_mode"] == "ArcadeSnake":
                    draw_arcade_snake_view(frame, game_state,
                                           tracker_state=tracker_state)

                elif app_state["play_mode"] == "RPSLS":
                    draw_rpsls_view(
                        frame, game_state,
                        tracker_state=tracker_state,
                        hand_state=hand_state,
                    )

                else:
                    draw_game_mode_view(
                        frame, game_state,
                        emotion_state=app_state.get("emotion_state"),
                        voice_mode_active=app_state.get("voice_mode_active", False),
                        last_heard_word=app_state["voice_controller"].get_last_word()
                            if app_state.get("voice_mode_active") else "",
                        tracker_state=tracker_state,
                        hand_state=hand_state,
                        flash_info=app_state["_flash_info"],
                        show_help=app_state.get("show_help", False),
                        sound_on=app_state["sound_player"].is_on(),
                        colourblind=app_state["config"].get("colourblind_mode", False),
                        show_session_summary=show_session_summary,
                    )

                # ── Help overlay for non-RPS modes (? key) ─────────────────
                # RPS modes handle show_help inside draw_game_mode_view.
                # All other modes need it drawn as a post-pass overlay here.
                if not _is_rps_mode and app_state.get("show_help", False):
                    voice_on = app_state.get("voice_mode_active", False)
                    draw_help_overlay(
                        frame,
                        "GAME_NONRPS",
                        voice_mode=voice_on,
                    )

                # ── Commentary overlay (draws on top of any game mode) ──
                commentary_line = app_state["commentary_engine"].get_latest()
                if commentary_line:
                    _h = frame.shape[0]
                    _w = frame.shape[1]
                    # Position above the bottom bar — bottom bar is ~8% from bottom
                    strip_y1 = _h - _ix(_h * 0.16)
                    strip_y2 = _h - _ix(_h * 0.09)
                    # Dark semi-transparent strip
                    overlay2 = frame.copy()
                    cv2.rectangle(overlay2, (0, strip_y1), (_w, strip_y2),
                                  (8, 6, 24), -1)
                    cv2.addWeighted(overlay2, 0.78, frame, 0.22, 0, frame)
                    # Coloured top edge
                    cv2.line(frame, (0, strip_y1), (_w, strip_y1),
                             (80, 60, 180), 1)
                    # Text centred in the strip
                    from ui_base import draw_centered_text_in_rect, _ix as _ix2
                    draw_centered_text_in_rect(frame,
                        f"\u25b6  {commentary_line}",
                        (0, strip_y1, _w, strip_y2),
                        base_scale=0.36,
                        color=(200, 220, 255),
                        thickness=1, outline=2)

                # ── Voice status badge (non-RPS game modes) ─────────────────
                # RPS modes have voice UI built into draw_game_mode_view.
                # Non-RPS modes get a compact badge: mic status + mode hint.
                # Positioned below the top bar (9%–18% height, right-aligned)
                # except Squid Game which has a full-width banner at 0–13%,
                # so its badge drops to 14%–23%.
                if (app_state.get("voice_mode_active") and not _is_rps_mode):
                    from ui_base import _ix as _ix
                    _vw = frame.shape[1]
                    _vh = frame.shape[0]
                    _lw = app_state["voice_controller"].get_last_word()
                    _ml = app_state["voice_controller"].get_mic_level()

                    # Badge text: mic indicator + last heard word
                    _badge = (f"MIC  {_lw.upper()}" if _lw else "MIC  ON")

                    # Per-mode hint line — what can actually be said
                    _voice_hints = {
                        "GestureRehab":    "SAY: START  BACK  QUIT",
                        "ReflexSolo":      "SAY: RESTART  BACK  QUIT",
                        "SimonSaysSolo":   "SAY: BACK  QUIT",
                        "SquidGame":       "SAY: BACK  QUIT",
                        "ArcadeSnake":     "GESTURE ONLY  |  BACK = ESC key",
                        # Two-player — voice not useful, show disclaimer
                        "ReflexTwoPlayer": "VOICE N/A  (2P hands needed)",
                        "SimonSays2P":     "VOICE N/A  (2P hands needed)",
                        "SquidGame2P":     "VOICE N/A  (2P hands needed)",
                    }
                    _hint = _voice_hints.get(_mode, "SAY: BACK  QUIT")

                    # Y position: below top bar normally, below squid banner for squid
                    _is_squid = _mode in ("SquidGame", "SquidGame2P")
                    _by1 = _ix(_vh * (0.135 if _is_squid else 0.093))
                    _by2 = _ix(_vh * (0.225 if _is_squid else 0.183))
                    _bx1 = _ix(_vw * 0.62)
                    _bx2 = _vw - _ix(_vw * 0.015)
                    _bh  = _by2 - _by1
                    _bw  = _bx2 - _bx1

                    # Only draw if badge fits — skip at very low resolutions
                    if _bw > 60 and _bh > 20:
                        # Semi-transparent dark panel
                        _ov = frame.copy()
                        cv2.rectangle(_ov, (_bx1, _by1), (_bx2, _by2),
                                      (4, 8, 20), -1)
                        cv2.addWeighted(_ov, 0.80, frame, 0.20, 0, frame)

                        # Border colour: green when mic active, dim when silent
                        _border_col = (50, 170, 90) if _ml > 0.04 else (35, 55, 45)
                        cv2.rectangle(frame, (_bx1, _by1), (_bx2, _by2),
                                      _border_col, 1)

                        # Line 1: badge (MIC ON / MIC <word>)
                        _badge_col = (70, 210, 120) if _ml > 0.04 else (55, 130, 85)
                        from ui_base import draw_centered_text_in_rect as _dctr
                        _dctr(frame, _badge,
                              (_bx1, _by1, _bx2, _by1 + _ix(_bh * 0.52)),
                              base_scale=0.30, color=_badge_col,
                              thickness=1, outline=1)

                        # Line 2: hint
                        _hint_col = ((200, 160, 60) if "N/A" in _hint
                                     else (85, 110, 95))
                        _dctr(frame, _hint,
                              (_bx1, _by1 + _ix(_bh * 0.52), _bx2, _by2),
                              base_scale=0.22, color=_hint_col,
                              thickness=1, outline=1)

                        # Mic level bar at bottom edge of badge
                        if _ml > 0.01:
                            _bar_y = _by2 - 3
                            cv2.rectangle(frame,
                                (_bx1, _bar_y), (_bx2, _by2),
                                (15, 35, 20), -1)
                            _fill_x = _bx1 + int(_bw * min(1.0, _ml))
                            _bar_col = ((50, 200, 90) if _ml < 0.70
                                        else (50, 160, 230))
                            cv2.rectangle(frame,
                                (_bx1, _bar_y), (_fill_x, _by2),
                                _bar_col, -1)

            elif app_state["app_screen"] == "MENU":
                _nav_enabled = app_state["config"].get("gesture_nav_enabled")
                if _nav_enabled and _nav_skip_tick == 0:
                    frame, nav_hand, _ = process_hand_frame(
                        frame=frame, hands=nav_hands,
                        target_hand=app_state["target_hand"], display_mode="Game",
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        hand_orientation=app_state["config"]["hand_orientation"],
                    )
                    _n   = len(app_state["menu_items"])
                    _gap = 0.80 * 0.55 / max(_n, 1)
                    _nav_result = _run_gesture_nav(
                        app_state, nav_hand, time.monotonic(),
                        item_count=_n,
                        set_index_fn=lambda i: app_state.__setitem__("menu_index", i),
                        content_top=0.44,
                        content_bottom=0.44 + (_n - 1) * _gap,
                    )
                    if _nav_result == "quit":
                        finalize_active_challenge_run(app_state, status="abandoned")
                        break
                else:
                    frame = cv2.flip(frame, 1)

                # Main menu - always draw the standard menu (no submenu in new UI)
                draw_menu_screen(
                    frame=frame,
                    menu_items=app_state["menu_items"],
                    selected_index=app_state["menu_index"],
                    config=app_state["config"],
                    show_help=app_state.get("show_help", False),
                    voice_mode_active=app_state.get("voice_mode_active", False),
                    in_submenu=False,
                    update_label=auto_updater.status_label(),
                )
                if _nav_enabled:
                    draw_gesture_nav_overlay(frame, app_state["gesture_nav"].get_cursor_info())

            elif app_state["app_screen"] == "GAME_CATEGORY":
                frame = cv2.flip(frame, 1)
                draw_game_category_screen(
                    frame=frame,
                    categories=GAME_CATEGORIES,
                    category_index=app_state["game_category_index"],
                    mode_index=app_state["game_mode_index"],
                    in_mode_list=app_state.get("in_game_category", False),
                )

            elif app_state["app_screen"] == "SIMULATIONS":
                frame = cv2.flip(frame, 1)
                draw_simulations_hub_screen(
                    frame=frame,
                    selected_index=app_state.get("sim_tab_index", 0),
                    sim_state=app_state.get("sim_state", {}),
                )

            elif app_state["app_screen"] == "RPSLS_SIDE_NOTICE":
                frame, _notice_hand, _ = process_hand_frame(
                    frame=frame, hands=hands,
                    target_hand=app_state["target_hand"],
                    display_mode="Game",
                    handedness_threshold=app_state["config"]["handedness_threshold"],
                    hand_orientation="Side",   # always Side on this screen
                    _ema_state=app_state["_ema_state"],
                    five_gesture_mode=True,    # need Spock/Lizard on this screen
                )
                # ── 1.5-second dwell-to-tick logic ───────────────────────────
                # We track how long a gesture has been held continuously.
                # Only tick it off after DWELL_SECS of stable confirmed output.
                _DWELL_SECS = 1.5
                _ticked  = app_state.setdefault("_rpsls_ticked", set())
                _dwell_g = app_state.setdefault("_rpsls_dwell_gesture", None)
                _dwell_t = app_state.setdefault("_rpsls_dwell_since",   0.0)

                if _notice_hand:
                    # process_hand_frame returns raw hand state (not tracker state).
                    # Use raw_gesture directly — dwell timer provides the stability.
                    _notice_raw  = _notice_hand.get("raw_gesture", "Unknown")
                    _notice_cur  = _notice_raw if _notice_raw in (
                        "Rock", "Paper", "Scissors", "Spock", "Lizard") else "Unknown"

                    _now_t = time.monotonic()
                    if _notice_cur != "Unknown" and _notice_cur not in _ticked:
                        if _notice_cur == _dwell_g:
                            # Same gesture — check if dwell time elapsed
                            if _now_t - _dwell_t >= _DWELL_SECS:
                                _ticked.add(_notice_cur)
                                app_state["_rpsls_dwell_gesture"] = None
                                app_state["_rpsls_dwell_since"]   = 0.0
                        else:
                            # New gesture — start dwell timer
                            app_state["_rpsls_dwell_gesture"] = _notice_cur
                            app_state["_rpsls_dwell_since"]   = _now_t
                    elif _notice_cur == "Unknown":
                        # No valid gesture — reset dwell
                        app_state["_rpsls_dwell_gesture"] = None
                        app_state["_rpsls_dwell_since"]   = 0.0

                    _notice_conf_g = _notice_cur
                    # Compute dwell progress for renderer
                    _dwell_pct = 0.0
                    if (_notice_conf_g != "Unknown" and
                            _notice_conf_g == app_state.get("_rpsls_dwell_gesture") and
                            _notice_conf_g not in _ticked):
                        _elapsed = time.monotonic() - app_state.get("_rpsls_dwell_since", 0)
                        _dwell_pct = min(1.0, _elapsed / _DWELL_SECS)
                else:
                    _notice_conf_g = "Unknown"
                    _dwell_pct     = 0.0
                    app_state["_rpsls_dwell_gesture"] = None
                    app_state["_rpsls_dwell_since"]   = 0.0

                draw_rpsls_side_notice(
                    frame=frame,
                    was_front_on=app_state.get("_rpsls_was_front_on", False),
                    confirmed_gesture=_notice_conf_g,
                    ticked=_ticked,
                    dwell_pct=_dwell_pct,
                )

            elif app_state["app_screen"] == "RPSLS_TUTORIAL":
                frame, _hand_for_tut, _ = process_hand_frame(
                    frame=frame, hands=hands,
                    target_hand=app_state["target_hand"],
                    display_mode="Game",
                    handedness_threshold=app_state["config"]["handedness_threshold"],
                    hand_orientation=app_state["config"]["hand_orientation"],
                    _ema_state=app_state["_ema_state"],
                )
                draw_rpsls_tutorial_screen(
                    frame=frame,
                    step=app_state.get("rpsls_tutorial_step", 0),
                    hand_state=_hand_for_tut,
                )

            elif app_state["app_screen"] == "SIMULATION":
                frame = cv2.flip(frame, 1)
                draw_simulation_screen(frame, app_state.get("sim_state", {}))

            elif app_state["app_screen"] == "SETTINGS":
                _nav_enabled = app_state["config"].get("gesture_nav_enabled") and \
                               not app_state.get("_settings_text_edit", False)
                if _nav_enabled and _nav_skip_tick == 0:
                    frame, nav_hand, _ = process_hand_frame(
                        frame=frame, hands=nav_hands,
                        target_hand=app_state["target_hand"], display_mode="Game",
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        hand_orientation=app_state["config"]["hand_orientation"],
                    )
                    _n = len(SETTINGS_SCHEMA)
                    _adj = {i for i, s in enumerate(SETTINGS_SCHEMA)
                            if s.get("type") in ("choice", "float")}
                    _run_gesture_nav(
                        app_state, nav_hand, time.monotonic(),
                        item_count=_n,
                        set_index_fn=lambda i: app_state.__setitem__("settings_index", i),
                        content_top=0.240,
                        content_bottom=0.240 + (_n - 1) * 0.060,
                        adjust_items=_adj,
                        adjust_fn=lambda d: apply_setting_change(app_state, d),
                    )
                else:
                    frame = cv2.flip(frame, 1)
                draw_settings_screen(
                    frame=frame,
                    settings_schema=SETTINGS_SCHEMA,
                    selected_index=app_state["settings_index"],
                    config=app_state["config"],
                    format_value_fn=lambda item: format_setting_value(app_state, item),
                    cursor_info=app_state["gesture_nav"].get_cursor_info() if _nav_enabled else None,
                    text_edit=app_state.get("_settings_text_edit", False),
                )
                if _nav_enabled:
                    draw_gesture_nav_overlay(frame, app_state["gesture_nav"].get_cursor_info())

            elif app_state["app_screen"] == "FEATURES":
                _nav_enabled = app_state["config"].get("gesture_nav_enabled")
                if _nav_enabled and _nav_skip_tick == 0:
                    frame, nav_hand, _ = process_hand_frame(
                        frame=frame, hands=nav_hands,
                        target_hand=app_state["target_hand"], display_mode="Game",
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        hand_orientation=app_state["config"]["hand_orientation"],
                    )
                    _n = len(FEATURES_SCHEMA)
                    _fadj = {i for i, s in enumerate(FEATURES_SCHEMA)
                             if s.get("type") == "choice"}
                    _run_gesture_nav(
                        app_state, nav_hand, time.monotonic(),
                        item_count=_n,
                        set_index_fn=lambda i: app_state.__setitem__("features_index", i),
                        content_top=0.28,
                        content_bottom=0.28 + (_n - 1) * 0.0504,
                        adjust_items=_fadj,
                        adjust_fn=lambda d: apply_feature_toggle(
                            app_state,
                            FEATURES_SCHEMA[app_state["features_index"]]["key"],
                            direction=d,
                        ),
                    )
                else:
                    frame = cv2.flip(frame, 1)
                draw_features_screen(
                    frame=frame,
                    features_schema=FEATURES_SCHEMA,
                    selected_index=app_state["features_index"],
                    config=app_state["config"],
                    cursor_info=app_state["gesture_nav"].get_cursor_info() if _nav_enabled else None,
                )
                if _nav_enabled:
                    draw_gesture_nav_overlay(frame, app_state["gesture_nav"].get_cursor_info())

            elif app_state["app_screen"] == "PERSONALITY_SELECT":
                frame = cv2.flip(frame, 1)
                cur_name = PERSONALITY_NAMES[app_state.get("personality_index", 0)]
                draw_personality_settings(frame, cur_name, [])

            elif app_state["app_screen"] == "CLONE_SETUP":
                _nav_enabled = app_state["config"].get("gesture_nav_enabled")
                if _nav_enabled:
                    frame, nav_hand, _ = process_hand_frame(
                        frame=frame, hands=nav_hands,
                        target_hand=app_state["target_hand"], display_mode="Game",
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        hand_orientation=app_state["config"]["hand_orientation"],
                    )
                    available = app_state.get("clone_available", [])
                    _n = max(len(available), 1)
                    # Clone select_opponent layout: item_top = y1+(y2-y1)*0.36 = h*0.408
                    # item_gap = (y2-y1)*0.09 = h*0.072
                    _run_gesture_nav(
                        app_state, nav_hand, time.monotonic(),
                        item_count=_n,
                        set_index_fn=lambda i: app_state.__setitem__("clone_opponent_index", i),
                        content_top=0.408,
                        content_bottom=0.408 + (_n - 1) * 0.072,
                    )
                else:
                    frame = cv2.flip(frame, 1)
                draw_clone_setup_screen(frame, {
                    "step":               app_state.get("clone_step", "enter_name"),
                    "text_buffer":        app_state.get("clone_text_buffer", ""),
                    "player_name":        app_state["config"].get("player_name", ""),
                    "available":          app_state.get("clone_available", []),
                    "selected_index":     app_state.get("clone_opponent_index", 0),
                    "all_players":        app_state.get("clone_all_players", []),
                    "message":            app_state.get("clone_message", ""),
                    "profiles_updating":  app_state.get("clone_profiles_updating", False),
                })
                if _nav_enabled:
                    draw_gesture_nav_overlay(frame, app_state["gesture_nav"].get_cursor_info())

            elif app_state["app_screen"] == "PLAYER_STATS":
                _nav_enabled = app_state["config"].get("gesture_nav_enabled")
                if _nav_enabled:
                    frame, nav_hand, _ = process_hand_frame(
                        frame=frame, hands=nav_hands,
                        target_hand=app_state["target_hand"], display_mode="Game",
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        hand_orientation=app_state["config"]["hand_orientation"],
                    )
                    players = app_state.get("stats_players", [])
                    _n = max(len(players), 1)
                    _run_gesture_nav(
                        app_state, nav_hand, time.monotonic(),
                        item_count=_n,
                        set_index_fn=lambda i: app_state.__setitem__("stats_player_index", i),
                        content_top=0.328,
                        content_bottom=0.328 + (_n - 1) * 0.072,
                    )
                else:
                    frame = cv2.flip(frame, 1)
                draw_player_stats_screen(frame, {
                    "step":              app_state.get("stats_step", "select"),
                    "players":           app_state.get("stats_players", []),
                    "selected_index":    app_state.get("stats_player_index", 0),
                    "data":              app_state.get("stats_data"),
                    "traits":            app_state.get("stats_traits", []),
                    "rounds":            app_state.get("stats_rounds", []),
                    "sessions":          app_state.get("stats_sessions", []),
                    "filter":            app_state.get("stats_filter", "All"),
                    "tab":               app_state.get("stats_tab", "overview"),
                    "player_name_hint":  app_state.get("stats_current_player", ""),
                })
                if _nav_enabled:
                    draw_gesture_nav_overlay(frame, app_state["gesture_nav"].get_cursor_info())

            elif app_state["app_screen"] == "TUTORIAL":
                frame, hand_state, _ = process_hand_frame(
                    frame=frame, hands=hands,
                    target_hand=app_state["target_hand"], display_mode="Game",
                    handedness_threshold=app_state["config"]["handedness_threshold"],
                    hand_orientation=app_state["config"]["hand_orientation"],
                )
                tracker_state = app_state["tracker"].update(hand_state["raw_gesture"])
                update_tutorial(app_state, hand_state, tracker_state)

                if app_state["config"].get("gesture_nav_enabled"):
                    steps_t = _tutorial_steps(app_state)
                    _n = len(steps_t)
                    _run_gesture_nav(
                        app_state, hand_state, time.monotonic(),
                        item_count=_n,
                        set_index_fn=lambda i: app_state.__setitem__("tutorial_step", i),
                        content_top=0.44,
                        content_bottom=0.44 + (_n - 1) * (0.80 * 0.55 / max(_n, 1)),
                    )

                steps      = _tutorial_steps(app_state)
                step_idx   = app_state.get("tutorial_step", 0)
                step_data  = steps[step_idx] if step_idx < len(steps) else steps[-1]

                draw_tutorial_screen(frame, {
                    "step_index":          step_idx,
                    "step":                step_data,
                    "total_steps":         len(steps),
                    "detected_gesture":    app_state.get("tutorial_detected", "Unknown"),
                    "hold_count":          app_state.get("tutorial_hold_count", 0),
                    "hold_needed":         step_data.get("hold_frames", 0),
                    "pump_count":          app_state.get("tutorial_pump_count", 0),
                    "shot_gesture":        app_state.get("tutorial_shot_gesture"),
                    "voice_mode":          app_state.get("tutorial_voice_mode", False),
                    "shoot_visible_since": app_state.get("tutorial_shoot_visible_since"),
                })
                if app_state["config"].get("gesture_nav_enabled"):
                    draw_gesture_nav_overlay(frame, app_state["gesture_nav"].get_cursor_info())

            elif app_state["app_screen"] == "LOGIN":
                frame = cv2.flip(frame, 1)
                verified = app_state["fingerprint_store"].list_verified()
                # "Continue as X" if a name is already saved
                saved_name = app_state["config"].get("player_name", "").strip()
                draw_login_screen(
                    frame=frame,
                    login_text=app_state["_login_text"],
                    saved_name=saved_name,
                    verified_players=verified,
                )

            elif app_state["app_screen"] == "FP_ENROLL":
                ctrl = app_state.get("_fp_enroll_controller")
                frame, hand_state, _ = process_hand_frame(
                    frame=frame, hands=hands,
                    target_hand=app_state["target_hand"],
                    display_mode="Game",
                    handedness_threshold=app_state["config"]["handedness_threshold"],
                    hand_orientation=app_state["config"]["hand_orientation"],
                )
                if ctrl:
                    fp_game_state = ctrl.update(hand_state=hand_state)
                    draw_hand_enroll_view(frame, fp_game_state,
                                                 hand_state=hand_state)

            elif app_state["app_screen"] == "FP_LOGIN":
                ctrl = app_state.get("_fp_login_controller")
                frame, hand_state, _ = process_hand_frame(
                    frame=frame, hands=hands,
                    target_hand=app_state["target_hand"],
                    display_mode="Game",
                    handedness_threshold=app_state["config"]["handedness_threshold"],
                    hand_orientation=app_state["config"]["hand_orientation"],
                )
                if ctrl:
                    fp_game_state = ctrl.update(hand_state=hand_state)
                    draw_hand_login_view(frame, fp_game_state,
                                                hand_state=hand_state)

            elif app_state["app_screen"] == "HARDWARE_TEST":
                ctrl = app_state.get("hardware_test")
                if ctrl:
                    ctrl.update()
                    disp = ctrl.get_display_state()
                    draw_hardware_test_view(frame, disp)

            elif app_state["app_screen"] == "NOTES":
                draw_notes_screen(frame,
                    text_buffer  = app_state["_notes_text"],
                    submitted    = app_state["_notes_submitted"],
                    saved_path   = app_state["_notes_saved_path"])

            elif app_state["app_screen"] == "CONSENT":
                draw_consent_screen(frame,
                    selected=app_state.get("_consent_selected", 0))

            # --- Emotion landmark debug overlay (Diagnostic mode only) ---
            if app_state.get("emotion_debug") and app_state.get("display_mode") == "Diagnostic":
                debug_info = app_state["emotion_tracker"].get_debug_overlay(
                    frame.shape[1], frame.shape[0]
                )
                draw_emotion_debug(frame, debug_info)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            # --- Global voice nav dispatch (all screens) ---
            if app_state.get("voice_mode_active"):
                for event in app_state["voice_controller"].drain_events():
                    if event["type"] == "nav":
                        result = handle_voice_nav(app_state, event["action"])
                        if result == "quit":
                            finalize_active_challenge_run(app_state, status="abandoned")
                            cap.release()
                            cv2.destroyAllWindows()
                            app_state["voice_controller"].stop()
                            app_state["emotion_tracker"].close()
                            _close_terminal()
                            return
                    elif event["type"] in ("beat", "throw"):
                        # Beat/throw only apply during GAME or TUTORIAL
                        if app_state["app_screen"] == "GAME":
                            app_state.setdefault("_voice_game_events", []).append(event)
                        elif app_state["app_screen"] == "TUTORIAL" and app_state.get("tutorial_voice_mode"):
                            handle_voice_tutorial_event(app_state, event)

            if key == ord("q"):
                finalize_active_challenge_run(app_state, status="abandoned")
                break

            if app_state["app_screen"] == "MENU":
                result = handle_menu_key(app_state, key)
                if result == "quit":
                    finalize_active_challenge_run(app_state, status="abandoned")
                    break
                # U key — apply update if one is available
                if key in (ord("u"), ord("U")):
                    if auto_updater.get_state()["status"] == "update_available":
                        auto_updater.apply_and_restart(
                            on_error=lambda msg: app_state.update(
                                {"collector_message": f"Update failed: {msg[:60]}"}
                            )
                        )
                # N key — open player notes/feedback screen
                if key in (ord("n"), ord("N")):
                    app_state["_notes_text"]       = ""
                    app_state["_notes_submitted"]  = False
                    app_state["_notes_saved_path"] = ""
                    app_state["app_screen"]        = "NOTES"

            elif app_state["app_screen"] == "GAME_CATEGORY":
                result = handle_menu_key(app_state, key)
                if result == "quit":
                    finalize_active_challenge_run(app_state, status="abandoned")
                    break

            elif app_state["app_screen"] == "SIMULATIONS":
                _sim_tabs = ["Fair Play vs AI", "3-Way PvPvAI", "Clone Tournament"]
                _n_tabs   = len(_sim_tabs)
                if key == KEY_ESC:
                    open_menu(app_state)
                elif key in KEY_UP:
                    app_state["sim_tab_index"] = (app_state["sim_tab_index"] - 1) % _n_tabs
                elif key in KEY_DOWN:
                    app_state["sim_tab_index"] = (app_state["sim_tab_index"] + 1) % _n_tabs
                elif key in KEY_ENTER:
                    if app_state["sim_tab_index"] == 0:
                        _launch_simulation(app_state)
                    elif app_state["sim_tab_index"] == 1:
                        _launch_pvpvai_simulation(app_state)
                    else:
                        _launch_tournament_simulation(app_state)

            elif app_state["app_screen"] == "SIMULATION":
                if key == KEY_ESC:
                    # Only allow back when done or errored - not mid-run
                    status = app_state.get("sim_state", {}).get("status", "idle")
                    if status in ("done", "error", "idle"):
                        app_state["app_screen"]    = "SIMULATIONS"
                        app_state["sim_tab_index"] = 0

            elif app_state["app_screen"] == "SETTINGS":
                handle_settings_key(app_state, key)

            elif app_state["app_screen"] == "FEATURES":
                handle_features_key(app_state, key)

            elif app_state["app_screen"] == "PERSONALITY_SELECT":
                if key == KEY_ESC:
                    app_state["app_screen"] = "FEATURES"
                elif key in KEY_UP:
                    app_state["personality_index"] = (
                        app_state["personality_index"] - 1) % len(PERSONALITY_NAMES)
                elif key in KEY_DOWN:
                    app_state["personality_index"] = (
                        app_state["personality_index"] + 1) % len(PERSONALITY_NAMES)
                elif key in KEY_ENTER:
                    chosen = PERSONALITY_NAMES[app_state["personality_index"]]
                    app_state["config"]["ai_personality"] = chosen
                    for ctrl_key in ("fair_controller", "challenge_controller",
                                     "clone_controller", "bluff_controller"):
                        ctrl = app_state.get(ctrl_key)
                        if ctrl and hasattr(ctrl, "ai") and hasattr(ctrl.ai, "set_personality"):
                            ctrl.ai.set_personality(chosen)
                    app_state["app_screen"] = "FEATURES"
                    print(f"[Personality] Set to: {chosen}")

            elif app_state["app_screen"] == "CLONE_SETUP":
                handle_clone_setup_key(app_state, key)

            elif app_state["app_screen"] == "PLAYER_STATS":
                handle_player_stats_key(app_state, key)

            elif app_state["app_screen"] == "TUTORIAL":
                handle_tutorial_key(app_state, key)

            elif app_state["app_screen"] == "RPSLS_SIDE_NOTICE":
                if key in KEY_ENTER or key == ord(" "):
                    # Allow Enter to skip straight through (gesture check is optional)
                    if app_state.get("_rpsls_diagnostic"):
                        start_game(app_state, "RPSLS",
                                   from_category=app_state.get("_came_from_category", True))
                        app_state["display_mode"] = "Diagnostic"
                    else:
                        start_game(app_state, "RPSLS",
                                   from_category=app_state.get("_came_from_category", True))
                elif key == KEY_ESC or key == ord("q"):
                    app_state["app_screen"]       = "GAME_CATEGORY"
                    app_state["in_game_category"] = True
                    app_state.setdefault("_rpsls_ticked", set()).clear()

            elif app_state["app_screen"] == "RPSLS_TUTORIAL":
                n_steps = 6   # total tutorial steps
                if key == KEY_ESC or key == ord("q"):
                    if app_state.get("_came_from_category"):
                        app_state["app_screen"]       = "GAME_CATEGORY"
                        app_state["in_game_category"] = True
                    else:
                        open_menu(app_state)
                elif key in KEY_RIGHT or key in KEY_DOWN or key in KEY_ENTER:
                    step = app_state.get("rpsls_tutorial_step", 0)
                    if step < n_steps - 1:
                        app_state["rpsls_tutorial_step"] = step + 1
                    else:
                        # Last step - launch the game
                        start_game(app_state, "RPSLS", from_category=True)
                elif key in KEY_LEFT or key in KEY_UP:
                    step = app_state.get("rpsls_tutorial_step", 0)
                    app_state["rpsls_tutorial_step"] = max(0, step - 1)

            elif app_state["app_screen"] == "GAME":
                if key == KEY_ESC:
                    app_state["show_help"] = False
                    if app_state.get("_came_from_category"):
                        if app_state["play_mode"] == "Challenge":
                            finalize_active_challenge_run(app_state, status="abandoned")
                        app_state["app_screen"]       = "GAME_CATEGORY"
                        app_state["in_game_category"] = True
                        reset_all_modes(app_state)
                    else:
                        open_menu(app_state)

                # ReflexSolo: Enter on GAME_OVER restarts the game
                elif (key in KEY_ENTER and
                      app_state["play_mode"] == "ReflexSolo" and
                      game_state.get("state") == "GAME_OVER"):
                    app_state["reflex_solo_controller"].reset()

                # SimonSaysSolo: Enter on INTRO starts playback
                elif (key in KEY_ENTER and
                      app_state["play_mode"] == "SimonSaysSolo" and
                      game_state.get("state") == "INTRO"):
                    app_state["simon_solo_controller"].start_playback()

                # SimonSays2P: Enter on INTRO starts the game
                elif (key in KEY_ENTER and
                      app_state["play_mode"] == "SimonSays2P" and
                      game_state.get("state") == "INTRO"):
                    app_state["simon_2p_controller"].start_playback()

                # PredictionRace: Enter on MATCH_RESULT to play again
                elif (key in KEY_ENTER and
                      app_state["play_mode"] == "PredictionRace" and
                      game_state.get("state") == "MATCH_RESULT"):
                    app_state["prediction_race_controller"].confirm_match_end()

                # GestureRehab: Enter on INTRO to begin session
                elif (key in KEY_ENTER and
                      app_state["play_mode"] == "GestureRehab" and
                      game_state.get("state") == "INTRO"):
                    app_state["gesture_rehab_controller"].start_session()
                elif key == ord("?"):
                    app_state["show_help"] = not app_state.get("show_help", False)
                elif key == ord("m"):
                    toggle_display_mode(app_state)
                elif key == ord("e"):
                    app_state["emotion_debug"] = not app_state["emotion_debug"]
                    print(f"[Emotion] Debug overlay: {'ON' if app_state['emotion_debug'] else 'OFF'}")
                elif key == ord("n"):
                    on = app_state["sound_player"].toggle()
                    print(f"[Sound] {'ON' if on else 'OFF'}")
                elif key == ord("c"):
                    on = app_state["commentary_engine"].toggle()
                    if not on:
                        app_state["commentary_engine"].clear()
                    print(f"[Commentary] {'ON' if on else 'OFF'}")
                elif key == ord("1"):
                    switch_play_mode(app_state, "Cheat")
                elif key == ord("2"):
                    switch_play_mode(app_state, "FairPlay")
                elif key == ord("3"):
                    switch_play_mode(app_state, "Challenge")

                # --- Data collection keys (Diagnostic mode only) ---
                elif app_state["display_mode"] == "Diagnostic":
                    if key == ord("f"):
                        is_on = app_state["landmark_collector"].toggle()
                        app_state["collector_message"] = (
                            "Collection ON - 7=Rock 8=Scissors 9=Paper"
                            if is_on else "Collection OFF"
                        )

                    elif key in (ord("7"), ord("8"), ord("9")):
                        ok, label, msg = app_state["landmark_collector"].try_record(key)
                        if msg:
                            app_state["collector_message"] = msg

                    elif key == ord("t"):
                        app_state["collector_message"] = "Training model..."
                        print("[Main] Training front-on model...")
                        from front_on_trainer import train_and_save
                        accuracy = train_and_save()
                        if accuracy is not None:
                            app_state["collector_message"] = f"Model trained! Accuracy: {accuracy:.0%}"
                        else:
                            app_state["collector_message"] = "Training failed - need more samples"
                        from front_on_classifier import reload_model
                        reload_model()

                    elif key == ord("r") or key == ord("R"):
                        app_state["collector_message"] = "Updating research report..."
                        _io_worker.submit(_run_report_updater_bg)

                    elif key == ord("h") or key == ord("H"):
                        # Launch dedicated Hardware Test screen
                        try:
                            from serial_bridge import SerialBridge
                            from hardware_test_mode import HardwareTestController
                            app_state["hardware_test"] = HardwareTestController(SerialBridge())
                            app_state["app_screen"] = "HARDWARE_TEST"
                        except ImportError:
                            app_state["collector_message"] = "Hardware test requires pyserial - pip install pyserial"

            # ── LOGIN screen ───────────────────────────────────────────────
            elif app_state["app_screen"] == "LOGIN":
                verified = app_state["fingerprint_store"].list_verified()
                if app_state["_login_mode"] == "type":
                    if key == 8 or key == 127:   # backspace
                        app_state["_login_text"] = app_state["_login_text"][:-1]
                    elif key in KEY_ENTER:
                        name = app_state["_login_text"].strip()
                        if name:
                            app_state["config"]["player_name"] = name
                            app_state["config"]["first_run_complete"] = True
                            save_config(app_state["config"])
                            # Load or create profile
                            ps = app_state["profile_store"]
                            ps.get_or_create_profile(name)
                            open_menu(app_state)
                    elif 32 <= key <= 126:  # printable ASCII
                        if len(app_state["_login_text"]) < 20:
                            app_state["_login_text"] += chr(key)
                    elif key == ord("\t") and verified:
                        # Tab switches to fingerprint login mode
                        app_state["_login_mode"] = "fingerprint"
                        store = app_state["fingerprint_store"]
                        app_state["_fp_login_controller"] = HandLoginController(store=store)
                        app_state["app_screen"] = "FP_LOGIN"

                elif app_state["_login_mode"] == "fingerprint":
                    if key == KEY_ESC:
                        app_state["_login_mode"] = "type"

            # ── CONSENT screen ────────────────────────────────────────────────
            elif app_state["app_screen"] == "CONSENT":
                sel = app_state.get("_consent_selected", 0)
                if key in KEY_LEFT:
                    app_state["_consent_selected"] = 0
                elif key in KEY_RIGHT:
                    app_state["_consent_selected"] = 1
                elif key == ord("\t"):
                    # TAB toggles between the two buttons
                    app_state["_consent_selected"] = 1 - sel
                elif key in KEY_ENTER:
                    accepted = (sel == 0)
                    set_consent(app_state["config"], accepted)
                    save_config(app_state["config"])
                    if not app_state["config"].get("player_name", "").strip():
                        app_state["app_screen"] = "LOGIN"
                    else:
                        open_menu(app_state)

            # ── NOTES screen ─────────────────────────────────────────────────
            elif app_state["app_screen"] == "NOTES":
                if app_state["_notes_submitted"]:
                    # Any key returns to menu after submission
                    open_menu(app_state)
                elif key == KEY_ESC:
                    open_menu(app_state)
                elif key in KEY_ENTER:
                    text = app_state["_notes_text"].strip()
                    if text:
                        player = app_state["config"].get("player_name", "unknown")
                        sha    = auto_updater.get_local_sha() or ""
                        path   = save_feedback(player, text, git_sha=sha)
                        app_state["_notes_submitted"]  = True
                        app_state["_notes_saved_path"] = str(path)
                        # Send to Discord if consent given
                        if has_consent(app_state["config"]):
                            webhook = get_webhook_url(app_state["config"])
                            discord_send_feedback(webhook, player, text, sha)
                elif key in (8, 127):  # backspace
                    app_state["_notes_text"] = app_state["_notes_text"][:-1]
                elif 32 <= key <= 126:  # printable ASCII
                    if len(app_state["_notes_text"]) < 500:
                        app_state["_notes_text"] += chr(key)

            # ── HARDWARE_TEST screen ─────────────────────────────────────────
            elif app_state["app_screen"] == "HARDWARE_TEST":
                ctrl = app_state.get("hardware_test")
                if ctrl:
                    result = ctrl.handle_key(key)
                    if result == "exit":
                        app_state["app_screen"] = "GAME"

            # ── FP_ENROLL screen ────────────────────────────────────────────
            elif app_state["app_screen"] == "FP_ENROLL":
                ctrl = app_state.get("_fp_enroll_controller")
                if key == KEY_ESC:
                    app_state["app_screen"] = "SETTINGS"
                elif ctrl and ctrl.fp_phase in ("VERIFIED", "FAILED"):
                    if key in KEY_ENTER:
                        app_state["app_screen"] = "SETTINGS"

            # ── FP_LOGIN screen ─────────────────────────────────────────────
            elif app_state["app_screen"] == "FP_LOGIN":
                ctrl = app_state.get("_fp_login_controller")
                if key == KEY_ESC:
                    app_state["_login_mode"] = "type"
                    app_state["app_screen"]  = "LOGIN"
                elif ctrl and ctrl.login_result:
                    if key in KEY_ENTER:
                        name = ctrl.login_result
                        app_state["config"]["player_name"] = name
                        app_state["config"]["first_run_complete"] = True
                        save_config(app_state["config"])
                        open_menu(app_state)

            # --- ? key toggles help on any screen ---
            if key == ord("?"):
                app_state["show_help"] = not app_state.get("show_help", False)

    _io_worker.flush()
    app_state["voice_controller"].stop()
    app_state["emotion_tracker"].close()
    cap.release()
    cv2.destroyAllWindows()
    _close_terminal()


def _close_terminal():
    """Close the terminal window that launched this app."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(
                ["osascript", "-e",
                 'tell application "Terminal" to close first window'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform == "win32":
            # Close the console window on Windows
            subprocess.Popen(
                ["cmd", "/c", "exit"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass


if __name__ == "__main__":
    import traceback as _tb
    import datetime as _dt

    try:
        run()
    except Exception as _exc:
        # ── Crash reporter ────────────────────────────────────────────────
        _ts        = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _crash_dir = os.path.join(os.path.expanduser("~"), "Desktop", "CapStone")
        os.makedirs(_crash_dir, exist_ok=True)
        _crash_path = os.path.join(_crash_dir, f"crash_{_ts}.txt")

        # Also save to crash_reports/ subfolder (easier to find and review)
        _reports_dir = os.path.join(_crash_dir, "crash_reports")
        os.makedirs(_reports_dir, exist_ok=True)
        _reports_path = os.path.join(_reports_dir, f"crash_{_ts}.txt")

        # Get version info for context
        try:
            _sha = auto_updater.get_local_sha() or "unknown"
            _version = _sha[:7]
        except Exception:
            _version = "unknown"

        import platform as _platform
        _report = (
            f"RPS Robot Crash Report\n"
            f"======================\n"
            f"Time:     {_ts}\n"
            f"Version:  {_version}\n"
            f"Platform: {_platform.system()} {_platform.release()} "
            f"({_platform.machine()})\n"
            f"Python:   {_platform.python_version()}\n"
            f"Error:    {type(_exc).__name__}: {_exc}\n\n"
            f"Traceback:\n"
            f"{_tb.format_exc()}\n"
        )

        try:
            with open(_crash_path, "w") as _f:
                _f.write(_report)
        except Exception:
            pass

        # Save copy to crash_reports/ subfolder
        try:
            with open(_reports_path, "w") as _f:
                _f.write(_report)
        except Exception:
            pass

        # Send to Discord if player gave consent
        try:
            from config_store import load_config as _load_cfg
            from privacy_notice import has_consent as _has_consent
            from privacy_notice import get_webhook_url as _get_webhook
            from discord_reporter import send_crash_report as _send_crash
            _cfg = _load_cfg()
            if _has_consent(_cfg):
                _webhook = _get_webhook(_cfg)
                if _webhook:
                    _send_crash(_webhook, _report)
        except Exception:
            pass

        print("\n" + "=" * 60)
        print("CRASH REPORT")
        print("=" * 60)
        print(_report)
        print(f"Report saved to: {_crash_path}")
        print("=" * 60)
        raise