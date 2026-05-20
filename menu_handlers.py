"""
menu_handlers.py
================
All keyboard and voice navigation handlers for the RPS Gesture Recogniser.

This module handles every user interaction that isn't part of the game itself:
  - Opening screens (settings, features, clone setup, stats, tutorial)
  - Keyboard handlers for each screen (handle_*_key functions)
  - Voice navigation dispatcher (handle_voice_nav)
  - Gesture-nav helper (_run_gesture_nav) — lets the user navigate menus by moving their hand
  - Tutorial state machine (update_tutorial, handle_voice_tutorial_event)
  - Background simulation launchers (_launch_simulation, _launch_pvpvai_simulation)
  - Menu activation logic (activate_menu_item)

How it fits in:
  Imported by main_slim.py (the entry point). Depends on app_state.py for schemas
  and core helpers that are being migrated there during an in-progress refactor.

NOTE: _io_worker is used in handle_clone_setup_key, _launch_simulation, and
_launch_pvpvai_simulation. It is NOT defined here — it is expected to be injected
by the caller, or provided by app_state once that refactor is complete.
"""

import time
import os
import subprocess
import threading
import queue as _queue
import cv2

# --- Game state controllers ---
from gesture_state import GestureStateTracker
from rps_game_state import RPSGameController
from fair_play_state import FairPlayController
from challenge_mode_state import ChallengeController
from robot_output import RobotOutputBuffer
from challenge_stats_logger import ChallengeStatsLogger
from player_profile_store import PlayerProfileStore
from player_clone_ai import PlayerCloneAI

# --- Computer-vision / tracking helpers ---
from hand_landmarks import (
    create_hands_detector,
    create_nav_detector,
    process_hand_frame,
    process_two_hands_frame,
    create_kalman_wrist_state,
)
from landmark_collector import LandmarkCollector
from emotion_tracker import EmotionTracker

# --- UI drawing functions ---
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
    draw_rpsls_view,
    draw_game_category_screen,
    draw_simulations_hub_screen,
    draw_rpsls_tutorial_screen,
)

# --- Config helpers ---
from config_store import (
    load_config,
    save_config,
    get_resolution_tuple,
    SUPPORTED_RESOLUTIONS,
)

# --- Feature modules ---
from sound_player import SoundPlayer
from voice_control import VoiceController, VOSK_AVAILABLE
from gesture_nav import GestureNavController
from two_player_state import TwoPlayerPvPController, PvPvAIController
from reflex_state import ReflexSoloController, ReflexTwoPlayerController
from bluff_mode_state import BluffModeController
from simon_says_state import SimonSaysSoloController, SimonSaysTwoPlayerController
from squid_game_state import SquidGameController
from rpsls_state import RPSLSController
from fair_play_ai import FairPlayAI, PERSONALITIES, PERSONALITY_NAMES


# ---------------------------------------------------------------------------
# Background report updater
# ---------------------------------------------------------------------------

def _run_report_updater_bg():
    """
    Import and run the research-report updater in the calling thread.

    Always dispatched via _io_worker.submit() so it runs on the background
    I/O thread, not the main loop. Errors are caught and printed so a broken
    report never crashes the app.
    """
    try:
        import sys, os
        # Make sure the project directory is importable even if the working dir differs.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from report_updater import update_report
        result = update_report(verbose=False)
        if result:
            print(f"[Report] Updated -> {result}")
    except Exception as exc:
        print(f"[Report] Updater error: {exc}")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The OpenCV window title — must match the string used in cv2.namedWindow().
WINDOW_NAME = "RPS Gesture Recogniser"

# Key-code sets. cv2.waitKey returns different codes on macOS vs Windows for
# arrow keys, so we bundle each direction into a set. A single `key in KEY_UP`
# check then covers all platforms and WASD at the same time.
KEY_ENTER = {10, 13}
KEY_ESC   = 27
KEY_UP    = {82, ord("w"), ord("W")}
KEY_DOWN  = {84, ord("s"), ord("S")}
KEY_LEFT  = {81, ord("a"), ord("A")}
KEY_RIGHT = {83, ord("d"), ord("D")}


# ---------------------------------------------------------------------------
# INCOMPLETE REFACTOR NOTE
# ---------------------------------------------------------------------------
# The imports below bring in symbols from app_state.py, which does not yet
# exist as a separate file. This block is part of an in-progress refactor
# moving shared state, schemas, and helpers out of main.py into app_state.py.
# Do NOT remove these imports — they will work once the refactor is finished.
from app_state import (
    SETTINGS_SCHEMA, FEATURES_SCHEMA, GAME_CATEGORIES, PERSONALITY_NAMES,
    start_game, open_menu, reset_all_modes, rebuild_controllers,
    _apply_voice_mode, apply_camera_resolution,
    finalize_active_challenge_run, update_challenge_logger_context,
    _dispatch_sounds, build_app_state, build_controllers,
)


# ---------------------------------------------------------------------------
# Settings / Features screen helpers
# ---------------------------------------------------------------------------

def open_settings(app_state):
    """Switch to the Settings screen and reset the cursor to the first item."""
    app_state["app_screen"]          = "SETTINGS"
    app_state["settings_index"]      = 0
    app_state["_settings_text_edit"] = False


def open_features(app_state):
    """Switch to the Features screen and reset the cursor to the first item."""
    app_state["app_screen"]    = "FEATURES"
    app_state["features_index"] = 0


def apply_feature_toggle(app_state, key, direction=0):
    """
    Toggle a boolean feature flag, or cycle a multi-choice feature.

    For boolean features: just flips the value.
    For choice features: steps through the options list by `direction`
        (-1 = previous, +1 = next, 0 treated as +1 / advance forward).

    After changing the value, the config is saved to disk and any immediate
    side effects (e.g. resetting gesture-nav, disabling emotion tracker) are
    applied right away so the running app sees them on the next frame.
    """
    config = app_state["config"]

    # Find the schema entry for this key so we know the type and available options.
    item = next((s for s in FEATURES_SCHEMA if s.get("key") == key), None)

    if item is None:
        # Unknown key — nothing to do.
        return

    if item.get("type") == "choice":
        options = item["options"]
        current = config.get(key, options[0])
        idx     = options.index(current) if current in options else 0
        # direction=0 means "advance forward" for choice items.
        if direction == 0:
            direction = 1
        idx = (idx + direction) % len(options)
        config[key] = options[idx]
    else:
        # Boolean: just flip it.
        config[key] = not config.get(key, False)

    save_config(config)

    # --- Immediate side effects: apply changes that need to happen this frame ---

    if key == "face_debug_enabled":
        # Mirror the flag into app_state so the draw loop reads it directly.
        app_state["emotion_debug"] = config[key]

    elif key == "gesture_nav_enabled":
        # If gesture nav was just turned off, reset the controller so the
        # cursor disappears from the screen.
        if not config[key]:
            app_state["gesture_nav"].reset()

    elif key == "emotion_enabled":
        # If emotion tracking was just turned off, clear the cached state
        # so old results don't linger on screen.
        if not config[key]:
            app_state["emotion_tracker"].reset()
            app_state["emotion_state"] = None

    elif key == "input_mode":
        # Start or stop the voice controller based on the new input mode.
        _apply_voice_mode(app_state)

    elif key == "colourblind_mode":
        print(f"[Features] Colourblind mode {'ON' if config[key] else 'OFF'}")

    print(f"[Features] {key} = {config[key]}")


def handle_features_key(app_state, key):
    """
    Handle a keypress on the Features screen.

    Up/Down move the selection cursor.
    Left/Right cycle choice values (Left does nothing on booleans).
    Enter toggles/advances the selected item, or navigates back if __back__ is selected.
    ESC always returns to the main menu.
    """
    schema = FEATURES_SCHEMA

    if key in KEY_UP:
        # Move the cursor up, wrapping around from the top to the bottom.
        app_state["features_index"] = (app_state["features_index"] - 1) % len(schema)
    elif key in KEY_DOWN:
        # Move the cursor down, wrapping around from the bottom to the top.
        app_state["features_index"] = (app_state["features_index"] + 1) % len(schema)
    elif key in KEY_LEFT:
        # Step the highlighted choice backwards (no effect on boolean items).
        item = schema[app_state["features_index"]]
        if item.get("key") != "__back__":
            apply_feature_toggle(app_state, item["key"], direction=-1)
    elif key in KEY_RIGHT or key in KEY_ENTER:
        item = schema[app_state["features_index"]]
        if item.get("key") == "__back__":
            # The back button returns to the main menu.
            open_menu(app_state)
        else:
            # Step the highlighted choice forwards, or toggle a boolean.
            apply_feature_toggle(app_state, item["key"], direction=1)
    elif key == KEY_ESC:
        open_menu(app_state)


# ---------------------------------------------------------------------------
# Clone Mode helpers
# ---------------------------------------------------------------------------

def open_clone_setup(app_state):
    """
    Switch to the Clone Mode setup screen and reset all its transient state.

    The user first types their own name, then picks an opponent from the list
    of saved profiles that have enough rounds to train the clone AI.
    """
    app_state["app_screen"]           = "CLONE_SETUP"
    app_state["clone_step"]           = "enter_name"
    # Pre-fill the name buffer with whatever was last saved so the user
    # doesn't have to retype their name every time.
    app_state["clone_text_buffer"]    = app_state["config"].get("player_name", "")
    app_state["clone_opponent_index"] = 0
    app_state["clone_available"]      = []
    app_state["clone_message"]        = ""


def _start_clone_game(app_state, opponent_name):
    """
    Load a saved player profile, build a clone AI from it, and start the game.

    If the profile exists but has fewer than 30 rounds of data, we refuse to
    start because the clone AI won't be meaningful with so little information.
    30 rounds is roughly the minimum for the pattern tables to have real signal.
    """
    store  = app_state["profile_store"]
    tables = store.build_pattern_tables(opponent_name)

    # Require at least 30 rounds so the pattern tables have enough data to be useful.
    if tables is None or tables["round_count"] < 30:
        count = tables["round_count"] if tables else 0
        app_state["clone_message"] = f"'{opponent_name}' has {count} rounds. Need 30+."
        return

    # Build the clone AI from the pattern tables and wire it into the controller.
    clone_ai = PlayerCloneAI(tables)
    app_state["clone_controller"].ai              = clone_ai
    app_state["clone_controller"].play_mode_label = f"vs {opponent_name}"
    app_state["clone_controller"].opponent_label  = opponent_name.upper()
    app_state["clone_controller"].win_target      = 3

    # Remember the last-used opponent so we can pre-select them next time.
    app_state["config"]["clone_opponent"] = opponent_name
    save_config(app_state["config"])

    print(f"[Clone] Playing vs '{opponent_name}' ({tables['round_count']} rounds)")
    start_game(app_state, "Clone")


def handle_clone_setup_key(app_state, key):
    """
    Handle keypresses on the Clone Setup screen.

    The screen has three sub-steps:
      'enter_name'      - user types their own player name
      'select_opponent' - user picks an opponent from the list
      'no_profiles'     - shown when no profiles have enough data yet
    """
    step = app_state.get("clone_step", "enter_name")

    if step == "enter_name":
        buf = app_state.get("clone_text_buffer", "")

        if key in KEY_ENTER and buf.strip():
            # Confirm the player name and move to opponent selection.
            app_state["config"]["player_name"] = buf.strip()
            save_config(app_state["config"])
            print(f"[Clone] Player name: '{buf.strip()}'")

            store       = app_state["profile_store"]
            all_players = store.list_players()

            # Kick off Excel report generation on a background thread so it
            # doesn't block the UI (can take 1-3 seconds on slow drives).
            app_state["clone_profiles_updating"] = True
            def _profiles_done():
                store.generate_all_player_reports()
                app_state["clone_profiles_updating"] = False
            _io_worker.submit(_profiles_done)

            # Only show opponents that have at least 30 rounds of data.
            playable = [
                (name, count) for name, count in all_players
                if count >= 30
            ]

            if playable:
                # There are valid opponents — let the user pick one.
                app_state["clone_available"]      = playable
                app_state["clone_opponent_index"] = 0
                app_state["clone_step"]           = "select_opponent"
                app_state["clone_message"]        = ""
            else:
                # No one has enough data yet — show all profiles so the
                # user can see how many rounds each has.
                app_state["clone_step"]        = "no_profiles"
                app_state["clone_all_players"] = all_players
                app_state["clone_message"]     = ""

        elif key == KEY_ESC:
            open_menu(app_state)

        elif key == 8 or key == 127:
            # Backspace: remove the last character from the name buffer.
            app_state["clone_text_buffer"] = buf[:-1]

        elif 32 <= key <= 126:
            # Printable ASCII: append the typed character to the buffer.
            app_state["clone_text_buffer"] = buf + chr(key)

    elif step == "select_opponent":
        available = app_state.get("clone_available", [])

        if key in KEY_UP and available:
            # Scroll up through the opponent list, wrapping at the top.
            app_state["clone_opponent_index"] = (
                (app_state["clone_opponent_index"] - 1) % len(available)
            )
        elif key in KEY_DOWN and available:
            # Scroll down through the opponent list, wrapping at the bottom.
            app_state["clone_opponent_index"] = (
                (app_state["clone_opponent_index"] + 1) % len(available)
            )
        elif key in KEY_ENTER and available:
            # Launch the clone game against the selected opponent.
            name, count = available[app_state["clone_opponent_index"]]
            _start_clone_game(app_state, name)
        elif key == KEY_ESC:
            # Go back to the name-entry step.
            app_state["clone_step"] = "enter_name"

    elif step == "no_profiles":
        # Both ESC and Enter take the user back to the menu so they can play
        # some games and build up profile data first.
        if key == KEY_ESC or key in KEY_ENTER:
            open_menu(app_state)


# ---------------------------------------------------------------------------
# Player Stats helpers
# ---------------------------------------------------------------------------

def open_player_stats(app_state):
    """
    Open the Player Stats viewer.

    If there's only one profile, skip the selection list and jump straight
    to the stats view. If there are no profiles at all, bail out silently.
    """
    store       = app_state["profile_store"]
    all_players = store.list_players()

    if not all_players:
        print("[Stats] No player profiles found.")
        return

    app_state["app_screen"]         = "PLAYER_STATS"
    app_state["stats_players"]      = all_players
    app_state["stats_player_index"] = 0
    # If there's only one player, jump straight to the stats view.
    # Otherwise show the selection list first.
    app_state["stats_step"]         = "select" if len(all_players) > 1 else "view"
    app_state["stats_data"]         = None
    app_state["stats_traits"]       = []

    if len(all_players) == 1:
        _load_stats_for_player(app_state, all_players[0][0])


def _load_stats_for_player(app_state, name, mode_filter=None):
    """
    Build pattern tables and personality traits for a player and store them
    in app_state so the renderer can display them.

    When mode_filter is set (e.g. "FairPlay"), only rounds from that game mode
    are included. This lets the user drill down into how they play each mode.
    """
    store = app_state["profile_store"]

    # Pick the right build method depending on whether a filter is active.
    if mode_filter and mode_filter != "All":
        tables = store.build_pattern_tables_filtered(name, mode_filter)
    else:
        tables = store.build_pattern_tables(name)

    if tables is None:
        # Profile exists but has no round data yet — show empty state.
        app_state["stats_data"]           = None
        app_state["stats_traits"]         = ["No data available"]
        app_state["stats_step"]           = "view"
        app_state["stats_rounds"]         = []
        app_state["stats_sessions"]       = store.get_session_history(name)
        app_state["stats_current_player"] = name
        return

    # Load the raw rounds list from the profile file so we can compute
    # per-round stats independently of the pattern tables.
    profile    = store.load_profile(name)
    all_rounds = profile.get("rounds", []) if profile else []

    # Apply the mode filter to the per-round list as well so the history
    # dots match what the pattern tables were computed from.
    if mode_filter and mode_filter != "All":
        filtered_rounds = [r for r in all_rounds if r.get("game_mode") == mode_filter]
    else:
        filtered_rounds = all_rounds

    # Tally win/loss/draw and add percentage fields for the renderer.
    wins   = sum(1 for r in filtered_rounds if r.get("outcome") == "win")
    losses = sum(1 for r in filtered_rounds if r.get("outcome") == "lose")
    draws  = sum(1 for r in filtered_rounds if r.get("outcome") == "draw")
    total  = max(wins + losses + draws, 1)   # avoid division by zero

    tables["wins"]     = wins
    tables["losses"]   = losses
    tables["draws"]    = draws
    tables["win_pct"]  = wins   / total
    tables["loss_pct"] = losses / total
    tables["draw_pct"] = draws  / total

    # Derive plain-English personality traits from the pattern data
    # (e.g. "tends to throw Rock after a loss").
    traits = store._compute_traits(tables)

    app_state["stats_data"]           = tables
    app_state["stats_traits"]         = traits
    app_state["stats_step"]           = "view"
    app_state["stats_rounds"]         = filtered_rounds
    app_state["stats_sessions"]       = store.get_session_history(name)
    app_state["stats_current_player"] = name


def handle_player_stats_key(app_state, key):
    """
    Handle keypresses on the Player Stats screen.

    The screen has two sub-steps:
      'select' - pick a player from the list (only shown when >1 profile exists)
      'view'   - display the stats for the chosen player

    In the 'view' step:
      T          - toggle between 'overview' and 'history' tabs
      Left/Right - cycle the game-mode filter
      X          - export to CSV
      ESC        - go back (to select step or main menu)
    """
    step     = app_state.get("stats_step", "select")
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
            # Reset filter and tab when entering the view for the first time.
            app_state["stats_filter"] = "All"
            app_state["stats_tab"]    = "overview"
            _load_stats_for_player(app_state, name)
        elif key == KEY_ESC:
            open_menu(app_state)

    elif step == "view":
        data = app_state.get("stats_data")
        # Use the stored player name because it stays populated even when
        # the filtered data is None (e.g. no rounds for that mode).
        name = (
            app_state.get("stats_current_player", "")
            or (data.get("player_name", "") if data else "")
        )

        if key == KEY_ESC:
            # If there are multiple players, go back to the selection list;
            # otherwise return directly to the main menu.
            if len(app_state.get("stats_players", [])) > 1:
                app_state["stats_step"] = "select"
            else:
                open_menu(app_state)

        elif key in (ord("t"), ord("T")):
            # Toggle between the overview and history tabs.
            cur = app_state.get("stats_tab", "overview")
            app_state["stats_tab"] = _TABS[(_TABS.index(cur) + 1) % len(_TABS)]

        elif key in KEY_LEFT:
            # Cycle the game-mode filter backwards.
            if name:
                cur_idx = _FILTERS.index(app_state.get("stats_filter", "All"))
                new_f   = _FILTERS[(cur_idx - 1) % len(_FILTERS)]
                app_state["stats_filter"] = new_f
                _load_stats_for_player(app_state, name, mode_filter=new_f)

        elif key in KEY_RIGHT:
            # Cycle the game-mode filter forwards.
            if name:
                cur_idx = _FILTERS.index(app_state.get("stats_filter", "All"))
                new_f   = _FILTERS[(cur_idx + 1) % len(_FILTERS)]
                app_state["stats_filter"] = new_f
                _load_stats_for_player(app_state, name, mode_filter=new_f)

        elif key in (ord("x"), ord("X")):
            # Export this player's data to a CSV file.
            if data:
                path = app_state["profile_store"].export_csv(name)
                if path:
                    print(f"[Stats] Exported to {path}")
                    app_state["collector_message"] = f"Exported: {path}"


# ---------------------------------------------------------------------------
# Tutorial data
# ---------------------------------------------------------------------------

# Each entry describes one step of the physical (gesture) tutorial.
# The renderer reads these dicts directly.
# 'hold_frames' is how many consecutive frames the gesture must be held before
# the step is considered complete. 0 means we advance via other logic.
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

# Voice-mode equivalent steps — same IDs so the renderer can share its
# status-panel drawing logic between both modes.
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


# ---------------------------------------------------------------------------
# Tutorial state machine
# ---------------------------------------------------------------------------

def open_tutorial(app_state):
    """
    Switch to the Tutorial screen and reset all tutorial tracking state.

    Chooses the voice vs. physical step list based on whether voice mode is
    currently active so the tutorial always matches the input method in use.
    """
    app_state["app_screen"]           = "TUTORIAL"
    app_state["tutorial_step"]        = 0
    app_state["tutorial_hold_count"]  = 0
    app_state["tutorial_complete"]    = False
    app_state["tutorial_detected"]    = "Unknown"

    # Pump / countdown tracking used in step 4 (the fist-pump exercise).
    app_state["tutorial_pump_count"]  = 0
    app_state["tutorial_pump_phase"]  = "ready_for_down"
    app_state["tutorial_pump_top_y"]  = None
    app_state["tutorial_pump_bot_y"]  = None

    # Shoot tracking for step 5 (the actual throw).
    app_state["tutorial_shot_gesture"]      = None
    app_state["tutorial_shoot_visible_since"] = None

    # Lock which step list to use for this whole session based on input mode.
    app_state["tutorial_voice_mode"] = app_state.get("voice_mode_active", False)

    # Reset the gesture tracker so old frames don't count as step completions.
    app_state["tracker"].reset()
    print(f"[Tutorial] Started ({'voice' if app_state['tutorial_voice_mode'] else 'physical'})")


def _tutorial_steps(app_state):
    """Return the correct step list (voice or physical) for the current session."""
    if app_state.get("tutorial_voice_mode"):
        return TUTORIAL_STEPS_VOICE
    return TUTORIAL_STEPS


def update_tutorial(app_state, hand_state, tracker_state):
    """
    Advance the tutorial state machine based on the current hand detection.

    Called every frame while app_screen == 'TUTORIAL'.
    This function only handles the physical (gesture) mode. Voice events are
    routed separately via handle_voice_tutorial_event().

    Steps 1-3 (rock/paper/scissors):
        Count consecutive frames where the target gesture is held.
        Advance once hold_frames is reached.

    Step 4 (pump):
        Track wrist Y position to count up/down pump cycles.
        Advance once 4 pumps are counted.

    Step 5 (shoot):
        Wait 2 seconds so the player has time to read the instruction,
        then advance when any valid gesture is detected.
    """
    if app_state.get("tutorial_voice_mode"):
        # Voice mode is handled by handle_voice_tutorial_event, not here.
        return

    steps    = _tutorial_steps(app_state)
    step_idx = app_state["tutorial_step"]
    if step_idx >= len(steps):
        return

    step      = steps[step_idx]
    confirmed = tracker_state.get("confirmed_gesture", "Unknown")
    stable    = tracker_state.get("stable_gesture",   "Unknown")
    wrist_y   = hand_state.get("wrist_y")

    # Keep the on-screen "detected" label current for the renderer.
    app_state["tutorial_detected"] = confirmed if confirmed != "Unknown" else stable

    # --- Steps 1-3: hold the target gesture for enough frames ---
    if step["id"] in ("rock", "paper", "scissors"):
        if confirmed == step["target_gesture"] or stable == step["target_gesture"]:
            app_state["tutorial_hold_count"] += 1
        else:
            # Decay slowly so brief detection gaps don't restart the count from zero.
            app_state["tutorial_hold_count"] = max(0, app_state["tutorial_hold_count"] - 1)

        if app_state["tutorial_hold_count"] >= step["hold_frames"]:
            _advance_tutorial(app_state)

    # --- Step 4: count pump cycles by tracking wrist Y movement ---
    elif step["id"] == "pump":
        is_rock = confirmed == "Rock" or stable == "Rock"

        if is_rock and wrist_y is not None:
            phase = app_state["tutorial_pump_phase"]
            top_y = app_state["tutorial_pump_top_y"]
            bot_y = app_state["tutorial_pump_bot_y"]

            # First two frames just initialise the reference Y positions
            # so we have a baseline before we start counting pumps.
            if top_y is None:
                app_state["tutorial_pump_top_y"] = wrist_y
                return
            if bot_y is None:
                app_state["tutorial_pump_bot_y"] = wrist_y

            if phase == "ready_for_down":
                # Track the highest point seen so far (lower Y value = higher on screen).
                app_state["tutorial_pump_top_y"] = min(top_y, wrist_y)
                # A downward movement of >=4% of frame height counts as a pump stroke.
                if (wrist_y - app_state["tutorial_pump_top_y"]) >= 0.04:
                    app_state["tutorial_pump_count"] += 1
                    app_state["tutorial_pump_phase"] = "waiting_for_up"
                    app_state["tutorial_pump_bot_y"] = wrist_y

                    if app_state["tutorial_pump_count"] >= 4:
                        # 4 pumps done — advance to the next step.
                        _advance_tutorial(app_state)
                        # Clear the pump-Rock gesture so it doesn't get counted
                        # as the actual throw on step 5.
                        app_state["tracker"].clear_for_new_throw()

            elif phase == "waiting_for_up":
                # Track the lowest point so we know when the hand recovers upward.
                app_state["tutorial_pump_bot_y"] = max(
                    bot_y if bot_y else wrist_y, wrist_y
                )
                # An upward recovery of >=3% resets us to "ready for the next downstroke".
                if (app_state["tutorial_pump_bot_y"] - wrist_y) >= 0.03:
                    app_state["tutorial_pump_phase"] = "ready_for_down"
                    app_state["tutorial_pump_top_y"] = wrist_y

    # --- Step 5: throw any gesture ---
    elif step["id"] == "shoot":
        # Don't accept a throw immediately — give the player 2 seconds to read
        # the instruction before we start listening for a gesture.
        if app_state.get("tutorial_shoot_visible_since") is None:
            app_state["tutorial_shoot_visible_since"] = time.monotonic()
        wait_done = (time.monotonic() - app_state["tutorial_shoot_visible_since"]) >= 2.0

        if wait_done:
            # Paper and Scissors are confirmed in a single frame.
            if confirmed in ("Paper", "Scissors"):
                app_state["tutorial_shot_gesture"] = confirmed
                _advance_tutorial(app_state)
            elif stable in ("Paper", "Scissors"):
                app_state["tutorial_shot_gesture"] = stable
                _advance_tutorial(app_state)
            elif confirmed == "Rock" or stable == "Rock":
                # Rock requires a longer hold (~15 frames / 0.5s at 30fps) to
                # distinguish it from leftover pump frames.
                app_state["tutorial_hold_count"] += 1
                if app_state["tutorial_hold_count"] >= 15:
                    app_state["tutorial_shot_gesture"] = "Rock"
                    _advance_tutorial(app_state)


def handle_voice_tutorial_event(app_state, event):
    """
    Advance the voice-mode tutorial in response to a recognised speech event.

    Called from the main loop whenever a 'beat' or 'throw' event arrives while
    the tutorial screen is active and tutorial_voice_mode is True.

    Steps 1-3 (rock/paper/scissors): advance when the matching throw word is heard.
    Step 4 (countdown):              count "one"/"two"/"three" in order; advance at 3.
    Step 5 (shoot):                  any throw word completes the step.
    Step 6 (done):                   handled by the "select" action in handle_voice_nav.
    """
    steps    = TUTORIAL_STEPS_VOICE
    step_idx = app_state.get("tutorial_step", 0)
    if step_idx >= len(steps):
        return

    step    = steps[step_idx]
    step_id = step["id"]

    if event["type"] == "throw":
        gesture = event["gesture"]
        app_state["tutorial_detected"] = gesture

        if step_id in ("rock", "paper", "scissors"):
            # Check both voice_word and target_gesture to handle either field.
            if gesture == step.get("voice_word") or gesture == step.get("target_gesture"):
                app_state["tutorial_shot_gesture"] = gesture
                _advance_tutorial(app_state)

        elif step_id == "shoot":
            # Any valid throw completes the shoot step.
            app_state["tutorial_shot_gesture"] = gesture
            _advance_tutorial(app_state)

    elif event["type"] == "beat":
        word = event["word"]

        if step_id == "pump" and word in ("one", "two", "three"):
            # Map each countdown word to its expected position in the sequence.
            word_to_num = {"one": 1, "two": 2, "three": 3}
            target_count = word_to_num[word]
            # Only advance if this is the next expected word — saying "three"
            # before "two" doesn't count.
            if target_count == app_state["tutorial_pump_count"] + 1:
                app_state["tutorial_pump_count"] = target_count
                if app_state["tutorial_pump_count"] >= 3:
                    _advance_tutorial(app_state)


def _advance_tutorial(app_state):
    """
    Move to the next tutorial step and reset per-step counters.

    Sets tutorial_complete to True once the last step is reached so the
    renderer can show a "finished" state and the key handler can offer
    the "return to menu" prompt.
    """
    app_state["tutorial_step"]      += 1
    app_state["tutorial_hold_count"] = 0
    steps = _tutorial_steps(app_state)

    # We're "complete" once we reach the last step (the "done" card).
    app_state["tutorial_complete"] = app_state["tutorial_step"] >= len(steps) - 1

    if app_state["tutorial_step"] < len(steps):
        step = steps[app_state["tutorial_step"]]
        print(f"[Tutorial] Step: {step['title']}")


def handle_tutorial_key(app_state, key):
    """
    Handle keypresses on the Tutorial screen.

    ESC always returns to the menu.
    Enter only works on the final 'done' step to dismiss the tutorial.
    """
    if key == KEY_ESC:
        open_menu(app_state)
    elif key in KEY_ENTER:
        step_idx = app_state.get("tutorial_step", 0)
        # Only allow Enter to exit once the user has reached the last step.
        if step_idx >= len(TUTORIAL_STEPS) - 1:
            open_menu(app_state)


# ---------------------------------------------------------------------------
# Voice navigation dispatcher
# ---------------------------------------------------------------------------

def handle_voice_nav(app_state, action):
    """
    Dispatch a voice navigation action to the handler for the current screen.

    Called by the main loop for every 'nav' event emitted by the voice
    controller. Beat/throw events are handled separately inside the GAME block.

    Recognised actions:
        up / down          - scroll list cursor
        select / yes       - confirm / press Enter
        back / no          - cancel / press ESC
        quit               - quit the whole application
        left / right       - change a setting value
        cheat / fair / challenge / clone / stats / tutorial / settings / features
                           - voice shortcuts to jump directly to a screen (MENU only)

    Returns "quit" when the app should exit, None otherwise.
    """
    screen = app_state["app_screen"]

    # "quit" is a global action — works from any screen.
    if action == "quit":
        return "quit"

    if screen == "MENU":
        if action == "up":
            app_state["menu_index"] = (app_state["menu_index"] - 1) % len(app_state["menu_items"])
        elif action == "down":
            app_state["menu_index"] = (app_state["menu_index"] + 1) % len(app_state["menu_items"])
        elif action == "select":
            return activate_menu_item(app_state)
        elif action == "back":
            pass  # already at the top level, nothing to navigate back to

        # Direct voice shortcuts — jump straight to a mode without using the menu cursor.
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
                # Open the AI personality sub-screen.
                app_state["app_screen"] = "PERSONALITY_SELECT"
                cur = app_state["config"].get("ai_personality", "Normal")
                app_state["personality_index"] = (
                    PERSONALITY_NAMES.index(cur) if cur in PERSONALITY_NAMES else 0
                )
            else:
                apply_feature_toggle(app_state, item["key"], direction=1)
        elif action == "back":
            open_menu(app_state)

    elif screen == "PERSONALITY_SELECT":
        if action == "up":
            app_state["personality_index"] = (
                (app_state["personality_index"] - 1) % len(PERSONALITY_NAMES)
            )
        elif action == "down":
            app_state["personality_index"] = (
                (app_state["personality_index"] + 1) % len(PERSONALITY_NAMES)
            )
        elif action in ("select", "right"):
            # Confirm the chosen personality and apply it to all AI controllers.
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

    elif screen == "SETTINGS":
        if action == "up":
            app_state["settings_index"] = (
                (app_state["settings_index"] - 1) % len(SETTINGS_SCHEMA)
            )
        elif action == "down":
            app_state["settings_index"] = (
                (app_state["settings_index"] + 1) % len(SETTINGS_SCHEMA)
            )
        elif action == "left":
            apply_setting_change(app_state, -1)
        elif action == "right":
            apply_setting_change(app_state, 1)
        elif action in ("select", "back"):
            activate_settings_item(app_state)
            if action == "back":
                open_menu(app_state)

    elif screen == "CLONE_SETUP":
        step = app_state.get("clone_step", "enter_name")
        if step == "enter_name":
            if action == "select":
                buf = app_state.get("clone_text_buffer", "").strip()
                if buf:
                    # Reuse the keyboard handler by simulating an Enter keypress.
                    handle_clone_setup_key(app_state, 10)
                else:
                    app_state["clone_message"] = (
                        "Say your name on the keyboard first, then say SELECT"
                    )
            elif action == "back":
                open_menu(app_state)
        elif step == "select_opponent":
            available = app_state.get("clone_available", [])
            if action == "up" and available:
                app_state["clone_opponent_index"] = (
                    (app_state["clone_opponent_index"] - 1) % len(available)
                )
            elif action == "down" and available:
                app_state["clone_opponent_index"] = (
                    (app_state["clone_opponent_index"] + 1) % len(available)
                )
            elif action == "select" and available:
                name, _ = available[app_state["clone_opponent_index"]]
                _start_clone_game(app_state, name)
            elif action == "back":
                app_state["clone_step"] = "enter_name"
        elif step == "no_profiles":
            if action in ("select", "back"):
                open_menu(app_state)

    elif screen == "PLAYER_STATS":
        step    = app_state.get("stats_step", "select")
        players = app_state.get("stats_players", [])
        if step == "select":
            if action == "up" and players:
                app_state["stats_player_index"] = (
                    (app_state["stats_player_index"] - 1) % len(players)
                )
            elif action == "down" and players:
                app_state["stats_player_index"] = (
                    (app_state["stats_player_index"] + 1) % len(players)
                )
            elif action == "select" and players:
                name, _ = players[app_state["stats_player_index"]]
                _load_stats_for_player(app_state, name)
            elif action == "back":
                open_menu(app_state)
        elif step == "view":
            if action == "back":
                if len(players) > 1:
                    app_state["stats_step"] = "select"
                else:
                    open_menu(app_state)

    elif screen == "TUTORIAL":
        if action == "back":
            open_menu(app_state)
        elif action == "select":
            step_idx = app_state.get("tutorial_step", 0)
            steps    = _tutorial_steps(app_state)
            # Only allow "select" to exit once the player has reached the done step.
            if step_idx >= len(steps) - 1:
                open_menu(app_state)

    elif screen == "GAME":
        if action == "back":
            open_menu(app_state)

    return None


# ---------------------------------------------------------------------------
# Gesture navigation helper
# ---------------------------------------------------------------------------

def _run_gesture_nav(app_state, hand_state, now, item_count, set_index_fn,
                     content_top=0.44, content_bottom=0.83,
                     adjust_items=None, adjust_fn=None):
    """
    Run one frame of the GestureNavController and dispatch any events it fires.

    The gesture nav lets the user hover over and select menu items by moving their
    hand, without touching the keyboard. It fires three event types:
      'hover'  - hand moved over a new item; update the highlighted index
      'select' - fist-closed dwell; confirm the current item (like pressing Enter)
      'adjust' - horizontal nudge; call adjust_fn with the direction (+1 or -1)

    Returns "quit" if handle_voice_nav returns "quit", None otherwise.
    """
    sp       = app_state.get("sound_player")
    result   = None
    # Remember the previous index so we only play a sound when the cursor moves.
    prev_idx = app_state["gesture_nav"]._last_item_idx

    # Process all gesture-nav events for this frame.
    for ev in app_state["gesture_nav"].update(
        hand_state, now, item_count,
        content_top=content_top, content_bottom=content_bottom,
        adjust_items=adjust_items,
    ):
        if ev["type"] == "hover":
            # Update the highlighted menu item.
            set_index_fn(ev["item_index"])
            # Play a sound only when the cursor actually changes to a new item.
            if prev_idx != -1 and ev["item_index"] != prev_idx and sp:
                sp.play("menu_move")
        elif ev["type"] == "select":
            # The user dwelled on an item — treat it as pressing Enter.
            if sp:
                sp.play("menu_select")
            result = handle_voice_nav(app_state, "select")
        elif ev["type"] == "adjust" and adjust_fn is not None:
            # The user nudged left or right — change the highlighted value.
            adjust_fn(ev["direction"])

    return result


# ---------------------------------------------------------------------------
# Display / mode helpers
# ---------------------------------------------------------------------------

def toggle_display_mode(app_state):
    """
    Toggle between 'Game' and 'Diagnostic' display modes.

    Diagnostic mode shows raw landmark data, gesture confidence, and the
    data-collection overlay. Game mode is the clean player-facing view.
    """
    app_state["display_mode"] = (
        "Diagnostic" if app_state["display_mode"] == "Game" else "Game"
    )
    # Keep the challenge logger in sync so it records the correct context.
    update_challenge_logger_context(app_state)
    print(f"Display mode: {app_state['display_mode']}")


def switch_play_mode(app_state, new_mode):
    """
    Switch to a different play mode while already in-game.

    Only the four single-player modes support hot-switching via number keys.
    Does nothing if the requested mode is already active.
    """
    if new_mode not in {"Cheat", "FairPlay", "Challenge", "Clone"}:
        return

    # Don't restart the same mode unnecessarily.
    if new_mode == app_state["play_mode"] and app_state["app_screen"] == "GAME":
        return

    start_game(app_state, new_mode)


def get_active_controller(app_state):
    """
    Return the game controller that corresponds to the current play mode.

    Each mode has its own controller object stored in app_state. This function
    just reads play_mode and returns the matching one. Falls back to the cheat
    controller if the mode string isn't recognised (shouldn't happen in practice).
    """
    mode = app_state["play_mode"]
    if mode == "FairPlay":        return app_state["fair_controller"]
    if mode == "Challenge":       return app_state["challenge_controller"]
    if mode == "Clone":           return app_state["clone_controller"]
    if mode == "TwoPlayerPvP":    return app_state["pvp_controller"]
    if mode == "PvPvAI":          return app_state["pvpvai_controller"]
    if mode == "ReflexSolo":      return app_state["reflex_solo_controller"]
    if mode == "ReflexTwoPlayer": return app_state["reflex_2p_controller"]
    if mode == "BluffMode":       return app_state["bluff_controller"]
    if mode == "SimonSaysSolo":   return app_state["simon_solo_controller"]
    if mode == "SimonSays2P":     return app_state["simon_2p_controller"]
    if mode == "SquidGame":       return app_state["squid_controller"]
    if mode == "RPSLS":           return app_state["rpsls_controller"]
    return app_state["cheat_controller"]


# ---------------------------------------------------------------------------
# Settings screen helpers
# ---------------------------------------------------------------------------

def apply_setting_change(app_state, direction):
    """
    Increment or decrement the value of the currently selected setting.

    direction: +1 = next/higher, -1 = previous/lower.
    Does nothing for 'action' or 'text' type items (those have no numeric value).
    After saving, rebuilds the AI controllers and reapplies voice mode so any
    config-dependent behaviour updates immediately.
    """
    item = SETTINGS_SCHEMA[app_state["settings_index"]]

    # Action and text items are not adjustable with left/right keys.
    if item["type"] in ("action", "text"):
        return

    key    = item["key"]
    config = app_state["config"]

    if item["type"] == "choice":
        # Step through the options list in the given direction, wrapping around.
        options       = item["options"]
        current_index = options.index(config[key])
        config[key]   = options[(current_index + direction) % len(options)]

    elif item["type"] == "float":
        # Clamp to [min, max] and round to 2 decimal places to avoid floating-point drift.
        value       = config[key] + item["step"] * direction
        config[key] = round(max(item["min"], min(item["max"], value)), 2)

    app_state["config"]       = save_config(config)
    app_state["display_mode"] = app_state["config"]["default_display_mode"]
    update_challenge_logger_context(app_state)

    # If the camera resolution changed, push the new resolution to OpenCV now.
    if key == "camera_resolution" and app_state.get("cap") is not None:
        apply_camera_resolution(app_state["cap"], app_state["config"])

    # Rebuild controllers so they pick up the new config values.
    rebuild_controllers(app_state)
    _apply_voice_mode(app_state)


def activate_settings_item(app_state):
    """
    Handle Enter/confirm on the Settings screen.

    Currently the only interactive item is '__back__', which returns to the
    main menu. All other items are adjusted via left/right, not Enter.
    """
    item = SETTINGS_SCHEMA[app_state["settings_index"]]
    if item["key"] == "__back__":
        open_menu(app_state)


def format_setting_value(app_state, item):
    """
    Format the current value of a settings item for display on screen.

    Returns an empty string for action-type items (they show no value),
    a placeholder for unset text fields, a cleaned-up string for choice
    items (e.g. "FairPlay" -> "Fair Play"), and a 2-decimal float otherwise.
    """
    if item["type"] == "action":
        return ""

    value = app_state["config"][item["key"]]

    if item["type"] == "text":
        display = str(value).strip()
        return display if display else "(not set)"

    if item["type"] == "choice":
        # Special-case: show "Fair Play" with a space for readability.
        if value == "FairPlay":
            return "Fair Play"
        return str(value)

    # Float: always show two decimal places.
    return f"{value:.2f}"


# ---------------------------------------------------------------------------
# Menu key handlers
# ---------------------------------------------------------------------------

def handle_menu_key(app_state, key):
    """
    Handle a keypress when the MENU or GAME_CATEGORY screen is active.

    The menu has three levels:
      Level 1: main menu (root list of actions)
      Level 2: GAME_CATEGORY screen (list of game-mode categories)
      Level 3: mode list inside a selected category

    Returns "quit" if the user chose Quit, None otherwise.
    """
    sp = app_state.get("sound_player")

    # --- Level 3: inside a category's mode list ---
    if app_state.get("in_game_category"):
        cat = GAME_CATEGORIES[app_state["game_category_index"]]
        n   = len(cat["modes"])
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
            # ESC or Left exits the mode list back to the category list.
            app_state["in_game_category"] = False
            if sp: sp.play("menu_move")
        return None

    # --- Level 2: GAME_CATEGORY screen (pick a category) ---
    if app_state.get("app_screen") == "GAME_CATEGORY":
        n = len(GAME_CATEGORIES)
        if key in KEY_UP:
            app_state["game_category_index"] = (app_state["game_category_index"] - 1) % n
            if sp: sp.play("menu_move")
        elif key in KEY_DOWN:
            app_state["game_category_index"] = (app_state["game_category_index"] + 1) % n
            if sp: sp.play("menu_move")
        elif key in KEY_ENTER or key in KEY_RIGHT:
            # Enter or Right opens the mode list for the highlighted category.
            app_state["in_game_category"] = True
            app_state["game_mode_index"]  = 0
            if sp: sp.play("menu_select")
        elif key == KEY_ESC:
            app_state["app_screen"] = "MENU"
            if sp: sp.play("menu_move")
        return None

    # --- Level 1: main menu ---
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


def activate_menu_item(app_state):
    """
    Execute the action for the currently highlighted menu item.

    Handles all three navigation levels:
      Level 3 (inside a category's mode list): launch a game mode or tutorial.
      Level 2 (in_submenu, legacy path):       just clear the flag.
      Level 1 (main menu):                     open a screen or start a game.

    Returns "quit" if the user selected Quit, None otherwise.
    """
    # --- Level 3: launch a mode from inside a category ---
    if app_state.get("in_game_category"):
        cat          = GAME_CATEGORIES[app_state["game_category_index"]]
        label, action = cat["modes"][app_state["game_mode_index"]]
        # Exit the category view before launching so we don't get stuck in it.
        app_state["in_game_category"] = False
        app_state["app_screen"]       = "MENU"

        if action == "Clone":
            open_clone_setup(app_state)
        elif action in {"Cheat", "FairPlay", "Challenge", "TwoPlayerPvP", "PvPvAI",
                        "ReflexSolo", "ReflexTwoPlayer", "BluffMode",
                        "SimonSaysSolo", "SimonSays2P", "SquidGame", "RPSLS"}:
            start_game(app_state, action, from_category=True)
        elif action == "RPSLSTutorial":
            # Show the RPSLS rules slides before launching the game.
            app_state["app_screen"]          = "RPSLS_TUTORIAL"
            app_state["rpsls_tutorial_step"] = 0
            app_state["_came_from_category"] = True
        elif action == "RPSLSDiagnostic":
            # Start RPSLS but force Diagnostic display so landmarks are visible.
            start_game(app_state, "RPSLS", from_category=True)
            app_state["display_mode"] = "Diagnostic"
        return None

    # --- Level 2: legacy in_submenu path (not used in the new UI) ---
    if app_state.get("in_submenu"):
        # Kept for voice-nav compatibility; the current UI doesn't reach this path.
        app_state["in_submenu"] = False
        return None

    # --- Level 1: main menu ---
    label, action = app_state["menu_items"][app_state["menu_index"]]

    if action == "GameModes":
        # Open the category picker screen.
        app_state["app_screen"]          = "GAME_CATEGORY"
        app_state["game_category_index"] = 0
        app_state["game_mode_index"]     = 0
        app_state["in_game_category"]    = False
    elif action == "Simulations":
        app_state["app_screen"]    = "SIMULATIONS"
        app_state["sim_tab_index"] = 0
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


# ---------------------------------------------------------------------------
# Simulation launchers
# ---------------------------------------------------------------------------

def _launch_simulation(app_state):
    """
    Run a high-fidelity Fair-Play vs AI simulation in a background thread.

    Runs approximately 100,000 rounds:
        6 player strategies x 3 AI opponents x 55 runs x 100 rounds = 99,000

    Progress is pushed into app_state["sim_state"] every run so the
    draw_simulation_screen renderer can show a live progress bar.

    After completion, the research report is auto-updated in the background.
    """
    # Switch to the simulation progress screen immediately so the user
    # can see that something is happening.
    app_state["app_screen"] = "SIMULATION"
    app_state["sim_state"]  = {
        "status":        "running",
        "progress":      0.0,
        "progress_text": "Initialising...",
        "results":       None,
        "error":         None,
    }

    def _run():
        """Inner function that runs on the background I/O thread."""
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import simulation_mode as _sm
            from simulation_mode import PLAYER_STRATEGIES, AI_OPPONENTS

            # ~100k rounds total: 6 strategies x 3 opponents x 55 runs x 100 rounds
            RUNS   = 55
            ROUNDS = 100
            total  = len(PLAYER_STRATEGIES) * len(AI_OPPONENTS) * RUNS
            done   = [0]   # list so the nested closure can mutate it

            # Monkey-patch run_single_game to intercept calls and update
            # the progress bar after each run completes.
            _orig = _sm.run_single_game
            def _patched(strategy, ai_type, rounds):
                result  = _orig(strategy, ai_type, rounds)
                done[0] += 1
                pct = done[0] / max(total, 1)
                app_state["sim_state"].update({
                    "progress":      pct,
                    "progress_text": (
                        f"{strategy}  vs  {ai_type}  ({done[0]:,}/{total:,} runs)"
                    ),
                })
                return result
            _sm.run_single_game = _patched

            results = _sm.run_simulation(
                runs_per_combo=RUNS,
                rounds_per_run=ROUNDS,
                save_excel=True,
            )

            # Restore the original function before computing summaries.
            _sm.run_single_game = _orig

            combos       = results.get("combo_results", [])
            total_rounds = results.get("total_rounds", 0)

            if combos:
                # Find the most balanced matchup (closest to a 50/50 win rate).
                most_balanced = min(combos, key=lambda c: abs(c["player_win_rate"] - 0.5))
                results["most_balanced"] = (
                    f"{most_balanced['strategy']} vs {most_balanced['ai']} "
                    f"({most_balanced['player_win_rate']:.1%} player)"
                )

                # Average AI win rate across all player strategies.
                ai_avg = {}
                for ai in AI_OPPONENTS:
                    rows = [c for c in combos if c["ai"] == ai]
                    if rows:
                        ai_avg[ai] = sum(c["robot_win_rate"] for c in rows) / len(rows)
                results["ai_win_rates"] = ai_avg

                # Average player win rate across all AI opponents.
                strat_avg = {}
                for s in PLAYER_STRATEGIES:
                    rows = [c for c in combos if c["strategy"] == s]
                    if rows:
                        strat_avg[s] = sum(c["player_win_rate"] for c in rows) / len(rows)
                results["strategy_win_rates"]  = strat_avg
                results["total_rounds_actual"] = total_rounds

            app_state["sim_state"].update({
                "status":   "done",
                "progress": 1.0,
                "results":  results,
            })

            # Refresh the research report with the new simulation data.
            _run_report_updater_bg()

        except Exception as exc:
            import traceback
            app_state["sim_state"].update({
                "status": "error",
                "error":  f"{exc}\n{traceback.format_exc()[-200:]}",
            })

    _io_worker.submit(_run)


def _launch_pvpvai_simulation(app_state):
    """
    Simulate the 1v1v1 PvPvAI format across all strategy pairings.

    Tests every combination of P1 strategy x P2 strategy x AI type.
    Scoring: beating one opponent = +1 pt, beating two = +2 pts.
    A match ends when any player reaches WIN_TARGET (5) points.

    Progress is pushed into app_state["sim_state"] each run.
    """
    app_state["app_screen"] = "SIMULATION"
    app_state["sim_state"]  = {
        "status":        "running",
        "progress":      0.0,
        "progress_text": "Initialising 3-way simulation...",
        "results":       None,
        "error":         None,
    }

    def _run():
        """Inner function that runs on the background I/O thread."""
        try:
            import sys, random
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from simulation_mode import (
                SimulatedPlayer, create_ai_opponent, PLAYER_STRATEGIES, BEATS
            )
            from two_player_state import PvPvAIController

            RUNS       = 30    # runs per strategy combo — fast enough for ~10s total
            ROUNDS     = 100   # max rounds per run before forcing a winner
            WIN_TARGET = 5     # first player to 5 points wins the match

            # Build every (P1 strategy, P2 strategy, AI type) combination.
            combos = [
                (s1, s2, ai)
                for s1 in PLAYER_STRATEGIES
                for s2 in PLAYER_STRATEGIES
                for ai in ("random", "fair_play", "challenge")
            ]
            total = len(combos) * RUNS
            done  = [0]

            # Store results per combo key for the summary step.
            results = {}

            def _cmp(a, b):
                """Return 'win', 'lose', or 'draw' for gesture a vs gesture b."""
                if a == b:        return "draw"
                if BEATS[a] == b: return "win"
                return "lose"

            def _score_three(g1, g2, g_ai):
                """
                Score a 3-way round. Each player earns +1 for each opponent they beat.
                Returns (p1_points, p2_points, ai_points) for this single round.
                """
                p1 = (1 if _cmp(g1, g2)   == "win" else 0) + (1 if _cmp(g1, g_ai) == "win" else 0)
                p2 = (1 if _cmp(g2, g1)   == "win" else 0) + (1 if _cmp(g2, g_ai) == "win" else 0)
                pa = (1 if _cmp(g_ai, g1) == "win" else 0) + (1 if _cmp(g_ai, g2) == "win" else 0)
                return p1, p2, pa

            # Run every combination.
            for s1, s2, ai_type in combos:
                key = (s1, s2, ai_type)
                p1_match_wins = p2_match_wins = ai_match_wins = 0

                for _ in range(RUNS):
                    player1      = SimulatedPlayer(strategy=s1)
                    player2      = SimulatedPlayer(strategy=s2)
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
                        # The AI uses P1's last gesture as a proxy for its history.
                        ai_hist = (
                            [{"round_number": i, "player_gesture": p1_last,
                              "player_outcome": p1_outcome}
                             for i in range(1)]
                            if p1_last else []
                        )
                        g_ai = ai_fn(ai_hist, 0, rn)

                        r1, r2, ra = _score_three(g1, g2, g_ai)
                        p1_pts += r1
                        p2_pts += r2
                        ai_pts += ra

                        p1_outcome = "win" if r1 > 0 else "lose"
                        p2_outcome = "win" if r2 > 0 else "lose"
                        p1_last = g1
                        p2_last = g2
                        ai_last = g_ai

                        # End the match as soon as any player reaches WIN_TARGET.
                        if (p1_pts >= WIN_TARGET or p2_pts >= WIN_TARGET
                                or ai_pts >= WIN_TARGET):
                            break

                    # Credit the match winner (whoever crossed the target first).
                    if   p1_pts >= WIN_TARGET: p1_match_wins += 1
                    elif p2_pts >= WIN_TARGET: p2_match_wins += 1
                    else:                      ai_match_wins += 1

                    done[0] += 1
                    app_state["sim_state"].update({
                        "progress":      done[0] / max(total, 1),
                        "progress_text": (
                            f"3-way: {s1} vs {s2} vs {ai_type}  ({done[0]:,}/{total:,})"
                        ),
                    })

                results[key] = {
                    "p1_strategy": s1,
                    "p2_strategy": s2,
                    "ai_type":     ai_type,
                    "p1_win_rate": p1_match_wins / RUNS,
                    "p2_win_rate": p2_match_wins / RUNS,
                    "ai_win_rate": ai_match_wins / RUNS,
                    "runs":        RUNS,
                }

            # --- Summary statistics ---
            all_res = list(results.values())

            # Average AI win rate across all strategy pairings.
            ai_avg = {}
            for ai in ("random", "fair_play", "challenge"):
                rows      = [r for r in all_res if r["ai_type"] == ai]
                ai_avg[ai] = sum(r["ai_win_rate"] for r in rows) / max(len(rows), 1)

            # Average player win rate, pooling P1 and P2 appearances together.
            strat_avg = {}
            for s in PLAYER_STRATEGIES:
                rows = [r for r in all_res if r["p1_strategy"] == s or r["p2_strategy"] == s]
                strat_avg[s] = sum(
                    r["p1_win_rate"] if r["p1_strategy"] == s else r["p2_win_rate"]
                    for r in rows
                ) / max(len(rows), 1)

            best_ai    = max(ai_avg,    key=ai_avg.get)
            best_strat = max(strat_avg, key=strat_avg.get)

            # Most balanced combo: minimise total deviation from a perfect 33/33/33 split.
            most_balanced_key = min(results, key=lambda k: (
                abs(results[k]["p1_win_rate"] - 0.333) +
                abs(results[k]["p2_win_rate"] - 0.333) +
                abs(results[k]["ai_win_rate"] - 0.333)
            ))
            mb = results[most_balanced_key]

            app_state["sim_state"].update({
                "status":   "done",
                "progress": 1.0,
                "results":  {
                    "mode":          "pvpvai",
                    "best_ai":       best_ai,
                    "best_strategy": best_strat,
                    "ai_win_rates":  ai_avg,
                    "strategy_win_rates": strat_avg,
                    "combo_results": [
                        {
                            "strategy":        r["p1_strategy"],
                            "ai":              r["ai_type"],
                            # Average P1 and P2 win rates as the combined "player" rate.
                            "player_win_rate": (r["p1_win_rate"] + r["p2_win_rate"]) / 2,
                            "robot_win_rate":  r["ai_win_rate"],
                            "draw_rate":       0.0,
                            "runs":            r["runs"],
                        }
                        for r in all_res
                    ],
                    "most_balanced": (
                        f"{mb['p1_strategy']} vs {mb['p2_strategy']} vs {mb['ai_type']} "
                        f"(P1:{mb['p1_win_rate']:.0%} P2:{mb['p2_win_rate']:.0%} "
                        f"AI:{mb['ai_win_rate']:.0%})"
                    ),
                    "elapsed_seconds":     0,
                    "total_rounds_actual": total * ROUNDS,
                },
            })

        except Exception as exc:
            import traceback
            app_state["sim_state"].update({
                "status": "error",
                "error":  f"{exc}\n{traceback.format_exc()[-300:]}",
            })

    _io_worker.submit(_run)


# ---------------------------------------------------------------------------
# Settings screen key handler (continued)
# ---------------------------------------------------------------------------

def handle_settings_key(app_state, key):
    """
    Handle a keypress on the Settings screen.

    When a text field is being edited (e.g. player name), all keys route to
    the text-edit branch: printable characters append, backspace deletes,
    Enter confirms, ESC cancels without saving.

    Otherwise normal navigation applies:
      Up/Down    - move the selection cursor
      Left/Right - change the value of the highlighted item
      Enter      - activate action items (currently just __back__)
      ESC        - return to the main menu
    """
    item         = SETTINGS_SCHEMA[app_state["settings_index"]]
    is_text_edit = app_state.get("_settings_text_edit", False)

    if is_text_edit:
        if key == KEY_ESC:
            # Cancel text editing without saving any changes.
            app_state["_settings_text_edit"] = False
        elif key in KEY_ENTER:
            val = app_state["config"].get(item["key"], "").strip()
            app_state["config"][item["key"]] = val
            # Mark first-run complete once a player name has been entered.
            if item["key"] == "player_name" and val:
                app_state["config"]["first_run_complete"] = True
            save_config(app_state["config"])
            app_state["_settings_text_edit"] = False
        elif key in (8, 127):
            # Backspace: remove the last character from the field.
            current = app_state["config"].get(item["key"], "")
            app_state["config"][item["key"]] = current[:-1]
        elif 32 <= key <= 126:
            # Printable ASCII: append up to a 20-character limit.
            current = app_state["config"].get(item["key"], "")
            if len(current) < 20:
                app_state["config"][item["key"]] = current + chr(key)
        # Don't fall through to normal navigation while editing text.
        return

    # Normal navigation (not in text-edit mode).
    if key in KEY_UP:
        app_state["settings_index"] = (
            (app_state["settings_index"] - 1) % len(SETTINGS_SCHEMA)
        )
    elif key in KEY_DOWN:
        app_state["settings_index"] = (
            (app_state["settings_index"] + 1) % len(SETTINGS_SCHEMA)
        )
    elif key in KEY_LEFT:
        apply_setting_change(app_state, -1)
    elif key in KEY_RIGHT:
        apply_setting_change(app_state, 1)
    elif key in KEY_ENTER:
        if item.get("type") == "text":
            # Enter on a text item starts editing that field.
            app_state["_settings_text_edit"] = True
        else:
            activate_settings_item(app_state)
    elif key == KEY_ESC:
        open_menu(app_state)
