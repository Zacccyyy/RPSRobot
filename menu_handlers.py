"""
menu_handlers.py
================
All keyboard handlers, menu navigation, clone/stats/tutorial/voice
handlers, simulation launchers, and activate_menu_item logic.
"""
import time
import os
import subprocess
import threading
import queue as _queue
import cv2

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
    draw_rpsls_view,
    draw_game_category_screen,
    draw_simulations_hub_screen,
    draw_rpsls_tutorial_screen,
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
from squid_game_state import SquidGameController
from rpsls_state import RPSLSController
from fair_play_ai import FairPlayAI, PERSONALITIES, PERSONALITY_NAMES

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

from app_state import (
    SETTINGS_SCHEMA, FEATURES_SCHEMA, GAME_CATEGORIES, PERSONALITY_NAMES,
    start_game, open_menu, reset_all_modes, rebuild_controllers,
    _apply_voice_mode, apply_camera_resolution,
    finalize_active_challenge_run, update_challenge_logger_context,
    _dispatch_sounds, build_app_state, build_controllers,
)

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
        if item.get("key") != "__back__":
            apply_feature_toggle(app_state, item["key"], direction=-1)
    elif key in KEY_RIGHT or key in KEY_ENTER:
        item = schema[app_state["features_index"]]
        if item.get("key") == "__back__":
            open_menu(app_state)
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

    # Filtered round list for history dots
    if mode_filter and mode_filter != "All":
        filtered_rounds = [r for r in all_rounds if r.get("game_mode") == mode_filter]
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

    Called every frame for all voice "nav" events regardless of which
    screen is active. Beat/throw events are still handled separately
    inside the GAME block.

    Actions:
        up / down     - scroll lists
        select / yes  - confirm / enter
        back / no     - ESC / cancel
        quit          - quit the app
        left / right  - change setting value
        cheat / fair / challenge / clone / stats / tutorial / settings
                      - direct menu shortcuts (work from MENU screen only)
    """
    screen = app_state["app_screen"]

    # Quit works everywhere
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
            pass   # already at top level
        # Direct shortcuts - jump straight to the relevant mode
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
                # Open personality sub-screen
                app_state["app_screen"] = "PERSONALITY_SELECT"
                cur = app_state["config"].get("ai_personality", "Normal")
                app_state["personality_index"] = PERSONALITY_NAMES.index(cur) \
                    if cur in PERSONALITY_NAMES else 0
            else:
                apply_feature_toggle(app_state, item["key"], direction=1)
        elif action == "back":
            open_menu(app_state)

    elif screen == "PERSONALITY_SELECT":
        if action == "up":
            app_state["personality_index"] = (app_state["personality_index"] - 1) % len(PERSONALITY_NAMES)
        elif action == "down":
            app_state["personality_index"] = (app_state["personality_index"] + 1) % len(PERSONALITY_NAMES)
        elif action in ("select", "right"):
            chosen = PERSONALITY_NAMES[app_state["personality_index"]]
            app_state["config"]["ai_personality"] = chosen
            # Apply to AI controllers immediately
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

    elif screen == "CLONE_SETUP":
        step = app_state.get("clone_step", "enter_name")
        if step == "enter_name":
            if action == "select":
                # Confirm whatever name is in the buffer (must be non-empty)
                buf = app_state.get("clone_text_buffer", "").strip()
                if buf:
                    # Simulate pressing Enter - reuse the keyboard handler logic
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
                        (app_state["clone_opponent_index"] - 1) % len(available)
                    )
            elif action == "down":
                available = app_state.get("clone_available", [])
                if available:
                    app_state["clone_opponent_index"] = (
                        (app_state["clone_opponent_index"] + 1) % len(available)
                    )
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

    elif screen == "PLAYER_STATS":
        step = app_state.get("stats_step", "select")
        if step == "select":
            players = app_state.get("stats_players", [])
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
                players = app_state.get("stats_players", [])
                if len(players) > 1:
                    app_state["stats_step"] = "select"
                else:
                    open_menu(app_state)

    elif screen == "TUTORIAL":
        if action == "back":
            open_menu(app_state)
        elif action == "select":
            step_idx = app_state.get("tutorial_step", 0)
            steps = _tutorial_steps(app_state)
            if step_idx >= len(steps) - 1:
                open_menu(app_state)

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
                        "SimonSaysSolo", "SimonSays2P", "SquidGame", "RPSLS"}:
            start_game(app_state, action, from_category=True)
        elif action == "RPSLSTutorial":
            app_state["app_screen"]          = "RPSLS_TUTORIAL"
            app_state["rpsls_tutorial_step"] = 0
            app_state["_came_from_category"] = True
        elif action == "RPSLSDiagnostic":
            start_game(app_state, "RPSLS", from_category=True)
            app_state["display_mode"] = "Diagnostic"
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


def activate_settings_item(app_state):
    item = SETTINGS_SCHEMA[app_state["settings_index"]]

    if item["key"] == "__back__":
        open_menu(app_state)


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


