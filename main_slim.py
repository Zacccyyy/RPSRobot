"""
main_slim.py
============
Entry point for the RPS Gesture Recogniser robot-game.

Opens the webcam, then loops every frame to:
  - track the player's hand gesture via MediaPipe
  - update the active game-mode controller
  - draw the correct screen on the OpenCV window
  - handle keyboard and voice input

All shared state lives in a single dict (app_state) built by build_app_state().
The bulk of app logic is split across:
  app_state.py      -- state construction, schemas, and core helpers
  menu_handlers.py  -- keyboard/voice/menu/nav/tutorial logic
  ui_renderer.py    -- every draw_* function
"""

import time
import os
import subprocess
import cv2

# --- Game state / controller imports ---
from gesture_state import GestureStateTracker
from rps_game_state import RPSGameController
from fair_play_state import FairPlayController
from challenge_mode_state import ChallengeController
from robot_output import RobotOutputBuffer
from challenge_stats_logger import ChallengeStatsLogger
from player_profile_store import PlayerProfileStore
from player_clone_ai import PlayerCloneAI

# --- Computer-vision helpers ---
from hand_landmarks import (
    create_hands_detector,
    create_nav_detector,
    process_hand_frame,
    process_two_hands_frame,
    create_kalman_wrist_state,
)
from landmark_collector import LandmarkCollector
from emotion_tracker import EmotionTracker

# --- UI drawing functions (one import per screen/mode) ---
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

# --- Shared state, schemas, and helpers (split out of main.py in a refactor) ---
# NOTE: app_state.py must exist for these imports to work.
from app_state import (
    SETTINGS_SCHEMA, FEATURES_SCHEMA, GAME_CATEGORIES, PERSONALITY_NAMES,
    start_game, open_menu, reset_all_modes, rebuild_controllers,
    _apply_voice_mode, apply_camera_resolution,
    finalize_active_challenge_run, update_challenge_logger_context,
    _dispatch_sounds, build_app_state, build_controllers,
    _AsyncChallengeStatsLogger, _IOWorker,
)

# --- Keyboard/voice/menu handlers (also split out in the same refactor) ---
from menu_handlers import (
    open_settings, open_features, apply_feature_toggle, handle_features_key,
    open_clone_setup, handle_clone_setup_key,
    open_player_stats, handle_player_stats_key,
    open_tutorial, _tutorial_steps, update_tutorial, handle_tutorial_key,
    handle_voice_tutorial_event, _advance_tutorial, handle_voice_nav,
    _run_gesture_nav, toggle_display_mode, switch_play_mode,
    get_active_controller, apply_setting_change,
    activate_menu_item, _launch_simulation, _launch_pvpvai_simulation,
    activate_settings_item, format_setting_value,
    handle_menu_key, handle_settings_key,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The OpenCV window title — must match the string passed to cv2.namedWindow().
WINDOW_NAME = "RPS Gesture Recogniser"

# Arrow keys return different integer codes on macOS vs Windows, so each
# direction is stored as a set that covers both platforms plus WASD.
KEY_ENTER = {10, 13}
KEY_ESC   = 27
KEY_UP    = {82, ord("w"), ord("W")}
KEY_DOWN  = {84, ord("s"), ord("S")}
KEY_LEFT  = {81, ord("a"), ord("A")}
KEY_RIGHT = {83, ord("d"), ord("D")}


# ---------------------------------------------------------------------------
# Background report updater
# ---------------------------------------------------------------------------

def _run_report_updater_bg():
    """
    Run the research-report updater on the background I/O thread.

    Always called via _io_worker.submit() so it never blocks the frame loop.
    Any failure is printed rather than crashing the app.
    """
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from report_updater import update_report
        result = update_report(verbose=False)
        if result:
            print(f"[Report] Updated -> {result}")
    except Exception as exc:
        print(f"[Report] Updater error: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _close_terminal():
    """
    Close the macOS Terminal window that launched this script.

    Uses AppleScript so it only works on macOS — on any other platform the
    subprocess will fail silently and we just carry on.
    """
    try:
        subprocess.Popen(
            ["osascript", "-e", 'tell application "Terminal" to close first window'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _update_streak(app_state, result_type):
    """
    Update the running win/lose streak counter based on the latest round result.

    A draw always resets the streak. A win extends the win streak (or starts a
    new one if we were on a lose streak), and vice versa for a loss.
    """
    if result_type == "draw":
        # Draws break any active streak.
        app_state["_streak_count"] = 0
        app_state["_streak_type"]  = ""
        return

    # If the new result continues the same streak type, just increment.
    # Otherwise reset to 1 for the new type.
    if app_state["_streak_type"] == result_type:
        app_state["_streak_count"] += 1
    else:
        app_state["_streak_type"]  = result_type
        app_state["_streak_count"] = 1


def _classify_result(banner):
    """
    Turn a result-banner string into one of "win", "lose", or "draw".

    The banner is a human-readable string like "YOU WIN!" or "ROBOT WINS".
    We check for win keywords first, then draw, then default to lose.
    """
    banner_upper = banner.upper()
    if "YOU WIN" in banner_upper or "SURVIVE" in banner_upper:
        return "win"
    elif "DRAW" in banner_upper:
        return "draw"
    else:
        return "lose"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    """
    Build app state, open the camera, and run the main frame loop.

    Each iteration of the loop:
      1. Reads a frame from the webcam.
      2. Updates a rolling FPS counter (smoothed over ~10 frames).
      3. Runs the correct hand-tracking path (single-player or two-player).
      4. Draws the appropriate screen for app_state["app_screen"].
      5. Shows the frame and waits 1 ms for a keypress.
      6. Dispatches voice events, then keyboard events, to the right handler.
    """
    # Build the big shared-state dictionary that holds everything.
    app_state = build_app_state()

    # Open the default camera (index 0).
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        raise SystemExit

    app_state["cap"] = cap
    # Apply the resolution the user has configured (e.g. 1280x720).
    apply_camera_resolution(cap, app_state["config"])

    cv2.namedWindow(WINDOW_NAME)

    # Open two separate MediaPipe hand-detector contexts:
    #   hands     -- used for in-game gesture recognition
    #   nav_hands -- used for gesture-nav cursor on menu screens
    # Keeping them separate prevents the nav cursor from interfering with
    # throw detection during a round.
    with create_hands_detector() as hands, create_nav_detector() as nav_hands:

        # On menu/settings screens we only run gesture-nav every other frame
        # to halve the MediaPipe load there.  _nav_skip_tick alternates 0/1.
        _nav_skip_tick = 0

        # ======================================================================
        # MAIN FRAME LOOP
        # ======================================================================
        while True:
            ret, frame = cap.read()
            if not ret:
                # Camera disconnected or end of file — bail out cleanly.
                print("Could not read frame.")
                finalize_active_challenge_run(app_state, status="abandoned")
                break

            # --- Rolling FPS (exponential moving average) ---
            # Weight 0.1 on the new sample smooths jitter over ~10 frames.
            _now = time.monotonic()
            _dt  = _now - app_state["_fps_last_t"]
            app_state["_fps_last_t"] = _now
            if _dt > 0:
                app_state["_fps_val"] = 0.9 * app_state["_fps_val"] + 0.1 / _dt

            # --- Gesture-nav throttle ---
            # On non-game screens we advance the skip tick so nav only runs on
            # even frames.  During GAME/TUTORIAL we always run every frame.
            _screen = app_state["app_screen"]
            _throttle_nav = _screen not in ("GAME", "TUTORIAL")
            if _throttle_nav:
                _nav_skip_tick = (_nav_skip_tick + 1) % 2
            else:
                _nav_skip_tick = 0

            # ==================================================================
            # GAME SCREEN
            # ==================================================================
            if _screen == "GAME":

                # Decide if the current mode needs two hands tracked at once.
                _is_two_player = app_state["play_mode"] in (
                    "TwoPlayerPvP", "PvPvAI", "ReflexTwoPlayer", "SimonSays2P"
                )
                # These are set inside the branches below; initialise to safe defaults.
                p1_tracker = p2_tracker = None
                show_session_summary    = False

                if _is_two_player:
                    # --- Two-player hand tracking ---
                    # process_two_hands_frame gives us separate hand dicts for each
                    # player so their gestures are tracked completely independently.
                    frame, p1_hand, p2_hand, _rgb = process_two_hands_frame(
                        frame=frame,
                        hands=hands,
                        hand_orientation=app_state["config"]["hand_orientation"],
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        ema_states=app_state["_tp_ema_states"],
                    )
                    p1_tracker = app_state["_tp_tracker_p1"].update(p1_hand["raw_gesture"])
                    p2_tracker = app_state["_tp_tracker_p2"].update(p2_hand["raw_gesture"])

                    # Cache the raw hand dicts so the diagnostic renderer can show them.
                    app_state["_tp_last_p1_hand"] = p1_hand
                    app_state["_tp_last_p2_hand"] = p2_hand

                    # Some shared code below reads hand_state/tracker_state using the
                    # single-player names, so alias P1 so it still works.
                    hand_state    = p1_hand
                    tracker_state = p1_tracker

                    # Emotion tracking isn't meaningful in two-player mode (only one
                    # face is expected on screen).
                    app_state["emotion_state"] = None

                    controller = get_active_controller(app_state)

                    # Reflex and Simon Says use a simpler update signature (no wrist Y).
                    if app_state["play_mode"] in ("ReflexTwoPlayer", "SimonSays2P"):
                        game_state = controller.update(
                            p1_tracker=p1_tracker,
                            p2_tracker=p2_tracker,
                            now=time.monotonic(),
                        )
                    else:
                        # TwoPlayerPvP and PvPvAI also need raw wrist Y for pump detection.
                        game_state = controller.update(
                            p1_tracker_state=p1_tracker,
                            p2_tracker_state=p2_tracker,
                            p1_wrist_y=p1_hand.get("raw_wrist_y") or p1_hand["wrist_y"],
                            p2_wrist_y=p2_hand.get("raw_wrist_y") or p2_hand["wrist_y"],
                            now=time.monotonic(),
                        )

                else:
                    # --- Single-player hand tracking ---
                    frame, hand_state, _rgb = process_hand_frame(
                        frame=frame,
                        hands=hands,
                        target_hand=app_state["target_hand"],
                        display_mode=app_state["display_mode"],
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        hand_orientation=app_state["config"]["hand_orientation"],
                        _ema_state=app_state["_ema_state"],
                    )

                    # --- Emotion tracking (optional, every 3rd frame to save CPU) ---
                    if app_state["config"].get("emotion_enabled"):
                        app_state["_emotion_frame_skip"] = (
                            app_state["_emotion_frame_skip"] + 1
                        ) % 3
                        if app_state["_emotion_frame_skip"] == 0:
                            app_state["emotion_state"] = app_state["emotion_tracker"].update(_rgb)
                    else:
                        app_state["emotion_state"] = None

                    tracker_state = app_state["tracker"].update(hand_state["raw_gesture"])

                    controller = get_active_controller(app_state)

                    # Give the AI a snapshot of the player's emotion so personality
                    # modes that adapt to mood can react accordingly.
                    if hasattr(controller, "set_emotion_snapshot"):
                        controller.set_emotion_snapshot(
                            app_state["emotion_tracker"].get_round_snapshot()
                        )

                    # Use the RAW (unsmoothed) wrist Y for pump/beat detection.
                    # The Kalman-filtered value adds lag that makes threshold
                    # detection fire at the wrong moment.
                    _pump_y = hand_state.get("raw_wrist_y") or hand_state["wrist_y"]

                    # Each mode's controller expects a slightly different signature,
                    # so dispatch to the right call.
                    _mode = app_state["play_mode"]
                    if _mode == "ReflexSolo":
                        game_state = controller.update(
                            tracker_state=tracker_state,
                            now=time.monotonic(),
                        )
                    elif _mode in ("BluffMode", "RPSLS"):
                        # Both of these modes watch wrist movement for bluff/pump timing.
                        game_state = controller.update(
                            tracker_state=tracker_state,
                            wrist_y=_pump_y,
                            now=time.monotonic(),
                        )
                    elif _mode == "SimonSaysSolo":
                        game_state = controller.update(
                            tracker_state=tracker_state,
                            now=time.monotonic(),
                        )
                    elif _mode == "SquidGame":
                        # SquidGame needs the full hand_state for freeze detection.
                        game_state = controller.update(
                            hand_state=hand_state,
                            now=time.monotonic(),
                        )
                    else:
                        # Cheat, FairPlay, Challenge, and Clone all share this signature.
                        game_state = controller.update(
                            wrist_y=_pump_y,
                            tracker_state=tracker_state,
                            now=time.monotonic(),
                        )

                # --- Sound effects ---
                # _dispatch_sounds also records the previous game state in
                # app_state["_snd_last_state"] so we can detect state transitions below.
                _dispatch_sounds(app_state, game_state)

                # --- Result flash + win/lose streak tracking ---
                # We want to trigger these exactly once: the first frame where the
                # game transitions INTO ROUND_RESULT.
                cur_state  = game_state.get("state", "")
                prev_state = app_state["_snd_last_state"]   # set by _dispatch_sounds above
                fi         = app_state["_flash_info"]

                if cur_state == "ROUND_RESULT" and prev_state != "ROUND_RESULT":
                    # Classify win/draw/lose so the flash uses the right colour.
                    result_type = _classify_result(game_state.get("result_banner", ""))
                    fi.update({"active": True, "result": result_type, "frame_idx": 0})

                    # Keep the result visible for 1.5 s after the round resolves so
                    # the player can read it before we return to WAITING_FOR_ROCK.
                    fi["replay_until"] = time.monotonic() + 1.5

                    # Update the streak counter using the helper above.
                    _update_streak(app_state, result_type)

                # Advance the flash animation frame counter and expire after 5 frames.
                if fi["active"]:
                    fi["frame_idx"] += 1
                    if fi["frame_idx"] >= 5:
                        fi["active"] = False

                # Show the current mic level in the flash overlay so the player
                # knows voice mode is listening.
                fi["mic_level"] = (
                    app_state["voice_controller"].get_mic_level()
                    if app_state.get("voice_mode_active") else 0.0
                )

                # Show the session summary overlay only when the whole match is done.
                show_session_summary = (
                    cur_state == "MATCH_RESULT"
                    and bool(game_state.get("session_summary"))
                )

                # Build the streak HUD label (shows e.g. "WIN STREAK  3").
                # We only show it when the streak is 2 or more — a single win
                # isn't interesting enough to display.
                streak_n = app_state["_streak_count"]
                streak_t = app_state["_streak_type"]
                if streak_n >= 2 and streak_t:
                    streak_label = (
                        f"WIN STREAK  {streak_n}"  if streak_t == "win"
                        else f"LOSE STREAK  {streak_n}"
                    )
                else:
                    streak_label = ""
                game_state["streak_label"] = streak_label

                # --- Voice beat/throw dispatch (GAME screen only) ---
                # These events were queued earlier this frame by the voice controller.
                # Forward them to the active game controller now that game_state exists.
                if app_state["voice_mode_active"]:
                    for event in app_state.pop("_voice_game_events", []):
                        if event["type"] == "beat" and hasattr(controller, "inject_voice_beat"):
                            controller.inject_voice_beat(event["word"])
                        elif event["type"] == "throw" and hasattr(controller, "inject_voice_throw"):
                            controller.inject_voice_throw(event["gesture"])

                # --- Tracker reset ---
                # When the game controller sets this flag it means a new throw window is
                # starting.  Clear the gesture tracker so the pump-Rock used during the
                # countdown isn't mistaken for the player's actual throw.
                if game_state.get("request_tracker_reset"):
                    app_state["tracker"].clear_for_new_throw()
                    app_state["_tp_tracker_p1"].clear_for_new_throw()
                    app_state["_tp_tracker_p2"].clear_for_new_throw()
                    if hasattr(controller, "consume_tracker_reset_request"):
                        controller.consume_tracker_reset_request()

                # --- Record round to player profile ---
                # We use a two-step approach so the emotion snapshot reflects the
                # player's REACTION to the result, not their face at the instant it resolves.
                #
                # Step 1 (entering ROUND_RESULT): store a pending entry with gestures + outcome.
                # Step 2 (leaving ROUND_RESULT): flush the pending entry with the current emotion.
                player_name = app_state["config"].get("player_name", "").strip()
                if player_name:
                    gs_state = game_state.get("state")

                    # Normalise the computer gesture field — different modes use different keys.
                    _comp_gest = (
                        game_state.get("computer_gesture")
                        or game_state.get("ai_actual")
                        or game_state.get("ai_gesture")
                        or "Unknown"
                    )

                    # A unique key for this round so we never double-record the same result.
                    gs_key = (
                        game_state.get("round_number", 0),
                        game_state.get("player_gesture"),
                        _comp_gest,
                    )

                    # Step 1: capture gestures + outcome on the first ROUND_RESULT frame.
                    if (
                        gs_state == "ROUND_RESULT"
                        and game_state.get("player_gesture") not in ("Unknown", "", None)
                        and _comp_gest not in ("Unknown", "", None)
                        and gs_key != app_state.get("_last_recorded_round")
                        and app_state.get("_pending_round_log") is None
                    ):
                        banner  = game_state.get("result_banner", "")
                        outcome = _classify_result(banner)
                        # Override the generic win/lose/draw with mode-specific language.
                        if "ROBOT" in banner.upper() or "AI WINS" in banner.upper() or "GAME OVER" in banner.upper():
                            outcome = "lose"

                        app_state["_pending_round_log"] = {
                            "key":            gs_key,
                            "player_gesture": game_state.get("player_gesture", "Unknown"),
                            "robot_gesture":  _comp_gest,
                            "outcome":        outcome,
                            "game_mode":      game_state.get("play_mode_label", ""),
                            "round_number":   game_state.get("round_number", 0),
                        }

                    # Step 2: once we leave ROUND_RESULT, flush the pending entry to disk.
                    # Dispatched to the background I/O thread so the JSON/Excel write
                    # doesn't stall the frame loop.
                    pending = app_state.get("_pending_round_log")
                    if pending and gs_state != "ROUND_RESULT":
                        app_state["_last_recorded_round"] = pending["key"]
                        app_state["_pending_round_log"]   = None
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

                # --- Rendering: pick the right draw function for the current mode ---

                if app_state["display_mode"] == "Diagnostic" and not _is_two_player:
                    # Diagnostic single-player view: shows raw landmarks, gesture data,
                    # and data-collection status instead of the normal game HUD.
                    app_state["landmark_collector"].update_landmarks(
                        hand_state.get("_landmarks")
                    )
                    collector_status = app_state["landmark_collector"].get_status_text()
                    top_right = (
                        collector_status
                        or "F Collect | T Train | E Face | 1-3 Mode | ESC Menu"
                    )
                    draw_top_bar(
                        frame,
                        f"DIAGNOSTIC | {game_state['play_mode_label'].upper()}",
                        top_right,
                    )
                    draw_info_panel(
                        frame=frame,
                        tracker_state=tracker_state,
                        game_state=game_state,
                        count_text=hand_state["count_text"],
                        status_text=hand_state["status_text"],
                        reason_text=hand_state["reason_text"],
                        ambiguous_count=hand_state["ambiguous_count"],
                        output_summary=(
                            app_state.get("collector_message")
                            or app_state["robot_output"].get_latest_summary()
                        ),
                        emotion_state=app_state.get("emotion_state"),
                        fps=app_state["_fps_val"],
                    )
                    draw_diagnostic_game_panel(frame, game_state)

                elif _is_two_player:
                    # Two-player rendering: pick the view that matches the active mode.
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
                    else:
                        # PvPvAI: three-way view.
                        draw_pvpvai_view(
                            frame, game_state,
                            p1_tracker_state=p1_tracker,
                            p2_tracker_state=p2_tracker,
                            colourblind=cb,
                        )

                # Single-player specialised views below.
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

                elif app_state["play_mode"] == "RPSLS":
                    draw_rpsls_view(
                        frame, game_state,
                        tracker_state=tracker_state,
                        hand_state=hand_state,
                    )

                else:
                    # Cheat, FairPlay, Challenge, and Clone all use the standard game view.
                    draw_game_mode_view(
                        frame, game_state,
                        emotion_state=app_state.get("emotion_state"),
                        voice_mode_active=app_state.get("voice_mode_active", False),
                        last_heard_word=(
                            app_state["voice_controller"].get_last_word()
                            if app_state.get("voice_mode_active") else ""
                        ),
                        tracker_state=tracker_state,
                        hand_state=hand_state,
                        flash_info=app_state["_flash_info"],
                        show_help=app_state.get("show_help", False),
                        sound_on=app_state["sound_player"].is_on(),
                        colourblind=app_state["config"].get("colourblind_mode", False),
                        show_session_summary=show_session_summary,
                    )

            # ==================================================================
            # MENU SCREEN
            # ==================================================================
            elif _screen == "MENU":
                _nav_enabled = app_state["config"].get("gesture_nav_enabled")

                if _nav_enabled and _nav_skip_tick == 0:
                    # Run the gesture-nav hand detector and update the cursor position.
                    frame, nav_hand, _ = process_hand_frame(
                        frame=frame, hands=nav_hands,
                        target_hand=app_state["target_hand"], display_mode="Game",
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        hand_orientation=app_state["config"]["hand_orientation"],
                    )
                    _n   = len(app_state["menu_items"])
                    # Space menu items evenly across the content region of the screen.
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
                    # No gesture nav — just mirror the frame so it feels like a webcam.
                    frame = cv2.flip(frame, 1)

                draw_menu_screen(
                    frame=frame,
                    menu_items=app_state["menu_items"],
                    selected_index=app_state["menu_index"],
                    config=app_state["config"],
                    show_help=app_state.get("show_help", False),
                    voice_mode_active=app_state.get("voice_mode_active", False),
                    in_submenu=False,
                )
                if _nav_enabled:
                    draw_gesture_nav_overlay(frame, app_state["gesture_nav"].get_cursor_info())

            # ==================================================================
            # GAME_CATEGORY SCREEN
            # ==================================================================
            elif _screen == "GAME_CATEGORY":
                frame = cv2.flip(frame, 1)
                draw_game_category_screen(
                    frame=frame,
                    categories=GAME_CATEGORIES,
                    category_index=app_state["game_category_index"],
                    mode_index=app_state["game_mode_index"],
                    in_mode_list=app_state.get("in_game_category", False),
                )

            # ==================================================================
            # SIMULATIONS HUB SCREEN
            # ==================================================================
            elif _screen == "SIMULATIONS":
                frame = cv2.flip(frame, 1)
                draw_simulations_hub_screen(
                    frame=frame,
                    selected_index=app_state.get("sim_tab_index", 0),
                    sim_state=app_state.get("sim_state", {}),
                )

            # ==================================================================
            # RPSLS TUTORIAL SCREEN
            # ==================================================================
            elif _screen == "RPSLS_TUTORIAL":
                # Track the hand so the renderer can show live gesture feedback.
                frame, _hand_for_tut, _ = process_hand_frame(
                    frame=frame, hands=hands,
                    target_hand=app_state["target_hand"], display_mode="Game",
                    handedness_threshold=app_state["config"]["handedness_threshold"],
                    hand_orientation=app_state["config"]["hand_orientation"],
                    _ema_state=app_state["_ema_state"],
                )
                draw_rpsls_tutorial_screen(
                    frame=frame,
                    step=app_state.get("rpsls_tutorial_step", 0),
                    hand_state=_hand_for_tut,
                )

            # ==================================================================
            # SIMULATION PROGRESS SCREEN
            # ==================================================================
            elif _screen == "SIMULATION":
                frame = cv2.flip(frame, 1)
                draw_simulation_screen(frame, app_state.get("sim_state", {}))

            # ==================================================================
            # SETTINGS SCREEN
            # ==================================================================
            elif _screen == "SETTINGS":
                # Disable gesture nav while the user is typing a text value — the
                # pinch gesture would conflict with text input.
                _nav_enabled = (
                    app_state["config"].get("gesture_nav_enabled")
                    and not app_state.get("_settings_text_edit", False)
                )
                if _nav_enabled and _nav_skip_tick == 0:
                    frame, nav_hand, _ = process_hand_frame(
                        frame=frame, hands=nav_hands,
                        target_hand=app_state["target_hand"], display_mode="Game",
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        hand_orientation=app_state["config"]["hand_orientation"],
                    )
                    _n   = len(SETTINGS_SCHEMA)
                    # Only choice and float items support left/right nudge gestures.
                    _adj = {
                        i for i, s in enumerate(SETTINGS_SCHEMA)
                        if s.get("type") in ("choice", "float")
                    }
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
                    cursor_info=(
                        app_state["gesture_nav"].get_cursor_info() if _nav_enabled else None
                    ),
                    text_edit=app_state.get("_settings_text_edit", False),
                )
                if _nav_enabled:
                    draw_gesture_nav_overlay(frame, app_state["gesture_nav"].get_cursor_info())

            # ==================================================================
            # FEATURES SCREEN
            # ==================================================================
            elif _screen == "FEATURES":
                _nav_enabled = app_state["config"].get("gesture_nav_enabled")
                if _nav_enabled and _nav_skip_tick == 0:
                    frame, nav_hand, _ = process_hand_frame(
                        frame=frame, hands=nav_hands,
                        target_hand=app_state["target_hand"], display_mode="Game",
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        hand_orientation=app_state["config"]["hand_orientation"],
                    )
                    _n    = len(FEATURES_SCHEMA)
                    # Only choice-type items (dropdowns) support left/right adjustment.
                    _fadj = {
                        i for i, s in enumerate(FEATURES_SCHEMA)
                        if s.get("type") == "choice"
                    }
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
                    cursor_info=(
                        app_state["gesture_nav"].get_cursor_info() if _nav_enabled else None
                    ),
                )
                if _nav_enabled:
                    draw_gesture_nav_overlay(frame, app_state["gesture_nav"].get_cursor_info())

            # ==================================================================
            # PERSONALITY SELECT SCREEN
            # ==================================================================
            elif _screen == "PERSONALITY_SELECT":
                frame    = cv2.flip(frame, 1)
                cur_name = PERSONALITY_NAMES[app_state.get("personality_index", 0)]
                draw_personality_settings(frame, cur_name, [])

            # ==================================================================
            # CLONE SETUP SCREEN
            # ==================================================================
            elif _screen == "CLONE_SETUP":
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
                    # Layout coordinates match the opponent list in draw_clone_setup_screen.
                    _run_gesture_nav(
                        app_state, nav_hand, time.monotonic(),
                        item_count=_n,
                        set_index_fn=lambda i: app_state.__setitem__(
                            "clone_opponent_index", i
                        ),
                        content_top=0.408,
                        content_bottom=0.408 + (_n - 1) * 0.072,
                    )
                else:
                    frame = cv2.flip(frame, 1)

                draw_clone_setup_screen(frame, {
                    "step":              app_state.get("clone_step", "enter_name"),
                    "text_buffer":       app_state.get("clone_text_buffer", ""),
                    "player_name":       app_state["config"].get("player_name", ""),
                    "available":         app_state.get("clone_available", []),
                    "selected_index":    app_state.get("clone_opponent_index", 0),
                    "all_players":       app_state.get("clone_all_players", []),
                    "message":           app_state.get("clone_message", ""),
                    "profiles_updating": app_state.get("clone_profiles_updating", False),
                })
                if _nav_enabled:
                    draw_gesture_nav_overlay(frame, app_state["gesture_nav"].get_cursor_info())

            # ==================================================================
            # PLAYER STATS SCREEN
            # ==================================================================
            elif _screen == "PLAYER_STATS":
                _nav_enabled = app_state["config"].get("gesture_nav_enabled")
                if _nav_enabled:
                    frame, nav_hand, _ = process_hand_frame(
                        frame=frame, hands=nav_hands,
                        target_hand=app_state["target_hand"], display_mode="Game",
                        handedness_threshold=app_state["config"]["handedness_threshold"],
                        hand_orientation=app_state["config"]["hand_orientation"],
                    )
                    players = app_state.get("stats_players", [])
                    _n      = max(len(players), 1)
                    _run_gesture_nav(
                        app_state, nav_hand, time.monotonic(),
                        item_count=_n,
                        set_index_fn=lambda i: app_state.__setitem__(
                            "stats_player_index", i
                        ),
                        content_top=0.328,
                        content_bottom=0.328 + (_n - 1) * 0.072,
                    )
                else:
                    frame = cv2.flip(frame, 1)

                draw_player_stats_screen(frame, {
                    "step":             app_state.get("stats_step", "select"),
                    "players":          app_state.get("stats_players", []),
                    "selected_index":   app_state.get("stats_player_index", 0),
                    "data":             app_state.get("stats_data"),
                    "traits":           app_state.get("stats_traits", []),
                    "rounds":           app_state.get("stats_rounds", []),
                    "sessions":         app_state.get("stats_sessions", []),
                    "filter":           app_state.get("stats_filter", "All"),
                    "tab":              app_state.get("stats_tab", "overview"),
                    "player_name_hint": app_state.get("stats_current_player", ""),
                })
                if _nav_enabled:
                    draw_gesture_nav_overlay(frame, app_state["gesture_nav"].get_cursor_info())

            # ==================================================================
            # TUTORIAL SCREEN
            # ==================================================================
            elif _screen == "TUTORIAL":
                # Run hand tracking so the tutorial can react to live gestures.
                frame, hand_state, _ = process_hand_frame(
                    frame=frame, hands=hands,
                    target_hand=app_state["target_hand"], display_mode="Game",
                    handedness_threshold=app_state["config"]["handedness_threshold"],
                    hand_orientation=app_state["config"]["hand_orientation"],
                )
                tracker_state = app_state["tracker"].update(hand_state["raw_gesture"])
                # Tick the tutorial state machine (advances steps when gestures are held).
                update_tutorial(app_state, hand_state, tracker_state)

                if app_state["config"].get("gesture_nav_enabled"):
                    steps_t = _tutorial_steps(app_state)
                    _n      = len(steps_t)
                    _run_gesture_nav(
                        app_state, hand_state, time.monotonic(),
                        item_count=_n,
                        set_index_fn=lambda i: app_state.__setitem__("tutorial_step", i),
                        content_top=0.44,
                        content_bottom=0.44 + (_n - 1) * (0.80 * 0.55 / max(_n, 1)),
                    )

                steps     = _tutorial_steps(app_state)
                step_idx  = app_state.get("tutorial_step", 0)
                # Clamp step_idx so we never index out of bounds.
                step_data = steps[step_idx] if step_idx < len(steps) else steps[-1]

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

            # --- Emotion debug overlay (Diagnostic mode only) ---
            # Draws facial landmark points over the frame for debugging the emotion
            # model.  Toggle with the 'e' key while in Diagnostic display mode.
            if app_state.get("emotion_debug") and app_state.get("display_mode") == "Diagnostic":
                debug_info = app_state["emotion_tracker"].get_debug_overlay(
                    frame.shape[1], frame.shape[0]
                )
                draw_emotion_debug(frame, debug_info)

            # Show the finished frame.  waitKey(1) blocks for 1 ms and returns any
            # key that was pressed, or -1 if none.  & 0xFF masks to a single byte
            # so it works correctly on all platforms.
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            # ----------------------------------------------------------------
            # Global voice navigation dispatch
            # ----------------------------------------------------------------
            # Drain every voice event that arrived since the last frame and
            # route it to the correct handler based on its type field.
            if app_state.get("voice_mode_active"):
                for event in app_state["voice_controller"].drain_events():
                    if event["type"] == "nav":
                        result = handle_voice_nav(app_state, event["action"])
                        if result == "quit":
                            # Voice-quit: clean up and exit immediately.
                            finalize_active_challenge_run(app_state, status="abandoned")
                            cap.release()
                            cv2.destroyAllWindows()
                            app_state["voice_controller"].stop()
                            app_state["emotion_tracker"].close()
                            _close_terminal()
                            return
                    elif event["type"] in ("beat", "throw"):
                        # Beat/throw events are only meaningful in GAME or TUTORIAL.
                        if app_state["app_screen"] == "GAME":
                            app_state.setdefault("_voice_game_events", []).append(event)
                        elif (
                            app_state["app_screen"] == "TUTORIAL"
                            and app_state.get("tutorial_voice_mode")
                        ):
                            handle_voice_tutorial_event(app_state, event)

            # ----------------------------------------------------------------
            # 'q' key: quit from any screen
            # ----------------------------------------------------------------
            if key == ord("q"):
                finalize_active_challenge_run(app_state, status="abandoned")
                break

            # ----------------------------------------------------------------
            # Per-screen keyboard dispatch
            # ----------------------------------------------------------------

            if _screen == "MENU":
                result = handle_menu_key(app_state, key)
                if result == "quit":
                    finalize_active_challenge_run(app_state, status="abandoned")
                    break

            elif _screen == "GAME_CATEGORY":
                # GAME_CATEGORY shares the same key handler as MENU (same navigation
                # model, just a different list of items).
                result = handle_menu_key(app_state, key)
                if result == "quit":
                    finalize_active_challenge_run(app_state, status="abandoned")
                    break

            elif _screen == "SIMULATIONS":
                # Let the user scroll between simulation tabs and launch one.
                _sim_tabs = ["Fair Play vs AI", "3-Way PvPvAI"]
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
                    else:
                        _launch_pvpvai_simulation(app_state)

            elif _screen == "SIMULATION":
                if key == KEY_ESC:
                    # Only allow going back once the simulation has finished or errored.
                    # Pressing ESC mid-run is intentionally ignored.
                    status = app_state.get("sim_state", {}).get("status", "idle")
                    if status in ("done", "error", "idle"):
                        app_state["app_screen"]    = "SIMULATIONS"
                        app_state["sim_tab_index"] = 0

            elif _screen == "SETTINGS":
                handle_settings_key(app_state, key)

            elif _screen == "FEATURES":
                handle_features_key(app_state, key)

            elif _screen == "PERSONALITY_SELECT":
                # Small inline handler — only a handful of keys are needed here.
                if key == KEY_ESC:
                    app_state["app_screen"] = "FEATURES"
                elif key in KEY_UP:
                    app_state["personality_index"] = (
                        (app_state["personality_index"] - 1) % len(PERSONALITY_NAMES)
                    )
                elif key in KEY_DOWN:
                    app_state["personality_index"] = (
                        (app_state["personality_index"] + 1) % len(PERSONALITY_NAMES)
                    )
                elif key in KEY_ENTER:
                    chosen = PERSONALITY_NAMES[app_state["personality_index"]]
                    app_state["config"]["ai_personality"] = chosen
                    # Apply the new personality to every AI controller immediately
                    # so the change takes effect without needing to restart a game.
                    for ctrl_key in ("fair_controller", "challenge_controller",
                                     "clone_controller", "bluff_controller"):
                        ctrl = app_state.get(ctrl_key)
                        if ctrl and hasattr(ctrl, "ai") and hasattr(ctrl.ai, "set_personality"):
                            ctrl.ai.set_personality(chosen)
                    app_state["app_screen"] = "FEATURES"
                    print(f"[Personality] Set to: {chosen}")

            elif _screen == "CLONE_SETUP":
                handle_clone_setup_key(app_state, key)

            elif _screen == "PLAYER_STATS":
                handle_player_stats_key(app_state, key)

            elif _screen == "TUTORIAL":
                handle_tutorial_key(app_state, key)

            elif _screen == "RPSLS_TUTORIAL":
                # Page through 6 slides with arrow/enter, then auto-launch RPSLS.
                n_steps = 6
                if key == KEY_ESC or key == ord("q"):
                    # Return to wherever the player came from.
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
                        # On the last slide, pressing next launches the actual game.
                        start_game(app_state, "RPSLS", from_category=True)
                elif key in KEY_LEFT or key in KEY_UP:
                    step = app_state.get("rpsls_tutorial_step", 0)
                    app_state["rpsls_tutorial_step"] = max(0, step - 1)

            elif _screen == "GAME":
                if key == KEY_ESC:
                    app_state["show_help"] = False
                    # If the game was launched from the category screen, go back there.
                    # Otherwise return to the main menu.
                    if app_state.get("_came_from_category"):
                        if app_state["play_mode"] == "Challenge":
                            finalize_active_challenge_run(app_state, status="abandoned")
                        app_state["app_screen"]       = "GAME_CATEGORY"
                        app_state["in_game_category"] = True
                        reset_all_modes(app_state)
                    else:
                        open_menu(app_state)
                elif key == ord("?"):
                    # Toggle the in-game help overlay.
                    app_state["show_help"] = not app_state.get("show_help", False)
                elif key == ord("m"):
                    # Cycle between Normal and Diagnostic display modes.
                    toggle_display_mode(app_state)
                elif key == ord("e"):
                    # Toggle the emotion debug overlay (facial landmark dots).
                    app_state["emotion_debug"] = not app_state["emotion_debug"]
                    print(f"[Emotion] Debug overlay: {'ON' if app_state['emotion_debug'] else 'OFF'}")
                elif key == ord("n"):
                    # Toggle sound effects on or off.
                    on = app_state["sound_player"].toggle()
                    print(f"[Sound] {'ON' if on else 'OFF'}")
                elif key == ord("1"):
                    switch_play_mode(app_state, "Cheat")
                elif key == ord("2"):
                    switch_play_mode(app_state, "FairPlay")
                elif key == ord("3"):
                    switch_play_mode(app_state, "Challenge")

                # --- Data collection keys (only active in Diagnostic display mode) ---
                elif app_state["display_mode"] == "Diagnostic":
                    if key == ord("f"):
                        # Toggle the landmark data collector on/off.
                        is_on = app_state["landmark_collector"].toggle()
                        app_state["collector_message"] = (
                            "Collection ON - 7=Rock 8=Scissors 9=Paper"
                            if is_on else "Collection OFF"
                        )

                    elif key in (ord("7"), ord("8"), ord("9")):
                        # Record the current frame's landmarks labelled as Rock / Scissors / Paper.
                        _ok, _label, msg = app_state["landmark_collector"].try_record(key)
                        if msg:
                            app_state["collector_message"] = msg

                    elif key == ord("t"):
                        # Retrain the front-on gesture classifier from the collected data.
                        app_state["collector_message"] = "Training model..."
                        print("[Main] Training front-on model...")
                        from front_on_trainer import train_and_save
                        accuracy = train_and_save()
                        app_state["collector_message"] = (
                            f"Model trained! Accuracy: {accuracy:.0%}"
                            if accuracy is not None
                            else "Training failed - need more samples"
                        )
                        from front_on_classifier import reload_model
                        reload_model()

                    elif key in (ord("r"), ord("R")):
                        # Manually trigger the research report updater.
                        app_state["collector_message"] = "Updating research report..."
                        _io_worker.submit(_run_report_updater_bg)

                    elif key in (ord("h"), ord("H")):
                        # Toggle the hardware test mode for ESP32 serial testing.
                        try:
                            from serial_bridge import SerialBridge
                            from hardware_test_mode import HardwareTestController
                            if "hardware_test" not in app_state:
                                app_state["hardware_test"] = HardwareTestController(SerialBridge())
                                app_state["collector_message"] = (
                                    "Hardware Test: [ ] ports  Enter connect  R/P/S send  X quit"
                                )
                            else:
                                del app_state["hardware_test"]
                                app_state["collector_message"] = "Hardware Test exited"
                        except ImportError:
                            app_state["collector_message"] = (
                                "Hardware test requires pyserial - pip install pyserial"
                            )

            # '?' toggles the help overlay from any screen (handled above per-screen
            # for GAME, and here as a global fallback for all other screens).
            if key == ord("?"):
                app_state["show_help"] = not app_state.get("show_help", False)

    # ----------------------------------------------------------------
    # Shutdown
    # ----------------------------------------------------------------
    # Flush any queued background I/O tasks (e.g. pending round profile writes)
    # before releasing resources so we don't lose data on exit.
    _io_worker.flush()
    app_state["voice_controller"].stop()
    app_state["emotion_tracker"].close()
    cap.release()
    cv2.destroyAllWindows()
    _close_terminal()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback as _tb
    import datetime  as _dt

    try:
        run()
    except Exception as _exc:
        # --- Crash reporter ---
        # If run() throws an unhandled exception, write a timestamped crash
        # report to ~/Desktop/CapStone/ so it's easy to find and share.
        _ts        = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _crash_dir = os.path.join(os.path.expanduser("~"), "Desktop", "CapStone")
        os.makedirs(_crash_dir, exist_ok=True)
        _crash_path = os.path.join(_crash_dir, f"crash_{_ts}.txt")

        _report = (
            f"RPS Robot Crash Report\n"
            f"======================\n"
            f"Time:    {_ts}\n"
            f"Error:   {type(_exc).__name__}: {_exc}\n\n"
            f"Traceback:\n"
            f"{_tb.format_exc()}\n"
        )

        # Best-effort write — if the Desktop isn't writable we still print.
        try:
            with open(_crash_path, "w") as _f:
                _f.write(_report)
        except Exception:
            pass

        print("\n" + "=" * 60)
        print("CRASH REPORT")
        print("=" * 60)
        print(_report)
        print(f"Report saved to: {_crash_path}")
        print("=" * 60)
        raise
