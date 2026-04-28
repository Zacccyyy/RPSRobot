"""
main.py
=======
Entry point.  Imports app_state and menu_handlers, then calls run().
The bulk of the logic has been moved to:
  app_state.py      -- state construction, schemas, core helpers
  menu_handlers.py  -- key handlers, menu/nav/tutorial/simulation logic
  ui_base.py        -- drawing primitives
  ui_game.py        -- in-game screen renderers
  ui_modes.py       -- per-mode screen renderers
  ui_menus.py       -- menu/settings/stats/tutorial screen renderers
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
    _AsyncChallengeStatsLogger, _IOWorker,
)
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

def run():
    app_state = build_app_state()

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
                                                               "SimonSays2P")
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
                    )

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

                    _mode = app_state["play_mode"]
                    if _mode == "ReflexSolo":
                        game_state = controller.update(
                            tracker_state=tracker_state,
                            now=time.monotonic(),
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
                        )
                    elif _mode == "SquidGame":
                        game_state = controller.update(
                            hand_state=hand_state,
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

                # --- Flash + streak on round result TRANSITION only ---
                cur_state = game_state.get("state", "")
                prev_state = app_state["_snd_last_state"]  # already updated by _dispatch_sounds
                fi = app_state["_flash_info"]
                if cur_state == "ROUND_RESULT" and prev_state != "ROUND_RESULT":
                    banner = game_state.get("result_banner", "").upper()
                    if "YOU WIN" in banner or "SURVIVE" in banner:
                        result_type = "win"
                    elif "DRAW" in banner:
                        result_type = "draw"
                    else:
                        result_type = "lose"
                    fi.update({"active": True, "result": result_type, "frame_idx": 0})

                    # Streak update: only fires once per round
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
                        # Draw resets streak
                        app_state["_streak_count"] = 0
                        app_state["_streak_type"]  = ""

                if fi["active"]:
                    fi["frame_idx"] += 1
                    if fi["frame_idx"] >= 5:
                        fi["active"] = False
                fi["mic_level"] = app_state["voice_controller"].get_mic_level() \
                    if app_state.get("voice_mode_active") else 0.0

                # Replay: set a 1.5s window for the last-round display in WAITING_FOR_ROCK
                if cur_state == "ROUND_RESULT" and prev_state != "ROUND_RESULT":
                    fi["replay_until"] = time.monotonic() + 1.5

                # Session summary: show during MATCH_RESULT
                show_session_summary = (
                    cur_state == "MATCH_RESULT"
                    and bool(game_state.get("session_summary"))
                )

                # Streak label for HUD (persists across states)
                streak_n = app_state["_streak_count"]
                streak_t = app_state["_streak_type"]
                if streak_n >= 2 and streak_t:
                    label = f"WIN STREAK  {streak_n}" if streak_t == "win" else f"LOSE STREAK  {streak_n}"
                    game_state["streak_label"] = label
                else:
                    game_state["streak_label"] = ""

                # --- Voice beat/throw dispatch (GAME only) ---
                if app_state["voice_mode_active"]:
                    for event in app_state.pop("_voice_game_events", []):
                        if event["type"] == "beat":
                            if hasattr(controller, "inject_voice_beat"):
                                controller.inject_voice_beat(event["word"])
                        elif event["type"] == "throw":
                            if hasattr(controller, "inject_voice_throw"):
                                controller.inject_voice_throw(event["gesture"])

                if game_state.get("request_tracker_reset"):
                    app_state["tracker"].clear_for_new_throw()
                    # Two-player: clear both hand trackers so pump-Rock
                    # doesn't carry over as the throw in the SHOOT window
                    app_state["_tp_tracker_p1"].clear_for_new_throw()
                    app_state["_tp_tracker_p2"].clear_for_new_throw()

                    if hasattr(controller, "consume_tracker_reset_request"):
                        controller.consume_tracker_reset_request()

                # --- Record round to player profile ---
                # Emotion is captured at the END of the ROUND_RESULT display
                # (after the player has had time to react to win/loss/draw),
                # not at the moment the round resolves.
                player_name = app_state["config"].get("player_name", "").strip()
                if player_name:
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
                        banner = game_state.get("result_banner", "")
                        if "YOU WIN" in banner or "YOU SURVIVE" in banner:
                            outcome = "win"
                        elif "ROBOT" in banner or "AI WINS" in banner or "GAME OVER" in banner:
                            outcome = "lose"
                        else:
                            outcome = "draw"

                        app_state["_pending_round_log"] = {
                            "key":            gs_key,
                            "player_gesture": game_state.get("player_gesture", "Unknown"),
                            "robot_gesture":  _comp_gest,
                            "outcome":        outcome,
                            "game_mode":      game_state.get("play_mode_label", ""),
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

                if app_state["display_mode"] == "Diagnostic" and not _is_two_player:
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

            elif app_state["app_screen"] == "GAME_CATEGORY":
                result = handle_menu_key(app_state, key)
                if result == "quit":
                    finalize_active_challenge_run(app_state, status="abandoned")
                    break

            elif app_state["app_screen"] == "SIMULATIONS":
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
                    # Return to game category screen if that's where we came from,
                    # otherwise fall back to main menu
                    if app_state.get("_came_from_category"):
                        if app_state["play_mode"] == "Challenge":
                            finalize_active_challenge_run(app_state, status="abandoned")
                        app_state["app_screen"]       = "GAME_CATEGORY"
                        app_state["in_game_category"] = True   # keep mode list open
                        reset_all_modes(app_state)
                    else:
                        open_menu(app_state)
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
                        # Hardware test mode - wires serial bridge for ESP32 testing
                        try:
                            from serial_bridge import SerialBridge
                            from hardware_test_mode import HardwareTestController
                            if "hardware_test" not in app_state:
                                app_state["hardware_test"] = HardwareTestController(SerialBridge())
                                app_state["collector_message"] = "Hardware Test: [ ] ports  Enter connect  R/P/S send  X quit"
                            else:
                                del app_state["hardware_test"]
                                app_state["collector_message"] = "Hardware Test exited"
                        except ImportError:
                            app_state["collector_message"] = "Hardware test requires pyserial - pip install pyserial"

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
    """Close the terminal window that launched this app (macOS)."""
    try:
        subprocess.Popen(
            [
                "osascript", "-e",
                'tell application "Terminal" to close first window',
            ],
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

        _report = (
            f"RPS Robot Crash Report\n"
            f"======================\n"
            f"Time:    {_ts}\n"
            f"Error:   {type(_exc).__name__}: {_exc}\n\n"
            f"Traceback:\n"
            f"{_tb.format_exc()}\n"
        )

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