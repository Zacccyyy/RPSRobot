"""
rps_game_state.py
=================
Cheat Mode — the simplest game mode in the project.

The "AI" doesn't actually play fair. It watches what gesture the player
throws and immediately outputs the move that would beat it. This is mainly
used to demonstrate real-time gesture recognition — whatever the player shows,
the robot arm is commanded to play the winning counter.

Two input methods are supported:
  - Gesture (default): pump-beat countdown then throw with your hand.
  - Voice mode: say "Ready" → "One" → "Two" → "Three" → then name your throw.

How it fits into the project:
  - Standalone controller; no inheritance from other classes.
  - Renderer calls draw_rps_game_view() with the dict from _build_output().
  - Main loop calls update() every frame.
  - inject_voice_beat() / inject_voice_throw() are called by the speech thread.
"""

import time


# The three gestures this mode recognises
VALID_GESTURES = {"Rock", "Paper", "Scissors"}

# Given any gesture, this maps to the gesture that beats it
WIN_MAP = {
    "Rock":     "Paper",
    "Paper":    "Scissors",
    "Scissors": "Rock",
}


class RPSGameController:
    """
    Cheat Mode controller.
    Waits for the player to throw a gesture, then immediately plays the
    counter-move that beats it.
    """

    def __init__(
        self,
        robot_output=None,
        down_threshold=0.045,
        up_threshold=0.035,
        beat_cooldown=0.18,
        rock_grace_period=0.50,
        shoot_window_seconds=0.55,
        shoot_change_guard_seconds=0.05,
        rock_assume_seconds=0.13,
        result_display_seconds=1.80
    ):
        # Optional hardware interface — if connected, _lock_round() will publish
        # results so the robot arm can physically move
        self.robot_output = robot_output

        # Beat detection thresholds (calibrated — don't change these)
        self.down_threshold     = down_threshold      # how far the wrist must drop to count a beat
        self.up_threshold       = up_threshold        # how far it must rise before the next beat
        self.beat_cooldown      = beat_cooldown       # minimum time between beats
        self.rock_grace_period  = rock_grace_period   # how long we keep counting after Rock disappears

        # Shoot window timing (calibrated — don't change these)
        self.shoot_window_seconds       = shoot_window_seconds        # total window duration
        self.shoot_change_guard_seconds = shoot_change_guard_seconds  # ignore gestures right after window opens
        self.rock_assume_seconds        = rock_assume_seconds         # if only Rock seen this long, assume Rock
        self.result_display_seconds     = result_display_seconds      # how long to show the result

        self._voice_mode = False
        self.reset_round()

    def reset(self):
        """Public alias so the main loop can call reset() without knowing internals."""
        self.reset_round()

    def set_voice_mode(self, enabled):
        """Switch between gesture input (default) and voice command input."""
        self._voice_mode = bool(enabled)

    def inject_voice_beat(self, word, now=None):
        """
        Called by the speech recognition thread when a beat word is heard.
        Each recognised word advances the countdown by one step.

        Word mapping:
          "ready" → enters COUNTDOWN from WAITING_FOR_ROCK
          "one"   → beat 1
          "two"   → beat 2
          "three" → opens SHOOT_WINDOW immediately with a generous 2.5s window
        """
        if now is None:
            now = time.monotonic()

        cooldown_ok = (now - self.last_beat_time) >= self.beat_cooldown

        # "Ready" kicks off the countdown from the waiting screen
        if self.state == "WAITING_FOR_ROCK" and word == "ready":
            self.state          = "COUNTDOWN"
            self.beat_count     = 0
            self.last_beat_time = now
            return

        # All other beat words only apply during the countdown
        if self.state != "COUNTDOWN":
            return

        if word in ("one", "two") and cooldown_ok:
            # Each of these increments the beat count by one
            self.beat_count     += 1
            self.last_beat_time  = now

        elif word == "three" and cooldown_ok:
            # "Three" is the shoot cue — open the window immediately
            self.last_beat_time  = now
            self.beat_count      = 4   # 4 means "SHOOT" on the display
            self.state           = "SHOOT_WINDOW"
            self.shoot_open_time  = now
            # Give a generous 2.5s window so voice users have time to say the gesture
            self.shoot_close_time = now + max(self.shoot_window_seconds, 2.50)

    def inject_voice_throw(self, gesture, now=None):
        """
        Called by the speech recognition thread when the player says a gesture name.
        Only accepted during the SHOOT_WINDOW phase with a valid gesture.
        """
        if now is None:
            now = time.monotonic()

        if self.state == "SHOOT_WINDOW" and gesture in VALID_GESTURES:
            self._lock_round(gesture, now)

    def reset_round(self):
        """
        Reset all per-round state to get ready for a new round.
        Voice mode is a session-level setting and is preserved across resets.
        """
        # getattr with a default handles the case where this is called before
        # _voice_mode is first assigned (during __init__)
        self._voice_mode = getattr(self, "_voice_mode", False)

        self.state          = "WAITING_FOR_ROCK"
        self.beat_count     = 0
        self.phase          = "ready_for_down"  # beat detector phase: down or up stroke
        self.top_y          = None              # highest wrist Y seen since last beat
        self.bottom_y       = None              # lowest wrist Y seen this downstroke
        self.last_beat_time = 0.0
        self.last_rock_time = 0.0

        self.shoot_open_time  = None
        self.shoot_close_time = None

        self.player_gesture     = "Unknown"
        self.computer_gesture   = "Unknown"
        self.robot_move_command = "PENDING"
        self.result_banner      = ""
        self.result_until       = None
        self.gesture_assumed    = False  # True if we defaulted to Rock due to no detection

    def _lock_round(self, player_gesture, now):
        """
        Record the player's throw, compute the counter-move, and transition
        to the ROUND_RESULT state. Also notifies the robot hardware if connected.
        """
        self.player_gesture   = player_gesture
        # WIN_MAP gives us the move that beats whatever the player threw
        self.computer_gesture = WIN_MAP[player_gesture]
        self.robot_move_command = f"ROBOT_PLAY_{self.computer_gesture.upper()}"
        self.result_banner    = "ROBOT TAKES THE ROUND"

        self.state        = "ROUND_RESULT"
        self.result_until = now + self.result_display_seconds

        # Publish to hardware if a robot output interface is wired up
        if self.robot_output is not None:
            self.robot_output.publish_round_result(
                command=self.robot_move_command,
                game_mode="Cheat",
                round_result="robot_win",
                player_gesture=self.player_gesture,
                robot_gesture=self.computer_gesture,
                metadata={"banner": self.result_banner},
            )

    def _get_fallback_throw(self, tracker_state):
        """
        Try to recover a gesture when the SHOOT window closes without a clean
        detection. We try three confidence levels in order (most to least reliable):
          1. stable_gesture   — short-window stabilised reading
          2. confirmed_gesture — already tried in the main path, but check again
          3. raw_gesture       — single-frame reading, most lenient
        Returns "Unknown" if none of them give a valid gesture.
        """
        stable_gesture    = tracker_state.get("stable_gesture",    "Unknown")
        confirmed_gesture = tracker_state.get("confirmed_gesture", "Unknown")
        raw_gesture       = tracker_state.get("raw_gesture",       "Unknown")

        if stable_gesture    in VALID_GESTURES: return stable_gesture
        if confirmed_gesture in VALID_GESTURES: return confirmed_gesture
        if raw_gesture       in VALID_GESTURES: return raw_gesture

        return "Unknown"

    def _build_output(self, now):
        """
        Build the state dict that the renderer reads each frame.
        Every state gets its own main_text and sub_text so the view stays simple.
        """
        # These fields are included in every state's output
        base = {
            "play_mode_label":    "Cheat Mode",
            "state":              self.state,
            "beat_count":         self.beat_count,
            "time_left":          0.0,
            "player_gesture":     self.player_gesture,
            "computer_gesture":   self.computer_gesture,
            "robot_move_command": self.robot_move_command,
            "result_banner":      self.result_banner,
            "score_text":         "",
            "round_text":         "",
            "round_number":       0,
            "player_score":       0,
            "robot_score":        0,
            "gesture_assumed":    self.gesture_assumed,
        }

        if self.state == "WAITING_FOR_ROCK":
            # Prompt differs between gesture mode ("make a fist") and voice mode
            base.update({
                "state_label": "Waiting",
                "main_text":   "MAKE A FIST" if not self._voice_mode else 'Say  "READY"',
                "sub_text":    ("Hold Rock, then pump downward 4 times"
                                if not self._voice_mode
                                else "to start the countdown"),
            })
            return base

        if self.state == "COUNTDOWN":
            if self._voice_mode:
                # Tell the player which word to say next based on beat count
                next_words = {0: '"ONE"', 1: '"TWO"', 2: '"THREE"'}
                word      = next_words.get(self.beat_count, '"THREE"')
                main_text = f"Say  {word}"
            else:
                # Show "READY" on beat 0, then the beat number (cap at 3 for display)
                main_text = "READY" if self.beat_count == 0 else str(min(self.beat_count, 3))
            base.update({
                "state_label": "Countdown",
                "main_text":   main_text,
                "sub_text":    "Cheat mode counters after SHOOT",
            })
            return base

        if self.state == "SHOOT_WINDOW":
            base.update({
                "state_label": "Shoot Window",
                "main_text":   "SHOOT!",
                "sub_text":    "Throw Rock, Paper, or Scissors now",
                # Voice mode: no countdown bar — window stays open until a gesture is spoken
                "time_left": 0.0 if self._voice_mode else max(0.0, self.shoot_close_time - now),
            })
            return base

        if self.state == "ROUND_RESULT":
            base.update({
                "state_label": "Round Result",
                "main_text":   self.result_banner,
                "sub_text":    "Cheat mode always counters your throw",
                "time_left":   max(0.0, self.result_until - now),
            })
            return base

        # Catch-all for any unexpected state value
        base.update({"state_label": "Unknown", "main_text": "UNKNOWN", "sub_text": ""})
        return base

    def update(self, wrist_y, tracker_state, now=None):
        """
        Main tick — call once per frame.

        wrist_y       : normalised Y coordinate of the wrist (0.0 = top of frame)
        tracker_state : dict from the gesture tracker
        now           : optional monotonic timestamp (uses time.monotonic() if omitted)
        """
        if now is None:
            now = time.monotonic()

        confirmed_gesture = tracker_state.get("confirmed_gesture", "Unknown")
        stable_gesture    = tracker_state.get("stable_gesture",    "Unknown")

        # ── ROUND_RESULT: hold the result on screen, then start the next round ─
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                self.reset_round()
            return self._build_output(now)

        # Pre-compute Rock detection flags — used in the next two states
        confirmed_rock = confirmed_gesture == "Rock"
        stable_rock    = stable_gesture    == "Rock"

        # ── WAITING_FOR_ROCK: prompt the player to make a fist ───────────────
        if self.state == "WAITING_FOR_ROCK":
            if self._voice_mode:
                # In voice mode, inject_voice_beat("ready") handles the transition
                return self._build_output(now)
            if confirmed_rock and wrist_y is not None:
                # Rock detected — start tracking the beat and enter the countdown
                self.state          = "COUNTDOWN"
                self.phase          = "ready_for_down"
                self.top_y          = wrist_y
                self.bottom_y       = wrist_y
                self.last_rock_time = now
            return self._build_output(now)

        # ── COUNTDOWN: count 4 pump beats, then open the SHOOT window ────────
        if self.state == "COUNTDOWN":
            if self._voice_mode:
                # Voice mode: inject_voice_beat() drives beat_count — nothing to do here
                return self._build_output(now)

            # We keep tracking during the grace period even if Rock briefly drops out
            rock_detected = (confirmed_rock or stable_rock) and wrist_y is not None
            within_grace  = (now - self.last_rock_time) <= self.rock_grace_period
            can_track     = rock_detected or (within_grace and wrist_y is not None
                                              and self.beat_count > 0)

            if rock_detected:
                self.last_rock_time = now  # refresh the grace window

            if can_track:
                if self.phase == "ready_for_down":
                    # Track the highest (lowest Y value) wrist position seen since last beat
                    if self.top_y is None:
                        self.top_y = wrist_y
                    self.top_y = min(self.top_y, wrist_y)

                    moved_down_enough = (wrist_y - self.top_y) >= self.down_threshold
                    cooldown_ok       = (now - self.last_beat_time) >= self.beat_cooldown

                    if moved_down_enough and cooldown_ok:
                        # Downstroke counts as one beat
                        self.beat_count    += 1
                        self.last_beat_time = now
                        self.phase          = "waiting_for_up"
                        self.bottom_y       = wrist_y

                        # 4 beats means "shoot" — open the window
                        if self.beat_count >= 4:
                            self.state            = "SHOOT_WINDOW"
                            self.shoot_open_time  = now
                            self.shoot_close_time = now + self.shoot_window_seconds

                elif self.phase == "waiting_for_up":
                    # Track the lowest position reached during the downstroke
                    if self.bottom_y is None:
                        self.bottom_y = wrist_y
                    self.bottom_y = max(self.bottom_y, wrist_y)

                    if (self.bottom_y - wrist_y) >= self.up_threshold:
                        # Upstroke complete — ready for the next downstroke
                        self.phase = "ready_for_down"
                        self.top_y = wrist_y

            else:
                # Rock is gone and the grace period has expired — abort the countdown
                if not within_grace:
                    self.reset_round()

            return self._build_output(now)

        # ── SHOOT_WINDOW: detect the player's thrown gesture ─────────────────
        if self.state == "SHOOT_WINDOW":
            time_since_open = now - self.shoot_open_time

            if self._voice_mode:
                # Voice mode: inject_voice_throw() handles locking the round
                return self._build_output(now)

            # Guard period: ignore gestures right after the window opens because
            # the hand hasn't had time to transition away from Rock yet
            if time_since_open >= self.shoot_change_guard_seconds:
                # Paper or Scissors are unambiguous — lock in immediately
                if stable_gesture in {"Paper", "Scissors"}:
                    self._lock_round(stable_gesture, now)
                    return self._build_output(now)

            # If only Rock has been seen long enough, treat it as a Rock throw
            if time_since_open >= self.rock_assume_seconds:
                self.gesture_assumed = True
                self._lock_round("Rock", now)
                return self._build_output(now)

            # Window expired without an early detection — try the fallback chain
            if now >= self.shoot_close_time:
                fallback = self._get_fallback_throw(tracker_state)
                if fallback in VALID_GESTURES:
                    self._lock_round(fallback, now)
                else:
                    # Nothing detected at all — give up and reset
                    self.reset_round()

            return self._build_output(now)

        return self._build_output(now)
