"""
rps_game_state.py
=================
Cheat Mode — the simplest game mode in the project.

The "AI" does not really play.  Instead, it waits for the player to throw
a gesture and immediately outputs the counter-move that would beat it.
This is used to demonstrate real-time gesture recognition: whatever the
player shows, the robot arm is commanded to play the winning response.

Two input methods are supported:
  - Gesture (default): pump-beat countdown then throw with your hand.
  - Voice mode: say "Ready" → "One" → "Two" → "Three" → then name your throw.

Where this fits in the codebase:
  - Standalone controller; no inheritance
  - Renderer calls draw_rps_game_view() with the dict from _build_output()
  - Main loop calls update() every frame
  - inject_voice_beat() / inject_voice_throw() are called by the speech thread
"""

import time


# The three gestures this mode recognises
VALID_GESTURES = {"Rock", "Paper", "Scissors"}

# Maps each gesture to the one that beats it — used to pick the counter-move
WIN_MAP = {
    "Rock": "Paper",
    "Paper": "Scissors",
    "Scissors": "Rock",
}


class RPSGameController:
    """
    Cheat Mode controller.

    - counts the player's throw
    - instantly outputs the winning counter-move
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
        # Optional hardware interface — if provided, _lock_round() will publish
        # the result so the robot arm can move
        self.robot_output = robot_output

        # ── Beat detection thresholds (calibrated, do not change) ─────────────
        self.down_threshold              = down_threshold
        self.up_threshold                = up_threshold
        self.beat_cooldown               = beat_cooldown
        self.rock_grace_period           = rock_grace_period

        # ── Shoot window timing (calibrated, do not change) ──────────────────
        self.shoot_window_seconds        = shoot_window_seconds
        self.shoot_change_guard_seconds  = shoot_change_guard_seconds   # guard against residual Rock
        self.rock_assume_seconds         = rock_assume_seconds           # assume Rock if nothing else seen
        self.result_display_seconds      = result_display_seconds

        self._voice_mode = False
        self.reset_round()

    def reset(self):
        """Alias so the main loop can call reset() without knowing internals."""
        self.reset_round()

    def set_voice_mode(self, enabled):
        """Enable or disable voice-command input mode."""
        self._voice_mode = bool(enabled)

    def inject_voice_beat(self, word, now=None):
        """
        Called by the speech recognition thread when a beat word is spoken.
        Advances the countdown by one step per recognised word.

        Word mapping:
          "ready" → enters COUNTDOWN from WAITING_FOR_ROCK
          "one"   → beat 1 (beat_count becomes 1)
          "two"   → beat 2 (beat_count becomes 2)
          "three" → jumps straight to SHOOT_WINDOW with an extended window
        """
        if now is None:
            now = time.monotonic()

        cooldown_ok = (now - self.last_beat_time) >= self.beat_cooldown

        # "Ready" transitions from the waiting screen into the countdown
        if self.state == "WAITING_FOR_ROCK" and word == "ready":
            self.state          = "COUNTDOWN"
            self.beat_count     = 0
            self.last_beat_time = now
            return

        # Only process beat words during COUNTDOWN
        if self.state != "COUNTDOWN":
            return

        if word in ("one", "two") and cooldown_ok:
            self.beat_count     += 1
            self.last_beat_time  = now

        elif word == "three" and cooldown_ok:
            # "Three" is the shoot cue — open a generous 2.5 s window
            self.last_beat_time  = now
            self.beat_count      = 4         # display shows "SHOOT"
            self.state           = "SHOOT_WINDOW"
            self.shoot_open_time  = now
            self.shoot_close_time = now + max(self.shoot_window_seconds, 2.50)

    def inject_voice_throw(self, gesture, now=None):
        """
        Called by the speech recognition thread when the player names a gesture.
        Only accepts the throw if we are in SHOOT_WINDOW and the gesture is valid.
        """
        if now is None:
            now = time.monotonic()

        if self.state == "SHOOT_WINDOW" and gesture in VALID_GESTURES:
            self._lock_round(gesture, now)

    def reset_round(self):
        """
        Reset all per-round state to get ready for a new round.
        Voice mode is preserved across round resets (it's a session-level setting).
        """
        self._voice_mode = getattr(self, "_voice_mode", False)  # preserve across rounds
        self.state              = "WAITING_FOR_ROCK"
        self.beat_count         = 0
        self.phase              = "ready_for_down"
        self.top_y              = None
        self.bottom_y           = None
        self.last_beat_time     = 0.0
        self.last_rock_time     = 0.0

        self.shoot_open_time    = None
        self.shoot_close_time   = None

        self.player_gesture     = "Unknown"
        self.computer_gesture   = "Unknown"
        self.robot_move_command = "PENDING"
        self.result_banner      = ""
        self.result_until       = None
        self.gesture_assumed    = False   # True if Rock was assumed because nothing was detected

    def _lock_round(self, player_gesture, now):
        """
        Record the player's throw, compute the counter-move, and transition
        to the ROUND_RESULT state.  Also publishes to the robot output if one
        is connected.
        """
        self.player_gesture   = player_gesture
        # WIN_MAP[gesture] is what beats that gesture — that's the robot's move
        self.computer_gesture = WIN_MAP[player_gesture]
        self.robot_move_command = f"ROBOT_PLAY_{self.computer_gesture.upper()}"
        self.result_banner    = "ROBOT TAKES THE ROUND"

        self.state        = "ROUND_RESULT"
        self.result_until = now + self.result_display_seconds

        # Publish the result to the hardware interface if one is wired up
        if self.robot_output is not None:
            self.robot_output.publish_round_result(
                command=self.robot_move_command,
                game_mode="Cheat",
                round_result="robot_win",
                player_gesture=self.player_gesture,
                robot_gesture=self.computer_gesture,
                metadata={
                    "banner": self.result_banner,
                }
            )

    def _get_fallback_throw(self, tracker_state):
        """
        Try to salvage a gesture from the tracker when the SHOOT window closes
        without a confident confirmed gesture.  We try in order:
          1. stable_gesture   (short-window stabilised)
          2. confirmed_gesture (already tried in main path but check again)
          3. raw_gesture      (single-frame, most lenient)
        Returns "Unknown" if none work.
        """
        stable_gesture    = tracker_state.get("stable_gesture",    "Unknown")
        confirmed_gesture = tracker_state.get("confirmed_gesture",  "Unknown")
        raw_gesture       = tracker_state.get("raw_gesture",        "Unknown")

        if stable_gesture    in VALID_GESTURES:
            return stable_gesture
        if confirmed_gesture in VALID_GESTURES:
            return confirmed_gesture
        if raw_gesture       in VALID_GESTURES:
            return raw_gesture

        return "Unknown"

    def _build_output(self, now):
        """
        Build the state dict for the renderer.
        Each game state gets its own main_text and sub_text so the view
        code is kept simple.
        """
        # Common fields present in every state
        base = {
            "play_mode_label":   "Cheat Mode",
            "state":             self.state,
            "beat_count":        self.beat_count,
            "time_left":         0.0,
            "player_gesture":    self.player_gesture,
            "computer_gesture":  self.computer_gesture,
            "robot_move_command":self.robot_move_command,
            "result_banner":     self.result_banner,
            "score_text":        "",
            "round_text":        "",
            "round_number":      0,
            "player_score":      0,
            "robot_score":       0,
            "gesture_assumed":   self.gesture_assumed,
        }

        if self.state == "WAITING_FOR_ROCK":
            # Prompt text changes depending on whether we're in voice or gesture mode
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
                # Tell the player which word to say next
                next_words = {0: '"ONE"', 1: '"TWO"', 2: '"THREE"'}
                _nw        = next_words.get(self.beat_count, '"THREE"')
                main_text  = f"Say  {_nw}"
            else:
                # Show "READY" on beat 0, then the beat number (capped at 3)
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
                # Voice mode: no countdown — window stays open until throw is spoken
                "time_left":   0.0 if self._voice_mode else max(0.0, self.shoot_close_time - now),
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

        # Catch-all for any unexpected state
        base.update({
            "state_label": "Unknown",
            "main_text":   "UNKNOWN",
            "sub_text":    "",
        })
        return base

    def update(self, wrist_y, tracker_state, now=None):
        """
        Main tick — call once per frame.

        wrist_y       : normalised Y coordinate of the wrist (0.0 = top of frame)
        tracker_state : dict from the gesture tracker
        now           : optional monotonic timestamp
        """
        if now is None:
            now = time.monotonic()

        confirmed_gesture = tracker_state.get("confirmed_gesture", "Unknown")
        stable_gesture    = tracker_state.get("stable_gesture",    "Unknown")

        # ── ROUND_RESULT: wait for display timer then start the next round ────
        if self.state == "ROUND_RESULT":
            if now >= self.result_until:
                self.reset_round()
            return self._build_output(now)

        # Pre-compute Rock flags used by both WAITING_FOR_ROCK and COUNTDOWN
        confirmed_rock = confirmed_gesture == "Rock"
        stable_rock    = stable_gesture    == "Rock"

        # ── WAITING_FOR_ROCK: prompt player to make a fist ───────────────────
        if self.state == "WAITING_FOR_ROCK":
            if self._voice_mode:
                # In voice mode the speech thread handles state transitions
                return self._build_output(now)
            if confirmed_rock and wrist_y is not None:
                # Rock detected — initialise beat tracking and enter countdown
                self.state          = "COUNTDOWN"
                self.phase          = "ready_for_down"
                self.top_y          = wrist_y
                self.bottom_y       = wrist_y
                self.last_rock_time = now
            return self._build_output(now)

        # ── COUNTDOWN: count 4 pump beats then open SHOOT window ─────────────
        if self.state == "COUNTDOWN":
            if self._voice_mode:
                # Voice mode: the speech thread drives beat_count via inject_voice_beat()
                # — we do nothing here except return the current display state
                return self._build_output(now)

            # Determine whether we can track the wrist right now
            rock_detected = (confirmed_rock or stable_rock) and wrist_y is not None
            within_grace  = (now - self.last_rock_time) <= self.rock_grace_period
            # We track during grace even if Rock is temporarily not confirmed
            can_track     = rock_detected or (within_grace and wrist_y is not None
                                               and self.beat_count > 0)

            if rock_detected:
                self.last_rock_time = now

            if can_track:
                if self.phase == "ready_for_down":
                    # Update the highest wrist position seen since last beat
                    if self.top_y is None:
                        self.top_y = wrist_y
                    self.top_y = min(self.top_y, wrist_y)

                    moved_down_enough = (wrist_y - self.top_y) >= self.down_threshold
                    cooldown_ok       = (now - self.last_beat_time) >= self.beat_cooldown

                    if moved_down_enough and cooldown_ok:
                        # Downstroke registered — count the beat
                        self.beat_count    += 1
                        self.last_beat_time = now
                        self.phase          = "waiting_for_up"
                        self.bottom_y       = wrist_y

                        # 4 beats = transition to SHOOT_WINDOW
                        if self.beat_count >= 4:
                            self.state            = "SHOOT_WINDOW"
                            self.shoot_open_time  = now
                            self.shoot_close_time = now + self.shoot_window_seconds

                elif self.phase == "waiting_for_up":
                    # Track the lowest point reached after the downstroke
                    if self.bottom_y is None:
                        self.bottom_y = wrist_y
                    self.bottom_y = max(self.bottom_y, wrist_y)

                    moved_up_enough = (self.bottom_y - wrist_y) >= self.up_threshold
                    if moved_up_enough:
                        # Upstroke complete — ready for the next downstroke
                        self.phase = "ready_for_down"
                        self.top_y = wrist_y

            else:
                # Rock is gone and grace period expired — abort the countdown
                if not within_grace:
                    self.reset_round()

            return self._build_output(now)

        # ── SHOOT_WINDOW: detect the player's thrown gesture ─────────────────
        if self.state == "SHOOT_WINDOW":
            time_since_open = now - self.shoot_open_time

            if self._voice_mode:
                # Voice mode: inject_voice_throw() handles locking the round
                return self._build_output(now)

            # Guard period: ignore gestures immediately after the window opens
            # because the hand hasn't had time to transition away from Rock yet
            if time_since_open >= self.shoot_change_guard_seconds:
                # Paper or Scissors — confident detection, lock in immediately
                if stable_gesture in {"Paper", "Scissors"}:
                    self._lock_round(stable_gesture, now)
                    return self._build_output(now)

            # Rock assumption: if only Rock has been seen for long enough,
            # just treat it as a Rock throw
            if time_since_open >= self.rock_assume_seconds:
                self.gesture_assumed = True
                self._lock_round("Rock", now)
                return self._build_output(now)

            # Window timed out without an early detection — try the fallback chain
            if now >= self.shoot_close_time:
                fallback_throw = self._get_fallback_throw(tracker_state)

                if fallback_throw in VALID_GESTURES:
                    self._lock_round(fallback_throw, now)
                else:
                    # Completely failed to detect anything — reset and try again
                    self.reset_round()

            return self._build_output(now)

        return self._build_output(now)
