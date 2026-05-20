"""
robot_output.py
===============
Tracks what the robot arm has been told to do each round.

There are two phases for each round:
  - "locked"   -- the robot's move has been chosen internally but the
                  round is not over yet (player hasn't thrown yet).
  - "resolved" -- the round is finished and the result is known.

RobotOutputBuffer stores these events in memory so other parts of the
app (e.g. the UI, serial sender) can query what happened last.
"""

import time
from collections import deque


class RobotOutputBuffer:
    """
    Stores robot-output events so the rest of the app can read them.

    Uses a deque with a fixed max size so memory usage stays bounded
    even across a very long session.
    """

    def __init__(self, history_limit=200):
        # Circular buffer: oldest events fall off the end automatically.
        self.history = deque(maxlen=history_limit)

        # The single most recent packet (locked or resolved).
        self.latest_packet = None

        # The most recent "locked" packet that hasn't been resolved yet.
        # Set to None once the round resolves.
        self.pending_locked = None

    def clear_pending_locked(self):
        """Clear the pending locked move (call this after the round resolves)."""
        self.pending_locked = None

    def stage_locked_move(self, command, game_mode, metadata=None):
        """
        Record that the robot has internally chosen its move for this round.

        command   -- the serial command string sent to the robot arm
        game_mode -- which game mode is active (e.g. "FairPlay", "Cheat")
        metadata  -- optional dict of extra info for debugging

        Returns the packet dict that was stored.
        """
        packet = {
            "timestamp":    time.monotonic(),
            "phase":        "locked",
            "command":      command,
            "game_mode":    game_mode,
            "round_result": "pending",   # result is not known yet at this stage
            "metadata":     metadata or {},
        }

        # Store the packet in all three places so callers can find it easily.
        self.pending_locked = packet
        self.latest_packet  = packet
        self.history.append(packet)

        print(f"[RobotOutput] LOCKED | {game_mode} | {command}")
        return packet

    def publish_round_result(
        self,
        command,
        game_mode,
        round_result,
        player_gesture,
        robot_gesture,
        metadata=None
    ):
        """
        Record the final outcome of a round once both gestures are known.

        command        -- the serial command that was sent
        game_mode      -- active game mode
        round_result   -- "win", "lose", or "draw" (from the robot's perspective)
        player_gesture -- what the player threw (e.g. "Rock")
        robot_gesture  -- what the robot threw (e.g. "Paper")
        metadata       -- optional dict of extra info

        Returns the packet dict that was stored.
        """
        packet = {
            "timestamp":      time.monotonic(),
            "phase":          "resolved",
            "command":        command,
            "game_mode":      game_mode,
            "round_result":   round_result,
            "player_gesture": player_gesture,
            "robot_gesture":  robot_gesture,
            "metadata":       metadata or {},
        }

        self.latest_packet  = packet
        self.pending_locked = None   # round is over, no longer pending
        self.history.append(packet)

        print(
            f"[RobotOutput] RESOLVED | {game_mode} | {command} | "
            f"{round_result} | player={player_gesture} | robot={robot_gesture}"
        )
        return packet

    def get_latest_summary(self):
        """
        Return a short human-readable string describing the most recent event.
        Useful for debug overlays and log lines.
        """
        # Nothing has been recorded yet this session.
        if self.latest_packet is None:
            return "No robot output yet"

        # Pull the key fields out of the packet with safe fallbacks.
        phase     = self.latest_packet.get("phase",        "?").upper()
        game_mode = self.latest_packet.get("game_mode",    "?")
        command   = self.latest_packet.get("command",      "?")
        result    = self.latest_packet.get("round_result", "?")

        return f"{phase} | {game_mode} | {command} | {result}"
