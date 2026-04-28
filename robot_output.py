import time
from collections import deque


class RobotOutputBuffer:
    """
    Stores robot-output events in a serial-ready structure.

    Two event types:
    - locked: robot move has been pre-locked internally
    - resolved: round result is known and fully published
    """

    def __init__(self, history_limit=200):
        self.history = deque(maxlen=history_limit)
        self.latest_packet = None
        self.pending_locked = None

    def clear_pending_locked(self):
        self.pending_locked = None

    def stage_locked_move(self, command, game_mode, metadata=None):
        packet = {
            "timestamp": time.monotonic(),
            "phase": "locked",
            "command": command,
            "game_mode": game_mode,
            "round_result": "pending",
            "metadata": metadata or {},
        }

        self.pending_locked = packet
        self.latest_packet = packet
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
        packet = {
            "timestamp": time.monotonic(),
            "phase": "resolved",
            "command": command,
            "game_mode": game_mode,
            "round_result": round_result,
            "player_gesture": player_gesture,
            "robot_gesture": robot_gesture,
            "metadata": metadata or {},
        }

        self.latest_packet = packet
        self.pending_locked = None
        self.history.append(packet)

        print(
            f"[RobotOutput] RESOLVED | {game_mode} | {command} | "
            f"{round_result} | player={player_gesture} | robot={robot_gesture}"
        )
        return packet

    def get_latest_summary(self):
        if self.latest_packet is None:
            return "No robot output yet"

        phase = self.latest_packet.get("phase", "?").upper()
        game_mode = self.latest_packet.get("game_mode", "?")
        command = self.latest_packet.get("command", "?")
        result = self.latest_packet.get("round_result", "?")

        return f"{phase} | {game_mode} | {command} | {result}"