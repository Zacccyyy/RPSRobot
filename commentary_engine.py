"""
commentary_engine.py
====================
Live AI commentary for RPS games using the Claude API.

Generates 1-2 sentence scouting observations after each resolved round,
displayed as a subtitle overlay during ROUND_RESULT state.

Commentary analyses:
- What gesture the player threw and why it was/wasn't predicted
- Player's current pattern (win-stay, lose-shift, gesture bias)
- AI's opponent type detection
- Streak momentum
- Running session tendencies

Uses claude-sonnet-4-20250514. Runs async in a background thread
so it never blocks the 30fps camera loop.
"""

import threading
import time
import json

_SYSTEM_PROMPT = """You are a sharp, witty sports commentator for a Rock Paper Scissors AI match.
After each round, produce exactly ONE sentence (max 20 words) of commentary.
Focus on: the player's patterns, the AI's prediction, psychological tendencies, or match momentum.
Be specific, insightful, and occasionally cheeky. No generic lines.
Respond with ONLY the commentary sentence. No quotes, no prefix, no explanation."""


class CommentaryEngine:
    """
    Non-blocking commentary generator.

    Usage:
        engine = CommentaryEngine()
        engine.on_round_result(game_state)   # call after each resolved round
        line = engine.get_latest()            # read from render loop each frame
    """

    def __init__(self, enabled=True):
        self.enabled    = enabled
        self._latest    = ""
        self._pending   = False
        self._last_req  = 0.0
        self._min_gap   = 3.0   # don't spam — at most once per 3 seconds
        self._lock      = threading.Lock()

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            with self._lock:
                self._latest = ""
        return self.enabled

    def get_latest(self):
        with self._lock:
            return self._latest

    def clear(self):
        with self._lock:
            self._latest = ""
        self._pending = False

    def on_round_result(self, game_state):
        """Call this when a round resolves. Fires an async API call."""
        if not self.enabled:
            return
        now = time.monotonic()
        if self._pending or (now - self._last_req) < self._min_gap:
            return
        self._pending  = True
        self._last_req = now
        t = threading.Thread(
            target=self._fetch,
            args=(self._build_prompt(game_state),),
            daemon=True,
        )
        t.start()

    def _build_prompt(self, gs):
        player   = gs.get("player_gesture", "?")
        robot    = gs.get("computer_gesture", gs.get("ai_prediction", "?"))
        banner   = gs.get("result_banner", "")
        opp_type = gs.get("opponent_type", "")
        p_score  = gs.get("player_score", 0)
        r_score  = gs.get("robot_score", gs.get("ai_score", 0))
        rn       = gs.get("round_number", 1)
        history  = gs.get("history", [])
        mode     = gs.get("play_mode_label", "RPS")
        insight  = gs.get("last_insight", "")

        # Build recent history summary
        recent = history[-5:] if history else []
        hist_str = ", ".join(
            f"{r['player_gesture']}({r['player_outcome'][0].upper()})"
            for r in recent
            if r.get("player_gesture") and r.get("player_outcome")
        ) or "no history yet"

        prompt_parts = [
            f"Game: {mode}",
            f"Round {rn}: Player threw {player}, AI threw {robot}. Result: {banner}",
            f"Score: Player {p_score} - AI {r_score}",
            f"Recent history (last 5): {hist_str}",
        ]
        if opp_type and opp_type not in ("random", "unknown", ""):
            prompt_parts.append(f"AI detected player type: {opp_type}")
        if insight:
            prompt_parts.append(f"Context: {insight}")

        return "\n".join(prompt_parts)

    def _fetch(self, prompt):
        try:
            import urllib.request
            payload = json.dumps({
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 60,
                "system":     _SYSTEM_PROMPT,
                "messages":   [{"role": "user", "content": prompt}],
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type":      "application/json",
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data     = json.loads(resp.read())
                content  = data.get("content", [])
                text     = next((c["text"] for c in content if c.get("type") == "text"), "")
                text     = text.strip().strip('"').strip("'")
                if text:
                    with self._lock:
                        self._latest = text
        except Exception as exc:
            # Silently fail — commentary is non-critical
            pass
        finally:
            self._pending = False
