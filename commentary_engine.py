"""
commentary_engine.py
====================
Live AI commentary for RPS games using the Claude API.

After each resolved round this module sends a short game-state summary to
Claude and stores the reply as a subtitle for the renderer to display during
the ROUND_RESULT state.

Commentary can cover:
    - What the player threw and whether the AI predicted it
    - The player's current behavioural pattern (win-stay, gesture bias, etc.)
    - The AI's opponent-type detection
    - Streak momentum
    - Running session tendencies

Model: claude-sonnet-4-20250514

The API call runs in a background daemon thread so it NEVER blocks the
30 fps camera/render loop.  If the API is unavailable, commentary is simply
empty — the game always continues regardless.

Requirements:
    ANTHROPIC_API_KEY environment variable must be set.
    No extra packages needed — uses only stdlib urllib.
"""

import os
import threading
import time
import json

# The Claude model used for commentary generation.
_CLAUDE_MODEL = "claude-sonnet-4-20250514"

# System prompt: tells Claude exactly what role to play and what format to use.
# Keep this tight — shorter prompts = faster responses = less lag.
_SYSTEM_PROMPT = """You are a sharp, witty sports commentator for a Rock Paper Scissors AI match.
After each round, produce exactly ONE sentence (max 20 words) of commentary.
Focus on: the player's patterns, the AI's prediction, psychological tendencies, or match momentum.
Be specific, insightful, and occasionally cheeky. No generic lines.
Respond with ONLY the commentary sentence. No quotes, no prefix, no explanation."""


class CommentaryEngine:
    """
    Non-blocking commentary generator.

    Call on_round_result() when a round resolves.  Read get_latest() from
    the render loop each frame to display the most recent commentary line.

    At most one API call is in-flight at any time, and calls are rate-limited
    to at most one every _min_gap seconds so we don't spam the API.
    """

    def __init__(self, enabled=True):
        self.enabled  = enabled
        self._latest  = ""     # most recent commentary text for the renderer
        self._pending = False  # True while an API request is in-flight
        self._last_req = 0.0   # monotonic time of the last request
        self._min_gap  = 3.0   # minimum seconds between requests (rate limit)
        self._lock     = threading.Lock()  # guards _latest and _pending

    def toggle(self) -> bool:
        """
        Toggle commentary on or off.  Clears the displayed text when turning off.
        Returns the new enabled state (True = on).
        """
        self.enabled = not self.enabled
        if not self.enabled:
            with self._lock:
                self._latest = ""
        return self.enabled

    def get_latest(self) -> str:
        """
        Return the most recently fetched commentary line (thread-safe).
        Returns an empty string if no commentary is available yet.
        """
        with self._lock:
            return self._latest

    def clear(self):
        """Clear the displayed commentary immediately (e.g. at the start of a new game)."""
        with self._lock:
            self._latest = ""
        self._pending = False

    def on_round_result(self, game_state):
        """
        Trigger commentary generation after a round resolves.

        This is non-blocking — it fires a background thread and returns
        immediately.  Does nothing if:
            - Commentary is disabled.
            - A request is already in-flight.
            - The minimum gap between requests hasn't elapsed yet.

        game_state: the game's current state dict (see _build_prompt for the keys used).
        """
        if not self.enabled:
            return

        now = time.monotonic()

        # Respect the rate limit and don't stack requests.
        if self._pending or (now - self._last_req) < self._min_gap:
            return

        self._pending  = True
        self._last_req = now

        # Daemon thread so it doesn't block process exit if the app closes.
        t = threading.Thread(
            target=self._fetch,
            args=(self._build_prompt(game_state),),
            daemon=True,
        )
        t.start()

    def _build_prompt(self, gs) -> str:
        """
        Build a concise summary of the current game state to send to Claude.

        We only include information that's directly useful for commentary:
        recent history, scores, detected player type, and any AI insight.
        """
        player   = gs.get("player_gesture", "?")
        # Support both naming conventions used across different game modes.
        robot    = gs.get("computer_gesture", gs.get("ai_prediction", "?"))
        banner   = gs.get("result_banner", "")
        opp_type = gs.get("opponent_type", "")
        p_score  = gs.get("player_score", 0)
        r_score  = gs.get("robot_score", gs.get("ai_score", 0))
        rn       = gs.get("round_number", 1)
        history  = gs.get("history", [])
        mode     = gs.get("play_mode_label", "RPS")
        insight  = gs.get("last_insight", "")

        # Summarise the last five rounds into a compact string like
        # "Rock(W), Paper(L), Scissors(D)" so Claude has recent context.
        recent   = history[-5:] if history else []
        hist_str = ", ".join(
            f"{r['player_gesture']}({r['player_outcome'][0].upper()})"
            for r in recent
            if r.get("player_gesture") and r.get("player_outcome")
        ) or "no history yet"

        # Build the prompt as a list of lines then join them.
        prompt_parts = [
            f"Game: {mode}",
            f"Round {rn}: Player threw {player}, AI threw {robot}. Result: {banner}",
            f"Score: Player {p_score} - AI {r_score}",
            f"Recent history (last 5): {hist_str}",
        ]

        # Only include opponent type if it's something meaningful.
        if opp_type and opp_type not in ("random", "unknown", ""):
            prompt_parts.append(f"AI detected player type: {opp_type}")
        if insight:
            prompt_parts.append(f"Context: {insight}")

        return "\n".join(prompt_parts)

    def _fetch(self, prompt):
        """
        Make the HTTP request to the Claude API and store the result.

        Runs in a background daemon thread.  On success, stores the commentary
        text in _latest.  Silently ignores all failures — commentary is
        non-critical and the game must continue no matter what.
        """
        try:
            import urllib.request

            # Build the request payload for the Claude messages API.
            payload = json.dumps({
                "model":      _CLAUDE_MODEL,
                "max_tokens": 60,           # one sentence is well within 60 tokens
                "system":     _SYSTEM_PROMPT,
                "messages":   [{"role": "user", "content": prompt}],
            }).encode("utf-8")

            # If the API key isn't set there's nothing we can do.
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                return

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type":      "application/json",
                    "anthropic-version": "2023-06-01",
                    "x-api-key":         api_key,
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=8) as resp:
                data    = json.loads(resp.read())
                content = data.get("content", [])

                # The response contains a list of content blocks — we want the first text one.
                text = next((c["text"] for c in content if c.get("type") == "text"), "")
                text = text.strip().strip('"').strip("'")  # remove any stray quote characters

                if text:
                    with self._lock:
                        self._latest = text

        except Exception:
            pass  # commentary failures are silent — never crash or log here

        finally:
            # Always mark the request as done, even if it failed.
            self._pending = False
