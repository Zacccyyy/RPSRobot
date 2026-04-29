"""
discord_reporter.py
===================
Sends crash reports and player feedback to a private Discord channel
via webhook. This is the developer's channel — players never see it.

Data sent:
  Crash reports:  timestamp, OS, Python version, git hash, error + traceback
  Feedback:       player name, message, timestamp, git hash

Data NOT sent:
  - Gameplay data or round history
  - Camera or video data
  - IP addresses (Discord may log these server-side, per Discord's own policy)
  - Any data beyond what the player explicitly types in the feedback form

Consent:
  Nothing is sent unless the player explicitly accepted the privacy notice
  on first launch. The consent preference is stored in config.json and can
  be changed at any time in Settings.

Configuration:
  The webhook URL is stored in config.json under "discord_webhook_url".
  Set it once from Settings → Developer → Set Discord Webhook.
  Leave blank to disable sending entirely (local save still works).
"""

import json
import sys
import threading
import urllib.request
import urllib.error


# Character limit per Discord message (2000 is Discord's hard limit)
_DISCORD_LIMIT = 1950


def _post(webhook_url: str, content: str) -> bool:
    """
    POST a message to a Discord webhook.
    Truncates content if over Discord's 2000-char limit.
    Returns True on success, False on any error.
    """
    if not webhook_url or not webhook_url.startswith("https://discord.com/api/webhooks/"):
        return False

    if len(content) > _DISCORD_LIMIT:
        content = content[:_DISCORD_LIMIT - 20] + "\n...(truncated)"

    payload = json.dumps({"content": f"```\n{content}\n```"}).encode("utf-8")

    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status in (200, 204)
    except (urllib.error.URLError, OSError):
        return False
    except Exception:
        return False


def send_crash_report(webhook_url: str, report: str) -> None:
    """
    Send a crash report to Discord in a background thread.
    Fire-and-forget — never blocks the main process.
    """
    if not webhook_url:
        return

    header  = "🔴 CRASH REPORT"
    message = f"{header}\n{'─' * 40}\n{report}"

    t = threading.Thread(
        target=_post,
        args=(webhook_url, message),
        daemon=True,
        name="CrashReporter",
    )
    t.start()


def send_feedback(webhook_url: str, player: str, text: str,
                  version: str = "") -> None:
    """
    Send player feedback to Discord in a background thread.
    Fire-and-forget — never blocks the main process.
    """
    if not webhook_url:
        return

    ver     = version[:7] if version else "unknown"
    message = (
        f"💬 PLAYER FEEDBACK\n"
        f"{'─' * 40}\n"
        f"Player:  {player}\n"
        f"Version: {ver}\n"
        f"\n"
        f"{text.strip()}"
    )

    t = threading.Thread(
        target=_post,
        args=(webhook_url, message),
        daemon=True,
        name="FeedbackSender",
    )
    t.start()


def validate_webhook_url(url: str) -> bool:
    """
    Check if a string looks like a valid Discord webhook URL.
    Does not make a network request.
    """
    return (
        isinstance(url, str) and
        url.startswith("https://discord.com/api/webhooks/") and
        len(url) > 60
    )
