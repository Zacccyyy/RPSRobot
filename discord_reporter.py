"""
discord_reporter.py
===================
Sends crash reports and player feedback to a private Discord channel
via webhook. This is the developer's channel  -  players never see it.

The DEFAULT_WEBHOOK_URL is hardcoded so all users' reports reach the
developer automatically. The config can override it with a custom URL
(useful if the webhook ever needs to be rotated without a code push).

Data sent:
  Crash reports:  timestamp, OS, Python version, git hash, error + traceback
  Feedback:       player name, message, timestamp, git hash

Data NOT sent:
  - Gameplay data or round history
  - Camera or video data
  - Any data beyond what the player explicitly types in the feedback form

Consent:
  Nothing is sent unless the player explicitly accepted the privacy notice
  on first launch. Stored in config.json under "analytics_consent".
"""

import json
import threading
import urllib.request
import urllib.error

# -- Hardcoded default webhook (Option A) -------------------------------------
# Replace this URL with your actual Discord webhook URL.
# All users send to this channel automatically if they accept the privacy notice.
# To get a webhook URL:
#   Discord → your server → channel settings → Integrations → Webhooks → New Webhook → Copy URL
DEFAULT_WEBHOOK_URL = "https://discord.com/api/webhooks/1499027787757916302/yrriB3OWpwRvqhg-rKIhMkDU_m3wQ4AooRBZIknOE6WJSpKj7gkhEb0dUV9sRZuo0R6n"

# Discord hard limit per message
_DISCORD_LIMIT = 1950


def _resolve_url(config_url: str) -> str:
    """
    Use config URL if set and valid, otherwise fall back to hardcoded default.
    This allows rotating the webhook without a code push by updating config.
    """
    if config_url and config_url.startswith("https://discord.com/api/webhooks/"):
        return config_url
    if DEFAULT_WEBHOOK_URL.startswith("https://discord.com/api/webhooks/"):
        return DEFAULT_WEBHOOK_URL
    return ""


def _post(webhook_url: str, content: str) -> bool:
    """
    POST a message to a Discord webhook.
    Returns True on success, False on any error.
    """
    if not webhook_url:
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
    except Exception:
        return False


def send_crash_report(config_url: str, report: str) -> None:
    """
    Send a crash report to Discord in a background thread.
    Fire-and-forget  -  never blocks the main process.
    """
    url = _resolve_url(config_url)
    if not url:
        return

    message = f"[CRASH REPORT]\n{'-' * 40}\n{report}"
    threading.Thread(
        target=_post, args=(url, message),
        daemon=True, name="CrashReporter"
    ).start()


def send_feedback(config_url: str, player: str, text: str,
                  version: str = "") -> None:
    """
    Send player feedback to Discord in a background thread.
    Fire-and-forget  -  never blocks the main process.
    """
    url = _resolve_url(config_url)
    if not url:
        return

    ver     = version[:7] if version else "unknown"
    message = (
        f"[PLAYER FEEDBACK]\n"
        f"{'-' * 40}\n"
        f"Player:  {player}\n"
        f"Version: {ver}\n"
        f"\n"
        f"{text.strip()}"
    )
    threading.Thread(
        target=_post, args=(url, message),
        daemon=True, name="FeedbackSender"
    ).start()


def validate_webhook_url(url: str) -> bool:
    """Check if a string looks like a valid Discord webhook URL."""
    return (
        isinstance(url, str) and
        url.startswith("https://discord.com/api/webhooks/") and
        len(url) > 60
    )
