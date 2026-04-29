"""
privacy_notice.py
=================
Manages the player's consent preference for sending crash reports
and feedback to the developer via Discord webhook.

Consent is stored in config.json under "analytics_consent":
    None     -  not yet asked (show consent screen on next launch)
    True     -  player accepted (send crash reports + feedback)
    False    -  player declined (save locally only, never send)

The consent screen is shown ONCE on first launch, before the player
enters their name. It can be revisited and changed at any time from
Settings → Privacy.

What is collected (only if consent = True):
    - Crash reports: OS, Python version, git hash, error traceback
    - Feedback: player name, typed message, timestamp, git hash

What is NEVER collected:
    - Gameplay video or camera data
    - Round history or game statistics
    - Location data
    - Any data without explicit player action (feedback requires typing + Enter)
"""


def has_consent(config: dict) -> bool:
    """Return True only if player explicitly accepted."""
    return config.get("analytics_consent") is True


def has_declined(config: dict) -> bool:
    """Return True only if player explicitly declined."""
    return config.get("analytics_consent") is False


def needs_consent_prompt(config: dict) -> bool:
    """Return True if consent has never been asked."""
    return config.get("analytics_consent") is None


def set_consent(config: dict, accepted: bool) -> dict:
    """Record the player's consent decision. Returns updated config."""
    config["analytics_consent"] = accepted
    return config


def get_webhook_url(config: dict) -> str:
    """Return the Discord webhook URL, or empty string if not set."""
    return config.get("discord_webhook_url", "")


def set_webhook_url(config: dict, url: str) -> dict:
    """Store the Discord webhook URL. Returns updated config."""
    config["discord_webhook_url"] = url.strip()
    return config


def consent_summary(config: dict) -> str:
    """Human-readable one-line summary of current consent state."""
    if needs_consent_prompt(config):
        return "Not yet asked"
    if has_consent(config):
        return "Accepted  -  crash reports and feedback sent to developer"
    return "Declined  -  data saved locally only"
