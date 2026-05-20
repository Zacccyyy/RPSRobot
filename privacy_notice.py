"""
privacy_notice.py
=================
Manages the player's consent preference for sending crash reports
and feedback to the developer via Sentry.

Consent is stored in config.json under "analytics_consent":
    None   -- not yet asked  (show consent screen on next launch)
    True   -- player accepted  (send crash reports + feedback)
    False  -- player declined  (save locally only, never send)

The consent screen is shown ONCE on first launch, before the player
enters their name. It can be revisited and changed at any time from
Settings -> Privacy.

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
    """Return True only if the player explicitly accepted data collection."""
    return config.get("analytics_consent") is True


def has_declined(config: dict) -> bool:
    """Return True only if the player explicitly declined data collection."""
    return config.get("analytics_consent") is False


def needs_consent_prompt(config: dict) -> bool:
    """
    Return True if the player has never been asked about consent.

    A None value means the consent dialog has not been shown yet -- the app
    should display it before doing anything else on first launch.
    """
    return config.get("analytics_consent") is None


def set_consent(config: dict, accepted: bool) -> dict:
    """
    Record the player's consent decision in the config dict.

    Pass accepted=True if the player clicked "Accept", False for "Decline".
    Returns the updated config dict (the caller is responsible for saving it).
    """
    config["analytics_consent"] = accepted
    return config


def consent_summary(config: dict) -> str:
    """
    Return a human-readable one-line summary of the current consent state.
    Used in the Settings -> Privacy screen.
    """
    # Check from most to least specific so each case is mutually exclusive.
    if needs_consent_prompt(config):
        return "Not yet asked"
    if has_consent(config):
        return "Accepted  -  crash reports and feedback sent to developer"
    return "Declined  -  data saved locally only"
