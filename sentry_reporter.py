"""
sentry_reporter.py
==================
Crash reporting and player feedback via Sentry.io.

Why Sentry instead of Discord webhooks:
  - The DSN is a PUBLIC key, safe to hardcode and commit to GitHub.
  - Discord webhooks are PRIVATE tokens that get auto-revoked if pushed publicly.
  - Sentry provides a proper dashboard: crash grouping, stack traces, OS/version info.
  - Free tier: 5,000 events/month -- plenty for a capstone project.

What gets sent (only if the player accepted the privacy notice):
  Crash reports:  exception type, stack trace, OS, Python version, app version
  Feedback:       player name, message text, app version
  Nothing else.   No camera data, no gameplay history, no location.

The DSN below is safe to be public -- it is a receive-only ingest key
tied to this project only.
"""

import sys

# Public receive-only ingest key for this Sentry project.
SENTRY_DSN = "https://e7d1fb8248783a0aed7cb52f3f602036@o4511305628975104.ingest.de.sentry.io/4511305637494864"

# Tracks whether the SDK has been initialised this session.
# We only want to call sentry_sdk.init() once.
_initialised = False


def _init():
    """
    Initialise the Sentry SDK exactly once per session.

    Safe to call multiple times -- returns immediately on subsequent calls.
    Returns True if Sentry is ready to use, False if not (e.g. not installed).
    Falls back silently if sentry_sdk is not installed so the app still runs.
    """
    global _initialised

    # Already initialised this session -- nothing to do.
    if _initialised:
        return True

    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            send_default_pii=False,   # don't include IP addresses or similar PII
            traces_sample_rate=0.0,   # no performance tracing, just crashes + feedback
            environment="production", # tags every event so we can filter by env
        )
        _initialised = True
        return True
    except ImportError:
        print("[Sentry] sentry-sdk not installed - run: pip install sentry-sdk")
        return False
    except Exception as e:
        print(f"[Sentry] Init failed: {e}")
        return False


def send_crash_report(exc, report_text, player_name="unknown", version=""):
    """
    Send a crash report to Sentry after an unhandled exception.

    Typically called from the top-level except block in __main__.

    exc         -- the actual exception object (gives Sentry the full stack trace)
    report_text -- a formatted text summary (attached as extra context, capped at 2000 chars)
    player_name -- used as a Sentry tag so crashes can be filtered by player
    version     -- git SHA or similar, truncated to 7 chars (standard short-hash format)
    """
    # Bail out if the SDK failed to initialise.
    if not _init():
        return

    try:
        import sentry_sdk
        # new_scope() gives us a temporary scope so our tags don't bleed into future events.
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("player",      player_name)
            scope.set_tag("app_version", version[:7] if version else "unknown")
            scope.set_tag("platform",    sys.platform)
            scope.set_context("crash_report", {"text": report_text[:2000]})
            sentry_sdk.capture_exception(exc)
        print("[Sentry] Crash report sent")
    except Exception as e:
        print(f"[Sentry] Failed to send crash report: {e}")


def send_feedback(player_name, text, version=""):
    """
    Send player feedback to Sentry as a capture_message event.

    Appears in the Sentry dashboard under Issues.
    The message text is capped at 500 characters to stay within Sentry limits.
    """
    # Bail out if the SDK failed to initialise.
    if not _init():
        return

    try:
        import sentry_sdk
        # Use a fresh scope so these feedback tags don't affect other events.
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("player",      player_name)
            scope.set_tag("app_version", version[:7] if version else "unknown")
            scope.set_tag("event_type",  "player_feedback")
            scope.set_user({"username":  player_name})
            sentry_sdk.capture_message(
                f"[FEEDBACK] {player_name}: {text.strip()[:500]}",
                level="info",
            )
        print(f"[Sentry] Feedback sent for {player_name}")
    except Exception as e:
        print(f"[Sentry] Failed to send feedback: {e}")


def is_available():
    """
    Return True if sentry_sdk is installed and the SDK initialised successfully.

    The UI uses this to decide whether to offer the "send crash report" option.
    """
    return _init()
