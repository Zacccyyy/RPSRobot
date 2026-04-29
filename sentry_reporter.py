"""
sentry_reporter.py
==================
Crash reporting and player feedback via Sentry.io.

Why Sentry instead of Discord webhooks:
  - The DSN is a PUBLIC key, safe to hardcode in source code and commit to GitHub
  - Discord webhooks are PRIVATE tokens that get auto-revoked if found in public repos
  - Sentry gives a proper dashboard: crash grouping, stack traces, OS/version breakdown
  - Free tier: 5,000 events/month (more than enough for a capstone project)

What gets sent (only if player accepted the privacy notice):
  Crash reports:  exception type, stack trace, OS, Python version, app version
  Feedback:       player name, message text, app version
  Nothing else.   No camera data, no gameplay history, no location.

DSN is safe to be public - it is a receive-only ingest key tied to this project only.
"""

import sys

SENTRY_DSN = "https://e7d1fb8248783a0aed7cb52f3f602036@o4511305628975104.ingest.de.sentry.io/4511305637494864"

_initialised = False


def _init():
    """
    Initialise the Sentry SDK once.
    Safe to call multiple times - only runs on first call.
    Falls back silently if sentry_sdk is not installed.
    """
    global _initialised
    if _initialised:
        return True
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            # Don't send PII like IP addresses
            send_default_pii=False,
            # No performance tracing - crash/feedback only
            traces_sample_rate=0.0,
            # Tag every event with platform info automatically
            environment="production",
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
    Send a crash report to Sentry.
    Called from the except block in __main__ after a crash.

    exc:         the actual exception object (for full stack trace in Sentry)
    report_text: the formatted text report (attached as additional context)
    """
    if not _init():
        return
    try:
        import sentry_sdk
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("player", player_name)
            scope.set_tag("app_version", version[:7] if version else "unknown")
            scope.set_tag("platform", sys.platform)
            scope.set_context("crash_report", {"text": report_text[:2000]})
            sentry_sdk.capture_exception(exc)
        print("[Sentry] Crash report sent")
    except Exception as e:
        print(f"[Sentry] Failed to send crash report: {e}")


def send_feedback(player_name, text, version=""):
    """
    Send player feedback to Sentry as a capture_message event.
    Appears in Sentry under Issues > User Feedback.
    """
    if not _init():
        return
    try:
        import sentry_sdk
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("player", player_name)
            scope.set_tag("app_version", version[:7] if version else "unknown")
            scope.set_tag("event_type", "player_feedback")
            scope.set_user({"username": player_name})
            sentry_sdk.capture_message(
                f"[FEEDBACK] {player_name}: {text.strip()[:500]}",
                level="info",
            )
        print(f"[Sentry] Feedback sent for {player_name}")
    except Exception as e:
        print(f"[Sentry] Failed to send feedback: {e}")


def is_available():
    """Return True if sentry_sdk is installed and initialised."""
    return _init()
