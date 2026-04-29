# RPS Robot — Privacy Notice
**TrickWing Toys / RavensAgency**
*Last updated: April 2026*

---

## Overview

RPS Robot is a local application. It runs entirely on your device and does
not require an account, login, or internet connection to play.

You are asked once, on first launch, whether you consent to sending crash
reports and feedback to the developer. You can change this decision at any
time in **Settings → Privacy Settings**.

---

## What is collected — and only if you consent

### Crash Reports
Sent automatically if the app crashes unexpectedly.

| Data | Example | Why |
|---|---|---|
| Timestamp | `2026-04-29 22:15:30` | To identify when the crash occurred |
| App version | `abc1234` (git hash) | To know which version caused the crash |
| Operating system | `macOS 14.2 arm64` | To reproduce platform-specific bugs |
| Python version | `3.9.7` | To identify runtime environment issues |
| Error message | `AttributeError: ...` | The actual error |
| Stack trace | Function call chain | To identify exactly where the crash happened |

### Player Feedback
Sent only when you open the Notes screen (`N` key from menu) and press
**Enter** to explicitly submit a message.

| Data | Example | Why |
|---|---|---|
| Player name | `Zac` | To attribute the feedback |
| Message | Whatever you typed | The suggestion or report itself |
| Timestamp | `2026-04-29 22:15:30` | To track when feedback was submitted |
| App version | `abc1234` | To know which version the feedback refers to |

---

## What is NEVER collected

- **Camera or video data** — the camera feed never leaves your device
- **Gameplay data** — round history, scores, and statistics stay local
- **Location data** — the app does not access your location
- **Hand scan / biometric data** — stored locally only, never transmitted
- **Background tracking** — nothing is collected automatically except crash reports

---

## Where data goes

All data is sent to a **private Discord channel** accessible only to the
developer (TrickWing Toys / RavensAgency). It is not shared with third
parties, sold, or used for advertising.

Discord's own privacy policy applies to data transmitted through their
service: [discord.com/privacy](https://discord.com/privacy)

---

## Local storage

Regardless of your consent choice, the following is always saved locally
to `~/Desktop/CapStone/` on your device:

- Player statistics and game history (`profiles/`)
- Hand scan biometric profiles (`fingerprints/`)
- Research data (`*.xlsx`)
- Crash reports (`crash_reports/`)
- Feedback submissions (`feedback/`)

This data never leaves your device unless you manually share it.

---

## Your rights (Australian Privacy Act 1988)

Under the Australian Privacy Principles, you have the right to:

- **Access** the data held about you — check `~/Desktop/CapStone/`
- **Correct** inaccurate data — edit or delete files directly
- **Withdraw consent** — Settings → Privacy Settings → No Thanks
- **Request deletion** — delete the `~/Desktop/CapStone/` folder at any time

For any privacy concerns, contact: **[your contact email here]**

---

## Changes to this notice

This privacy notice may be updated as the app develops. The current version
is always available at:
`https://github.com/Zacccyyy/RPSRobot/blob/main/PRIVACY.md`

---

*RPS Robot — TrickWing Toys | RavensAgency*
