"""
capstone_paths.py
=================
Single source of truth for the CapStone data directory path.

macOS:   ~/Desktop/CapStone/   (existing users keep their data)
Windows: ~/CapStone/           (Desktop is cluttered enough on Windows)
Linux:   ~/CapStone/

Import this instead of hardcoding the path in each file:
    from capstone_paths import CAPSTONE_DIR
"""

import sys
import pathlib

def _get_capstone_dir() -> pathlib.Path:
    if sys.platform == "darwin":
        # macOS: keep on Desktop where existing users already have data
        return pathlib.Path.home() / "Desktop" / "CapStone"
    else:
        # Windows / Linux: home folder, not Desktop
        return pathlib.Path.home() / "CapStone"

CAPSTONE_DIR = _get_capstone_dir()
