"""
capstone_paths.py
=================
Single source of truth for the CapStone data directory path.

Every other module imports CAPSTONE_DIR from here instead of
hardcoding the path. If the location ever changes, we only edit
this one file and everything else picks it up automatically.

Platform rules:
  macOS:   ~/Desktop/CapStone/   -- existing users already have data here
  Windows: ~/CapStone/           -- Desktop is cluttered enough on Windows
  Linux:   ~/CapStone/

Usage:
    from capstone_paths import CAPSTONE_DIR
"""

import sys
import pathlib


def _get_capstone_dir() -> pathlib.Path:
    """
    Work out where the CapStone data folder should live on this machine.

    macOS gets the Desktop so that existing users don't lose their data.
    Windows and Linux get the home folder to avoid cluttering the Desktop.
    """
    if sys.platform == "darwin":
        # macOS: keep on Desktop where existing users already have their files
        return pathlib.Path.home() / "Desktop" / "CapStone"
    else:
        # Windows / Linux: home folder, not Desktop
        return pathlib.Path.home() / "CapStone"


# Module-level constant -- import this everywhere you need the data directory.
CAPSTONE_DIR = _get_capstone_dir()
