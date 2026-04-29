"""
ui_renderer.py -- Compatibility shim.

The renderer has been split into four focused modules:
  ui_base.py   -- colours, layout helpers, drawing primitives
  ui_game.py   -- in-game screens (arcade view, result, diagnostic)
  ui_modes.py  -- per-mode screens (2P, reflex, bluff, simon, squid, rpsls)
  ui_menus.py  -- menu/settings/features/stats/tutorial screens

This file re-exports everything so that existing code importing from
ui_renderer continues to work without any changes.
"""

from ui_base import *   # noqa: F401, F403
from ui_game import *   # noqa: F401, F403
from ui_modes import *  # noqa: F401, F403
from ui_menus import *  # noqa: F401, F403
