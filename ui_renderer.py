"""
ui_renderer.py
==============
Compatibility shim that re-exports everything from the four focused UI modules.

The renderer was originally one large file. It was split into:
  ui_base.py   -- colours, layout helpers, drawing primitives
  ui_game.py   -- in-game screens (arcade view, result, diagnostic)
  ui_modes.py  -- per-mode screens (2P, reflex, bluff, simon, squid, rpsls)
  ui_menus.py  -- menu/settings/features/stats/tutorial screens

This file exists so that any existing code that does
    from ui_renderer import something
continues to work without any changes. All four modules are star-imported
here, which re-exports every public name they define.
"""

# Re-export everything from each UI sub-module.
# noqa comments suppress "unused import" and "star import" linter warnings
# because these imports are intentionally public re-exports, not local use.
from ui_base  import *  # noqa: F401, F403
from ui_game  import *  # noqa: F401, F403
from ui_modes import *  # noqa: F401, F403
from ui_menus import *  # noqa: F401, F403
