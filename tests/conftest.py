from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_PARENT = Path(__file__).resolve().parents[1] / "plugins" / "memory"
sys.path.insert(0, str(PLUGIN_PARENT))
