"""PyInstaller runtime — tzdata su Windows .exe."""

import os
import sys

if getattr(sys, "frozen", False):
    base = getattr(sys, "_MEIPASS", "")
    for sub in ("tzdata/zoneinfo", "zoneinfo", "tzdata"):
        candidate = os.path.join(base, sub)
        if os.path.isdir(candidate):
            os.environ.setdefault("TZPATH", candidate)
            break
