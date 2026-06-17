"""Info download desktop — versione e URL .exe Windows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

STATIC_DIR = Path(os.environ.get("ORGANISM_STATIC_DIR", Path(__file__).parent / "static"))
MANIFEST_PATH = STATIC_DIR / "releases" / "manifest.json"
DEFAULT_GITHUB = (
    "https://github.com/valeriozanzarelli-debug/mind-runtime/releases/latest/download/ORGANISM-Windows.exe"
)


def _read_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.is_file():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "version": "0.5.0",
        "product": "ORGANISM",
        "windows": {
            "filename": "ORGANISM-Windows.exe",
            "github_release": DEFAULT_GITHUB,
            "min_windows": "10",
        },
    }


def download_info(*, base_path: str = "") -> dict[str, Any]:
    manifest = _read_manifest()
    win = dict(manifest.get("windows") or {})
    filename = str(win.get("filename") or "ORGANISM-Windows.exe")
    local = STATIC_DIR / "releases" / filename
    bp = (base_path or "").rstrip("/")
    local_url = f"{bp}/static/releases/{filename}" if local.is_file() else ""
    github_url = str(win.get("github_release") or DEFAULT_GITHUB)
    size_mb = round(local.stat().st_size / (1024 * 1024), 1) if local.is_file() else None
    return {
        "product": manifest.get("product", "ORGANISM"),
        "version": manifest.get("version", "0.5.0"),
        "windows": {
            "filename": filename,
            "available": bool(local.is_file()),
            "url": local_url or github_url,
            "mirror_github": github_url,
            "local": bool(local.is_file()),
            "size_mb": size_mb,
            "min_windows": win.get("min_windows", "10"),
            "notes": win.get("notes", ""),
        },
    }
