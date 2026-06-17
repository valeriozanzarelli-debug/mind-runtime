"""Info download desktop — versione e URL .exe Windows."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

STATIC_DIR = Path(os.environ.get("ORGANISM_STATIC_DIR", Path(__file__).parent / "static"))
MANIFEST_PATH = STATIC_DIR / "releases" / "manifest.json"
DEFAULT_GITHUB = (
    "https://github.com/valeriozanzarelli-debug/mind-runtime/releases/download/windows-latest/ORGANISM-Windows.exe"
)


def _read_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.is_file():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "version": "0.8.0",
        "product": "ORGANISM",
        "windows": {
            "filename": "ORGANISM-Windows.exe",
            "github_release": DEFAULT_GITHUB,
            "min_windows": "10",
        },
    }


def _github_asset_ready(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status in (200, 302)
    except Exception:
        return False


def download_info(*, base_path: str = "") -> dict[str, Any]:
    manifest = _read_manifest()
    win = dict(manifest.get("windows") or {})
    filename = str(win.get("filename") or "ORGANISM-Windows.exe")
    local = STATIC_DIR / "releases" / filename
    bp = (base_path or "").rstrip("/")
    local_url = f"{bp}/static/releases/{filename}" if local.is_file() else ""
    github_url = str(win.get("github_release") or DEFAULT_GITHUB)
    size_mb = round(local.stat().st_size / (1024 * 1024), 1) if local.is_file() else None
    has_local = local.is_file()
    github_ready = _github_asset_ready(github_url) if not has_local else False
    download_url = local_url if has_local else (github_url if github_ready else "")
    return {
        "product": manifest.get("product", "ORGANISM"),
        "version": manifest.get("version", "0.8.0"),
        "windows": {
            "filename": filename,
            "available": has_local or github_ready,
            "url": download_url,
            "mirror_github": github_url,
            "github_ready": github_ready,
            "local": has_local,
            "size_mb": size_mb,
            "min_windows": win.get("min_windows", "10"),
            "notes": win.get("notes", ""),
            "build_url": "https://github.com/valeriozanzarelli-debug/mind-runtime/actions/workflows/build-windows.yml",
        },
    }
