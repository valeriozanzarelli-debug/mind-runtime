"""Fuso orario ORGANISM — funziona anche su Windows .exe senza tzdata di sistema."""

from __future__ import annotations

from datetime import timedelta, timezone, tzinfo

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def organism_timezone() -> tzinfo:
    """Europe/Madrid se disponibile, altrimenti UTC+1 (Italia/Spagna)."""
    if ZoneInfo is not None:
        try:
            return ZoneInfo("Europe/Madrid")
        except Exception:
            pass
    return timezone(timedelta(hours=1))


TZ = organism_timezone()
