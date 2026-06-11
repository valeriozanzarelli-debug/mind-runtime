"""Colori — HSV + RGB camera (regione saliente)."""

from __future__ import annotations

import math
from typing import Any


def rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn
    if diff < 1e-6:
        h = 0.0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    s = 0.0 if mx < 1e-6 else diff / mx
    v = mx
    return h, s, v


def rgb_to_color_name(r: float, g: float, b: float) -> str:
    """Nome colore italiano — HSV prima, poi luminanza."""
    r, g, b = float(r), float(g), float(b)
    h, s, v = rgb_to_hsv(r, g, b)
    v255 = v * 255

    if v255 < 45:
        return "nero"
    if v255 > 225 and s < 0.12:
        return "bianco"
    if s < 0.15:
        if v255 > 180:
            return "bianco"
        if v255 < 70:
            return "nero"
        return "grigio"

    if h < 15 or h >= 345:
        return "rosso" if v255 > 100 else "marrone"
    if h < 40:
        return "arancione"
    if h < 70:
        return "giallo"
    if h < 155:
        return "verde"
    if h < 200:
        return "azzurro" if v255 > 160 else "blu"
    if h < 260:
        return "blu"
    if h < 310:
        return "viola"
    if h < 345:
        return "rosa"
    return "rosso"


def dominant_rgb_from_grid(
    rgb_grid: list[list[tuple[int, int, int]]],
    *,
    gray_grid: list[list[int]] | None = None,
) -> tuple[float, float, float]:
    """RGB dominante dalla regione più saliente (centro + blob luminoso)."""
    h = len(rgb_grid)
    w = len(rgb_grid[0]) if h else 0
    if not w:
        return 0.0, 0.0, 0.0

    cx, cy = w / 2, h / 2
    r_sum = g_sum = b_sum = 0.0
    weight = 0.0

    if gray_grid and len(gray_grid) == h and len(gray_grid[0]) == w:
        flat = [gray_grid[y][x] for y in range(h) for x in range(w)]
        mean = sum(flat) / len(flat)
        threshold = mean + max(10.0, (max(flat) - min(flat)) * 0.2)
        for y in range(h):
            for x in range(w):
                lum = gray_grid[y][x]
                if lum < threshold:
                    continue
                r, g, b = rgb_grid[y][x]
                dist = math.hypot(x - cx, y - cy)
                wgt = 1.0 + max(0, lum - threshold) / 80.0
                wgt *= 1.5 if dist < min(w, h) * 0.35 else 1.0
                r_sum += r * wgt
                g_sum += g * wgt
                b_sum += b * wgt
                weight += wgt

    if weight < 1:
        y0, y1 = int(h * 0.25), int(h * 0.75)
        x0, x1 = int(w * 0.25), int(w * 0.75)
        for y in range(y0, y1):
            for x in range(x0, x1):
                r, g, b = rgb_grid[y][x]
                dist = math.hypot(x - cx, y - cy)
                wgt = 2.0 - min(1.0, dist / (min(w, h) * 0.5))
                r_sum += r * wgt
                g_sum += g * wgt
                b_sum += b * wgt
                weight += wgt

    if weight < 1e-6:
        return 0.0, 0.0, 0.0
    return r_sum / weight, g_sum / weight, b_sum / weight


def analyze_color_rgb(r: float, g: float, b: float) -> dict[str, Any]:
    name = rgb_to_color_name(r, g, b)
    h, s, v = rgb_to_hsv(r, g, b)
    return {
        "r": round(r, 1),
        "g": round(g, 1),
        "b": round(b, 1),
        "hue": round(h, 1),
        "saturation": round(s, 3),
        "value": round(v, 3),
        "color": name,
        "symbols": [f"COL:{name}"],
    }


def gray_to_lightness_color(mean: float) -> str:
    if mean > 200:
        return "bianco"
    if mean > 140:
        return "chiaro"
    if mean < 50:
        return "nero"
    if mean < 100:
        return "scuro"
    return "grigio"
