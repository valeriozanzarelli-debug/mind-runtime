"""Decoder per media dal browser — microfono, webcam."""

from __future__ import annotations

import base64
import hashlib
import struct
from typing import Any


def decode_image_gray(flat: list[int] | tuple[int, ...], width: int, height: int) -> list[list[int]]:
    """Griglia luminanza da array browser (getImageData)."""
    from organism.sensory.visual_scene import gray_to_grid

    return gray_to_grid(flat, width, height)


def decode_rgba_flat(
    flat: list[int] | tuple[int, ...],
    width: int,
    height: int,
) -> tuple[list[list[int]], list[list[tuple[int, int, int]]]]:
    """RGBA browser → griglia gray + griglia RGB (per V4 colore su blob)."""
    gray = decode_image_gray(flat, width, height)
    rgb: list[list[tuple[int, int, int]]] = []
    i = 0
    data = list(flat)
    for y in range(height):
        row: list[tuple[int, int, int]] = []
        for x in range(width):
            r = int(data[i]) if i < len(data) else 0
            g = int(data[i + 1]) if i + 1 < len(data) else r
            b = int(data[i + 2]) if i + 2 < len(data) else r
            row.append((r, g, b))
            i += 4
        rgb.append(row)
    return gray, rgb


def decode_jpeg_b64(
    b64: str,
    *,
    max_dim: int = 1920,
) -> tuple[list[list[int]], list[list[tuple[int, int, int]]], int, int]:
    """JPEG base64 (browser) → grigio + RGB fino a Full HD."""
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow richiesto per visione HD") from e

    raw = base64.b64decode(b64.split(",")[-1] if "," in b64 else b64)
    img = Image.open(BytesIO(raw)).convert("RGBA")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    gray: list[list[int]] = []
    rgb: list[list[tuple[int, int, int]]] = []
    i = 0
    for _y in range(h):
        grow: list[int] = []
        rrow: list[tuple[int, int, int]] = []
        for _x in range(w):
            r, g, b, _a = pixels[i]
            grow.append(int(0.299 * r + 0.587 * g + 0.114 * b))
            rrow.append((int(r), int(g), int(b)))
            i += 1
        gray.append(grow)
        rgb.append(rrow)
    return gray, rgb, w, h


def decode_image_pixels(b64: str, width: int, height: int) -> list[list[int]]:
    """RGBA base64 → grayscale grid."""
    raw = base64.b64decode(b64.split(",")[-1] if "," in b64 else b64)
    png_grid = _try_decode_png_grayscale(raw, width, height)
    if png_grid is not None:
        return png_grid
    expected = width * height * 4
    if len(raw) < expected:
        return _hash_grid(raw, width, height)
    grid: list[list[int]] = []
    i = 0
    for y in range(height):
        row = []
        for x in range(width):
            r = raw[i]
            g = raw[i + 1] if i + 1 < len(raw) else r
            b = raw[i + 2] if i + 2 < len(raw) else r
            row.append(int(0.299 * r + 0.587 * g + 0.114 * b))
            i += 4
        grid.append(row)
    return grid


def decode_audio_b64(b64: str) -> bytes:
    return base64.b64decode(b64.split(",")[-1] if "," in b64 else b64)


def vision_hash(grid: list[list[int]]) -> str:
    flat = bytes(max(0, min(255, c)) for row in grid for c in row[:32])
    return hashlib.sha256(flat).hexdigest()[:12]


def audio_hash(data: bytes) -> str:
    return hashlib.sha256(data[:4096]).hexdigest()[:12]


def _try_decode_png_grayscale(data: bytes, width: int, height: int) -> list[list[int]] | None:
    """Decodifica PNG RGB/RGBA minimale via Pillow se disponibile."""
    if not data.startswith(b"\x89PNG"):
        return None
    try:
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(data)).convert("L")
        img = img.resize((width, height))
        pixels = list(img.getdata())
        grid: list[list[int]] = []
        i = 0
        for _ in range(height):
            grid.append([int(pixels[i + x]) for x in range(width)])
            i += width
        return grid
    except Exception:
        return None


def _hash_grid(data: bytes, w: int, h: int) -> list[list[int]]:
    grid = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            idx = (y * w + x) % max(1, len(data))
            grid[y][x] = data[idx]
    return grid
