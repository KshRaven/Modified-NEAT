"""
Pixel-art scenery: cloud band, city skyline, trees + bushes, floating clouds and the floor tile.

Everything is drawn on a design grid where one cell is `CELL` world pixels (the same cell size as the bird sprite)
using rectangles only, then scaled with nearest-neighbour by the renderer. Shapes are stepped and square-like on
purpose (like the original images), so they stay crisp with or without the pixel renderer. Layers are horizontally
tileable. Purely visual: a private `random.Random` is used, the simulation and its seeds are never touched.
"""
from __future__ import annotations

import math
import random

import pygame
from pygame import Surface

from .constants import (
    COLOR_CLOUD, COLOR_FLOOR, COLOR_FLOOR_ALT, COLOR_FLOOR_EDGE, COLOR_PIPE, COLOR_PIPE_DARK, COLOR_PIPE_LIGHT,
)
from .util import mix, shade

__all__ = ["CELL", "LAYERS", "make_layer", "make_cloud", "make_floor_tile", "make_pipe_strip", "make_pipe_cap"]

CELL = 4                        # world pixels per design cell (the bird sprite is 17 x 12 cells of this size)

# name: (tile width in cells, tile height in cells, scroll factor)
LAYERS = {
    "band": (56, 14, 0.08),
    "city": (96, 26, 0.14),
    "green": (72, 18, 0.30),
}

_BAND = (226, 243, 214)
_CITY_WALL, _CITY_EDGE, _CITY_WIN, _CITY_ROOF = (205, 232, 216), (146, 198, 196), (166, 214, 228), (176, 212, 206)
_BUSH_DARK, _BUSH_MID, _BUSH_LIGHT = (50, 138, 66), (86, 186, 90), (146, 228, 122)
_PINE_DARK, _PINE_MID, _PINE_LIGHT = (36, 108, 62), (58, 148, 78), (98, 186, 104)
_TRUNK, _TRUNK_DARK = (130, 90, 58), (88, 58, 44)


def _q(color, bits: int):
    if not 1 <= bits < 8:
        return tuple(int(v) for v in color[:3])
    sh = 8 - bits
    half = (1 << sh) >> 1
    return tuple(min(255, ((int(v) >> sh) << sh) | half) for v in color[:3])


def _fill(s: Surface, color, x: int, y: int, w: int, h: int, wrap: int = 0) -> None:
    """Fill a cell rectangle; with `wrap` it is repeated one tile left / right so layers tile seamlessly."""
    if w <= 0 or h <= 0:
        return
    for off in ((0,) if not wrap else (-wrap, 0, wrap)):
        s.fill(color, (x + off, y, w, h))


def _dome(s: Surface, color, cx: int, by: int, hw: int, h: int, wrap: int = 0) -> None:
    """Stepped half-ellipse standing on row `by` (exclusive), `hw` cells wide to each side, `h` rows high."""
    for j in range(h):
        half = max(1, round(hw * math.sqrt(max(0.0, 1.0 - ((h - 1 - j) / h) ** 2))))
        _fill(s, color, cx - half, by - h + j, 2 * half + 1, 1, wrap)


def _pine(s: Surface, cx: int, by: int, wrap: int, c: dict) -> None:
    _fill(s, c["trunk_dark"], cx - 1, by - 3, 3, 3, wrap)
    _fill(s, c["trunk"], cx - 1, by - 3, 2, 3, wrap)
    widths = (3, 3, 5, 5, 7, 7, 9, 9)
    top = by - 3 - len(widths)
    for r, w in enumerate(widths):
        _fill(s, c["pine_dark"], cx - w // 2 - 1, top + r, w + 2, 1, wrap)
    _fill(s, c["pine_dark"], cx - 1, top - 1, 3, 1, wrap)
    for r, w in enumerate(widths):
        _fill(s, c["pine_mid"], cx - w // 2, top + r, w, 1, wrap)
        _fill(s, c["pine_light"], cx - w // 2, top + r, max(1, w // 3), 1, wrap)


def _round_tree(s: Surface, cx: int, by: int, wrap: int, c: dict) -> None:
    _fill(s, c["trunk_dark"], cx - 1, by - 5, 3, 5, wrap)
    _fill(s, c["trunk"], cx - 1, by - 5, 2, 5, wrap)
    _dome(s, c["bush_dark"], cx, by - 3, 7, 9, wrap)
    _dome(s, c["bush_mid"], cx, by - 3, 6, 8, wrap)
    _dome(s, c["bush_light"], cx - 2, by - 5, 3, 4, wrap)


def _colors(bits: int) -> dict:
    return {k: _q(v, bits) for k, v in {
        "band": _BAND, "cloud": COLOR_CLOUD, "cloud_shade": mix(COLOR_CLOUD, (150, 205, 215), 0.35),
        "wall": _CITY_WALL, "edge": _CITY_EDGE, "win": _CITY_WIN, "roof": _CITY_ROOF,
        "bush_dark": _BUSH_DARK, "bush_mid": _BUSH_MID, "bush_light": _BUSH_LIGHT,
        "pine_dark": _PINE_DARK, "pine_mid": _PINE_MID, "pine_light": _PINE_LIGHT,
        "trunk": _TRUNK, "trunk_dark": _TRUNK_DARK,
    }.items()}


def make_layer(name: str, bits: int = 0, seed: int = 7) -> Surface:
    """One tileable layer at one pixel per cell (transparent where nothing is drawn)."""
    cw, ch, _ = LAYERS[name]
    rng = random.Random(seed + sum(map(ord, name)))
    c = _colors(bits)
    s = pygame.Surface((cw, ch), pygame.SRCALPHA)

    if name == "band":                      # cream cloud bumps along the horizon, solid below
        _fill(s, c["band"], 0, 5, cw, ch - 5)
        x = 3
        while x < cw + 8:
            hw, h = rng.randint(4, 7), rng.randint(4, 6)
            _dome(s, c["band"], x, 6, hw, h, cw)
            x += rng.randint(7, 11)

    elif name == "city":                    # rectangular towers with window grids, some with a rooftop block
        x = 0
        while x < cw:
            w = rng.randint(4, 8)
            if cw - x - w < 4:
                w = cw - x
            h = rng.randint(9, ch - 4)
            top = ch - h
            _fill(s, c["edge"], x, top, w, h)
            _fill(s, c["wall"], x, top + 1, w - 1, h - 1)
            for wy in range(top + 3, ch - 2, 3):
                for wx in range(x + 1, x + w - 2, 2):
                    _fill(s, c["win"], wx, wy, 1, 2)
            if w >= 6 and rng.random() < 0.5:
                _fill(s, c["edge"], x + w // 2 - 1, top - 2, 3, 2)
                _fill(s, c["roof"], x + w // 2 - 1, top - 2, 2, 2)
            x += w + rng.choice((0, 1))

    elif name == "green":                   # trees behind a row of bushes
        by = ch
        for cx, kind in ((9, "pine"), (33, "round"), (55, "pine")):
            (_pine if kind == "pine" else _round_tree)(s, cx, by - 3, cw, c)
        _fill(s, c["bush_dark"], 0, ch - 4, cw, 4)
        x = 2
        while x < cw + 6:
            hw, h = rng.randint(4, 6), rng.randint(5, 8)
            _dome(s, c["bush_dark"], x, ch, hw + 1, h + 1, cw)
            _dome(s, c["bush_mid"], x, ch, hw, h, cw)
            _dome(s, c["bush_light"], x - 1, ch - 2, max(1, hw - 2), max(1, h - 3), cw)
            x += rng.randint(7, 10)
        _fill(s, c["bush_mid"], 0, ch - 3, cw, 3)
    return s


# (width, [(dome centre x, half width, height), ...]) of the floating clouds, in cells
_CLOUDS = (
    (30, ((9, 6, 7), (17, 8, 10), (24, 5, 6))),
    (24, ((7, 5, 6), (15, 7, 9))),
)


def make_cloud(kind: int, bits: int = 0) -> Surface:
    """A small floating cloud at one pixel per cell: stepped bumps on a flat base with a shaded underside."""
    c = _colors(bits)
    width, domes = _CLOUDS[kind % len(_CLOUDS)]
    height = max(h for _, _, h in domes) + 4
    s = pygame.Surface((width, height), pygame.SRCALPHA)
    base = height - 2
    _fill(s, c["cloud"], 2, base - 3, width - 4, 3)
    for cx, hw, h in domes:
        _dome(s, c["cloud"], cx, base, hw, h)
    _fill(s, c["cloud_shade"], 2, base, width - 4, 1)
    _fill(s, c["cloud_shade"], 4, base + 1, width - 8, 1)
    return s


def make_floor_tile(cells_w: int, cells_h: int, bits: int = 0) -> Surface:
    """Floor tile (`cells_w` must be a multiple of 6 for the diagonal stripes to repeat): grass edge, sandy stripes."""
    sand, alt, edge = _q(COLOR_FLOOR, bits), _q(COLOR_FLOOR_ALT, bits), _q(COLOR_FLOOR_EDGE, bits)
    s = pygame.Surface((cells_w, cells_h))
    s.fill(sand)
    for y in range(4, cells_h):
        for x in range(cells_w):
            if (x + y) % 6 < 2:
                s.set_at((x, y), alt)
    s.fill(edge, (0, 0, cells_w, 2))
    s.fill(_q(shade(COLOR_FLOOR_EDGE, 0.6), bits), (0, 2, cells_w, 1))
    s.fill(_q(shade(COLOR_FLOOR_EDGE, 1.25), bits), (0, 0, cells_w, 1))
    s.fill(_q(shade(COLOR_FLOOR, 0.82), bits), (0, 3, cells_w, 1))
    return s


def _pipe_col(i: int, n: int, c: dict) -> tuple:
    """Colour of cell column `i` of `n` across a pipe: outline, light stripe, body, shade stripe, outline."""
    if i == 0 or i == n - 1:
        return c["dark"]
    if 2 <= i < 2 + max(1, n // 10):
        return c["light"]
    if i >= n - 1 - max(2, n // 7):
        return c["shade"]
    return c["body"]


def _pipe_colors(bits: int) -> dict:
    return {
        "body": _q(COLOR_PIPE, bits), "dark": _q(COLOR_PIPE_DARK, bits), "light": _q(COLOR_PIPE_LIGHT, bits),
        "shade": _q(mix(COLOR_PIPE, COLOR_PIPE_DARK, 0.55), bits),
    }


def make_pipe_strip(cells_w: int, bits: int = 0) -> Surface:
    """One row of a pipe shaft (`cells_w` cells wide); the renderer stretches it to the shaft height."""
    c = _pipe_colors(bits)
    s = pygame.Surface((cells_w, 1))
    for i in range(cells_w):
        s.set_at((i, 0), _pipe_col(i, cells_w, c))
    return s


def make_pipe_cap(cells_w: int, cells_h: int, bits: int = 0) -> Surface:
    """Pipe cap: outlined block with the same stripes plus a light top row and a shaded bottom row."""
    c = _pipe_colors(bits)
    s = pygame.Surface((cells_w, cells_h))
    for x in range(cells_w):
        col = _pipe_col(x, cells_w, c)
        for y in range(cells_h):
            s.set_at((x, y), col)
    for x in range(1, cells_w - 1):
        s.set_at((x, 1), c["light"])
        s.set_at((x, cells_h - 2), c["shade"])
        s.set_at((x, 0), c["dark"])
        s.set_at((x, cells_h - 1), c["dark"])
    return s