"""
Pixel-art bird sprites + rank halos.

Birds are drawn from a tiny character grid (17 x 12 cells, the proportions of the original 34 x 24 sprite) and
scaled with nearest-neighbour only, so they stay square and crisp at any scale and under the pixel renderer. Colour is
a free parameter: every bird gets a hue spread over the rainbow (body, light top, dark underside) while outline, eye,
beak and wing stay constant, so many birds remain easy to tell apart. The bird faces right and has three wing frames
(up, middle, down). Collision never reads these pixels: it is analytic, see `kernels.physics`.
"""
from __future__ import annotations

import colorsys
import os

import pygame
from pygame import Surface

from .util import mix, shade

__all__ = ["rainbow", "make_bird", "make_halo", "export_bird_images", "HALO_COLORS", "BASE_SIZE", "ANIMATIONS"]


BASE_SIZE = (23, 16)        # sprite grid (w, h) in cells: an oval body plus the beak, ~ the 34 x 24 proportions
ANIMATIONS = (0, 1, 2, 1)   # wing frame order (up, middle, down, middle)

# gold / silver / bronze
HALO_COLORS = ((255, 214, 64), (205, 215, 230), (214, 132, 70))

_BODY_RX, _BODY_RY = 10.0, 8.0      # body ellipse (cells); it spans columns 0..19, the beak fills 19..22
_BODY_CX, _BODY_CY = 10.0, 8.0

# Wing (8 x 5): O outline, W feather, G feather shade
_WING = (
    "..OOOO..",
    ".OWWWWO.",
    "OWWWWWGO",
    ".OGGGGO.",
    "..OOOO..",
)
_WING_ROW = (3, 5, 7)       # top row of the wing for frames 0 / 1 / 2
_WING_COL = 2

OUTLINE = (84, 56, 71)


def rainbow(index: int, total: int, sat: float = 0.85, val: float = 0.95) -> tuple[int, int, int]:
    """Evenly spread hues over the rainbow (red -> violet, never wrapping back onto red)."""
    h = 0.82 * index / max(total - 1, 1) if total > 1 else 0.12
    r, g, b = colorsys.hsv_to_rgb(h, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


def _palette(color: tuple[int, int, int]) -> dict[str, tuple[int, int, int]]:
    return {
        "O": OUTLINE,
        "B": color,
        "L": shade(color, 1.28),
        "D": shade(color, 0.7),
        "C": mix((250, 236, 186), color, 0.12),
        "W": (255, 255, 255),
        "K": OUTLINE,
        "R": (244, 92, 30),
        "r": (206, 52, 24),
        "G": mix((228, 214, 176), color, 0.12),
    }


def _inside(x: int, y: int) -> bool:
    return ((x + 0.5 - _BODY_CX) / _BODY_RX) ** 2 + ((y + 0.5 - _BODY_CY) / _BODY_RY) ** 2 <= 1.0


def _body_cells() -> dict[tuple[int, int], str]:
    """Oval body (stepped ellipse), eye and beak as {(x, y): palette letter}. Built once."""
    cells: dict[tuple[int, int], str] = {}
    w, h = BASE_SIZE
    for y in range(h):
        for x in range(w):
            if not _inside(x, y):
                continue
            edge = not (_inside(x - 1, y) and _inside(x + 1, y) and _inside(x, y - 1) and _inside(x, y + 1))
            if edge:
                ch = "O"
            elif y <= 4:
                ch = "L"
            elif y <= 10:
                ch = "B"
            elif y <= 12:
                ch = "C"
            else:
                ch = "D"
            cells[(x, y)] = ch
    for y in range(3, 7):                                   # eye: white 4 x 4, pupil on its front side
        for x in range(12, 16):
            cells[(x, y)] = "K" if (x >= 14 and 4 <= y <= 5) else "W"
    beak = ("OOOO", "RRRO", "RRRO", "rrrO", "OOOO")         # cols 19..22, rows 7..11
    for dy, row in enumerate(beak):
        for dx, ch in enumerate(row):
            cells[(19 + dx, 7 + dy)] = ch
    return cells


_BODY_CACHE: dict[tuple[int, int], str] | None = None


def _grid(color: tuple[int, int, int], frame: int) -> Surface:
    """The bird at one pixel per cell."""
    global _BODY_CACHE
    if _BODY_CACHE is None:
        _BODY_CACHE = _body_cells()
    pal = _palette(color)
    s = pygame.Surface(BASE_SIZE, pygame.SRCALPHA)
    for (x, y), ch in _BODY_CACHE.items():
        s.set_at((x, y), pal[ch])
    pal["W"] = mix((255, 255, 255), (255, 244, 214), 0.5)           # feathers are cream, the eye stays white
    top = _WING_ROW[max(0, min(2, int(frame)))]
    for y, row in enumerate(_WING):
        for x, ch in enumerate(row):
            if ch != ".":
                s.set_at((_WING_COL + x, top + y), pal[ch])
    return s


def make_bird(
    color: tuple[int, int, int], frame: int = 1, size: tuple[float, float] = (68, 48),
    scale: float = 1.0, hard: bool = True,
) -> Surface:
    """Pixel-art bird facing right, scaled with nearest-neighbour to exactly `size` x `scale`, so it fills the same
    box the (elliptical) hitbox is inscribed in, with or without the pixel renderer."""
    spr = _grid(color, frame)
    target = (max(2, round(size[0] * scale)), max(2, round(size[1] * scale)))
    return pygame.transform.scale(spr, target)


def make_halo(color: tuple[int, int, int], radius: int, hard: bool = True) -> Surface:
    """Square-cornered ring (pre-rendered, blitted centred under the bird; never rotated)."""
    r = max(6, int(radius))
    size = 2 * r + 8
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    rect = pygame.Rect(4, 4, 2 * r, 2 * r)
    pygame.draw.rect(s, (*color, 60), rect)
    pygame.draw.rect(s, (*color, 255), rect, width=max(2, r // 6), border_radius=max(2, r // 4))
    return s


def export_bird_images(directory: str, colors: dict | None = None, scale: float = 4.0, overwrite: bool = False) -> list[str]:
    """
    Write static ``<n>-bird<frame>.png`` previews (default: red, green, blue, yellow) into ``directory`` when they
    are not there already. Returns the files that were written.
    """
    palette = colors or {"red": (225, 40, 40), "green": (40, 190, 70), "blue": (45, 95, 230), "yellow": (250, 210, 40)}
    written = []
    try:
        os.makedirs(directory, exist_ok=True)
        for name, color in palette.items():
            for frame in range(3):
                path = os.path.join(directory, f"{name}-bird{frame + 1}.png")
                if os.path.exists(path) and not overwrite:
                    continue
                pygame.image.save(make_bird(color, frame, scale=scale), path)
                written.append(path)
    except Exception:           # read-only install etc.: previews are optional
        pass
    return written