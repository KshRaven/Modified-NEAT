"""
Procedural car sprites + rank halos.

Cars are drawn (not loaded), so colour is a free parameter: every car gets a hue spread over the
rainbow and the top 3 are marked with a halo ring instead of a different colour.
Sprites face up (angle 0 = north), exactly like the old PNGs, and keep the same footprint
(BASE_SIZE * car_scale) so the physics masks / LiDAR geometry are unchanged.
"""
import colorsys
import pygame

from pygame import Surface

BASE_SIZE = (38, 76)        # footprint of the original PNGs (w, h) before car_scale
SUPERSAMPLE = 8             # drawn big, then smooth-scaled down for clean edges at tiny sizes

# gold / silver / bronze
HALO_COLORS = ((255, 214, 64), (205, 215, 230), (214, 132, 70))


def rainbow(index: int, total: int, sat: float = 0.85, val: float = 0.95) -> tuple[int, int, int]:
    """Evenly spread hues over the rainbow (red -> violet, never wrapping back onto red)."""
    h = 0.82 * index / max(total - 1, 1) if total > 1 else 0.0
    r, g, b = colorsys.hsv_to_rgb(h, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


def _shade(c: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(v * f))) for v in c)


WHEEL = (7, 12)                 # base-pixel size of one wheel
WHEEL_CENTERS = {"fl": (3.5, 20.0), "fr": (34.5, 20.0), "rl": (3.5, 58.0), "rr": (34.5, 58.0)}
WHEEL_PAD = 5                   # extra base pixels each side so turned wheels are never clipped (render only)


def _wheel_surface(u: int) -> Surface:
    w = pygame.Surface((WHEEL[0] * u, WHEEL[1] * u), pygame.SRCALPHA)
    pygame.draw.rect(w, (22, 22, 24), w.get_rect(), border_radius=2 * u)
    pygame.draw.rect(w, (58, 58, 64), (2 * u, 1 * u, 3 * u, 2 * u))        # tread marks: make the turn readable
    pygame.draw.rect(w, (58, 58, 64), (2 * u, 5 * u, 3 * u, 2 * u))
    pygame.draw.rect(w, (58, 58, 64), (2 * u, 9 * u, 3 * u, 2 * u))
    return w


def make_car(color: tuple[int, int, int], scale: float = 0.2, steer: float = 0.0, pad: int = 0) -> Surface:
    """
    Top-down sports car with a rear wing: wheels, body, stripe, cockpit, lights, spoiler.

    steer : front-wheel angle in degrees (+ = wheels point left / counter-clockwise, like the car's own angle).
            Rear wheels never turn.
    pad   : extra base pixels left/right of the car so turned wheels fit (the footprint used for physics
            masks is the pad=0 sprite).
    """
    S = SUPERSAMPLE
    P = int(pad)
    W, H = (BASE_SIZE[0] + 2 * P) * S, BASE_SIZE[1] * S
    s = pygame.Surface((W, H), pygame.SRCALPHA)
    u = S                                   # 1 base pixel
    ox = P * u                              # x offset of the car inside the padded canvas
    dark, light = _shade(color, 0.55), _shade(color, 1.25)
    d = pygame.draw

    def X(v):
        return v * u + ox

    # wheels first (they sit under the body); front pair is rotated by the steering angle
    wheel = _wheel_surface(u)
    for name, (cx, cy) in WHEEL_CENTERS.items():
        img = pygame.transform.rotate(wheel, steer) if (steer and name[0] == "f") else wheel
        s.blit(img, img.get_rect(center=(int(X(cx)), int(cy * u))))

    # body: tapered nose, wide mid, narrow tail
    body = [(19, 1), (25, 6), (29, 18), (30, 34), (30, 56), (27, 66), (11, 66), (8, 56), (8, 34), (9, 18), (13, 6)]
    d.polygon(s, color, [(X(x), y * u) for x, y in body])
    d.polygon(s, dark, [(X(x), y * u) for x, y in body], width=u)

    # centre racing stripe + nose highlight
    d.rect(s, light, (X(17), 2 * u, 4 * u, 24 * u), border_radius=u)
    d.rect(s, light, (X(17), 44 * u, 4 * u, 20 * u), border_radius=u)

    # cockpit / windscreen
    d.polygon(s, (20, 24, 34), [(X(p[0]), p[1] * u) for p in ((14, 27), (24, 27), (26, 36), (12, 36))])
    d.rect(s, (20, 24, 34), (X(12), 36 * u, 14 * u, 8 * u))
    d.polygon(s, (90, 120, 160), [(X(p[0]), p[1] * u) for p in ((15, 28), (23, 28), (24, 32), (14, 32))])

    # headlights + tail lights
    d.ellipse(s, (255, 250, 210), (X(11), 8 * u, 5 * u, 4 * u))
    d.ellipse(s, (255, 250, 210), (X(22), 8 * u, 5 * u, 4 * u))
    d.rect(s, (255, 40, 40), (X(10), 62 * u, 5 * u, 2 * u))
    d.rect(s, (255, 40, 40), (X(23), 62 * u, 5 * u, 2 * u))

    # rear spoiler: struts, full-width wing, end plates
    d.rect(s, (30, 30, 34), (X(13), 66 * u, 3 * u, 3 * u))
    d.rect(s, (30, 30, 34), (X(22), 66 * u, 3 * u, 3 * u))
    d.rect(s, dark, (X(1), 68 * u, 36 * u, 6 * u), border_radius=u)
    d.rect(s, light, (X(2), 68 * u, 34 * u, 2 * u))
    d.rect(s, (25, 25, 28), (X(0), 66 * u, 2 * u, 10 * u))
    d.rect(s, (25, 25, 28), (X(36), 66 * u, 2 * u, 10 * u))

    size = (max(2, round((BASE_SIZE[0] + 2 * P) * scale)), max(4, round(BASE_SIZE[1] * scale)))
    return pygame.transform.smoothscale(s, size)


CAR_COLORS = {
    "red": (225, 40, 40), "green": (40, 190, 70), "blue": (45, 95, 230),
    "black": (28, 28, 32), "white": (240, 240, 244),
}


def export_car_images(directory: str, colors: dict | None = None, scale: float = 4.0, overwrite: bool = False) -> list[str]:
    """
    Write static ``<name>-car.png`` previews (default: red, green, blue, black, white) into ``directory``
    when they are not there already. Drawn large (``scale`` x the 38x76 footprint) for detail viewing.
    Returns the files that were written.
    """
    import os
    written = []
    try:
        os.makedirs(directory, exist_ok=True)
        for name, color in (colors or CAR_COLORS).items():
            path = os.path.join(directory, f"{name}-car.png")
            if os.path.exists(path) and not overwrite:
                continue
            pygame.image.save(make_car(color, scale), path)
            written.append(path)
    except Exception:           # read-only install etc.: previews are optional
        pass
    return written


def make_halo(color: tuple[int, int, int], radius: int) -> Surface:
    """Soft glowing ring (pre-rendered, blitted centred under the car; never rotated)."""
    r = max(6, int(radius))
    size = 2 * r + 8
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c = (size // 2, size // 2)
    for i in range(5):                                         # outer glow, fading outwards
        pygame.draw.circle(s, (*color, 22 + 14 * (4 - i)), c, r + 3 - i + 1, width=2)
    pygame.draw.circle(s, (*color, 235), c, r, width=2)         # crisp ring
    pygame.draw.circle(s, (*color, 45), c, r - 2)               # faint fill so the car pops
    return s