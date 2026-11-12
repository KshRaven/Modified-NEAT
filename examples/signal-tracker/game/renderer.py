"""Rendering of the Signal Tracker: pygame drawing (scrolling time plot + live frame panel) and a text fallback."""
from __future__ import annotations

import numpy as np

COL_BG       = (18, 20, 28)
COL_PANEL    = (28, 31, 42)
COL_GRID     = (52, 57, 74)
COL_TEXT     = (225, 228, 238)
COL_DIM      = (130, 136, 156)
COL_TARGET   = (80, 220, 120)
COL_CLEAN    = (80, 220, 120)
COL_DECOY    = (235, 80, 80)
COL_BEST     = (255, 255, 255)
COL_MEAN     = (110, 170, 255)
COL_WORST    = (255, 170, 60)
COL_CURSOR   = (110, 170, 255)
COL_HUMAN    = (255, 225, 70)
COL_HOVER    = (255, 255, 255)
GROUP_COLS   = [(255, 170, 60), (110, 170, 255), (255, 255, 255), (175, 140, 255), (255, 120, 200), (120, 230, 230)]


# live-perturbation sliders: (stream parameter, label, min, max); two columns of four in the control panel
SLIDERS = [
    ("decoy_frac", "decoy frac", 0.0, 0.95), ("decoy_run", "decoy run", 1.0, 40.0),
    ("cue_sigma", "cue noise", 0.0, 0.5), ("tag_sep", "tag sep", 0.0, 4.0),
    ("tag_sigma", "tag noise", 0.0, 1.0), ("drift", "drift", 0.0, 1.0),
    ("obs_noise", "obs noise", 0.0, 0.5), ("target_speed", "target spd", 0.0, 0.15),
]


def layout(w: int, h: int):
    """(plot, side, strip, ctrl) rectangles for a window of size (w, h); the Window maps the mouse with them."""
    import pygame
    pad = 14
    plot = pygame.Rect(pad, 54, int(w * 0.64) - pad, h - 54 - 150)
    side = pygame.Rect(plot.right + pad, 54, w - plot.right - 2 * pad, h - 54 - 150)
    bottom = pygame.Rect(pad, h - 134, w - 2 * pad, 120)
    strip = pygame.Rect(bottom.left, bottom.top, int(bottom.width * 0.55), bottom.height)
    ctrl = pygame.Rect(strip.right + pad, bottom.top, bottom.right - strip.right - pad, bottom.height)
    return plot, side, strip, ctrl


def slider_rects(ctrl) -> list[dict]:
    """Hit / draw geometry of every slider inside the control panel."""
    import pygame
    out, colw, rowh = [], (ctrl.width - 30) // 2, 22
    for i, (name, label, lo, hi) in enumerate(SLIDERS):
        c, r = divmod(i, 4)
        x0, y0 = ctrl.left + 10 + c * (colw + 10), ctrl.top + 30 + r * rowh
        out.append(dict(name=name, label=label, lo=lo, hi=hi, row=pygame.Rect(x0, y0, colw, rowh - 2),
                        track=pygame.Rect(x0 + 84, y0 + rowh // 2 - 4, max(colw - 84 - 46, 10), 8)))
    return out


def _draw_controls(screen, fonts: dict, ctrl, info: dict) -> None:
    import pygame
    small = fonts['small']
    ui, params = info.get('ui') or {}, info.get('params') or {}
    pygame.draw.rect(screen, COL_PANEL, ctrl, border_radius=8)
    screen.blit(small.render("drag sliders | click plot: kick target | right-click: decoy burst | R: reset", True, COL_DIM),
                (ctrl.left + 10, ctrl.top + 6))
    for s in slider_rects(ctrl):
        name, lo, hi, tr = s['name'], s['lo'], s['hi'], s['track']
        v = float(params.get(name, lo))
        frac = min(max((v - lo) / max(hi - lo, 1e-9), 0.0), 1.0)
        active, hover = ui.get('drag') == name, ui.get('hover_slider') == name
        lit = active or hover
        screen.blit(small.render(s['label'], True, COL_TEXT if lit else COL_DIM), (s['row'].left, s['row'].top + 2))
        pygame.draw.rect(screen, COL_BG, tr, border_radius=4)
        pygame.draw.rect(screen, COL_HUMAN if active else COL_CLEAN, (tr.left, tr.top, max(int(frac * tr.width), 1), tr.height), border_radius=4)
        pygame.draw.circle(screen, COL_HOVER if lit else COL_CURSOR, (tr.left + int(frac * tr.width), tr.centery), 7 if lit else 5)
        screen.blit(small.render(f"{v:.2f}", True, COL_TEXT if lit else COL_DIM), (tr.right + 10, s['row'].top + 2))
    pygame.draw.rect(screen, COL_HUMAN if ui.get('drag') else COL_GRID, ctrl, 2 if ui.get('drag') else 1, border_radius=8)


def render_line(x: np.ndarray, target: float, kind: int, width: int = 61) -> str:
    """Text fallback (render_mode='ansi')."""
    cells = [" "] * width

    def put(v, ch):
        cells[int(round((np.clip(v, -1, 1) + 1) / 2 * (width - 1)))] = ch

    err = np.abs(x - target)
    put(x[int(np.argmax(err))], "W")
    put(x[int(np.argmin(err))], "B")
    put(target, "T")
    return f"|{''.join(cells)}| {'DECOY' if kind else 'clean'}"


def _y(v: float, top: int, height: int) -> int:
    """Position in [-1, 1] -> pixel row (+1 at the top)."""
    return int(top + (1.0 - (float(np.clip(v, -1.5, 1.5)) + 1.0) / 2.0) * height)


def draw(screen, fonts: dict, trace: dict, agents_x: np.ndarray, state_row: np.ndarray | None, history: np.ndarray | None,
         info: dict) -> None:
    """
    Left: time plot of the last ``len(trace)`` steps. Target = green line; cues = dots (green clean, red decoy); best
    cursor = white, mean cursor = blue, worst = orange. Right: the live frame (cursor / cue / tag), the cursor positions
    of every genome on the track, and the strip of frames the model sees (``history``).
    """
    import pygame
    w, h = screen.get_size()
    screen.fill(COL_BG)
    small, mid = fonts['small'], fonts['mid']
    pad = 14
    plot, side, strip, ctrl = layout(w, h)
    for r in (plot, side, strip):
        pygame.draw.rect(screen, COL_PANEL, r, border_radius=8)

    # ---------------- header
    decoy = info.get('decoy', False)
    screen.blit(mid.render(f"Signal Tracker   step {info.get('step', 0)}/{info.get('horizon', 0)}   "
                           f"mean reward best {info.get('best', 0):.3f} / mean {info.get('mean', 0):.3f}", True, COL_TEXT), (pad, 14))
    tag = "DECOY frame" if decoy else "clean frame"
    img = mid.render(tag, True, COL_DECOY if decoy else COL_CLEAN)
    screen.blit(img, (w - img.get_width() - pad, 14))
    fps = info.get('fps')
    hint = (f"+/- zoom x: {info.get('view', 0)} steps   Shift +/- fps: {fps if fps else 'max'}   Space pause   Esc stop")
    screen.blit(small.render(hint, True, COL_DIM), (pad, 38))

    # ---------------- time plot
    n = len(trace['target'])
    inner = plot.inflate(-24, -24)
    for v in (-1, 0, 1):
        yy = _y(v, inner.top, inner.height)
        pygame.draw.line(screen, COL_GRID, (inner.left, yy), (inner.right, yy), 1)
        screen.blit(small.render(f"{v:+d}", True, COL_DIM), (plot.left + 2, yy - 7))
    if n > 1:
        cap = trace['cap']
        dx = inner.width / max(cap - 1, 1)
        off = cap - n                       # newest on the right
        xs = [inner.left + (off + i) * dx for i in range(n)]
        for i in range(n):                  # cue dots first (background)
            col = COL_DECOY if trace['kind'][i] else COL_CLEAN
            pygame.draw.circle(screen, col, (int(xs[i]), _y(trace['cue'][i], inner.top, inner.height)), 2 if trace['kind'][i] else 3)
        groups = info.get('groups')
        if groups:      # compare mode: one line per group (its best cursor so far) instead of best / mean / worst
            lines = [(f'g{i}', GROUP_COLS[i % len(GROUP_COLS)], 2) for i in range(len(groups)) if f'g{i}' in trace] + [('target', COL_TARGET, 3)]
        else:
            lines = (('worst', COL_WORST, 1), ('mean', COL_MEAN, 2), ('best', COL_BEST, 2), ('target', COL_TARGET, 3))
        for key, col, wd in lines:
            pts = [(xs[i], _y(trace[key][i], inner.top, inner.height)) for i in range(n)]
            pygame.draw.lines(screen, col, False, pts, wd)
    ui = info.get('ui') or {}
    mouse = ui.get('mouse')
    if ui.get('inside') and mouse is not None:      # visible proof that the mouse is read: crosshair + value + bright frame
        pygame.draw.line(screen, COL_HOVER, (inner.left, mouse[1]), (inner.right, mouse[1]), 1)
        pygame.draw.line(screen, COL_GRID, (mouse[0], inner.top), (mouse[0], inner.bottom), 1)
        screen.blit(small.render(f"{ui.get('value', 0.0):+.2f}", True, COL_HOVER), (min(mouse[0] + 8, inner.right - 40), max(mouse[1] - 16, inner.top)))
    pygame.draw.rect(screen, COL_HUMAN if ui.get('inside') else COL_GRID, plot, 2 if ui.get('inside') else 1, border_radius=8)
    flash = ui.get('flash')                 # feedback of the last external perturbation
    if flash and flash[0] == 'kick':
        yy = _y(flash[1], inner.top, inner.height)
        pygame.draw.line(screen, COL_HUMAN, (inner.left, yy), (inner.right, yy), 3)
        screen.blit(small.render("KICK", True, COL_HUMAN), (inner.right - 40, max(yy - 18, inner.top)))
    elif flash and flash[0] == 'burst':
        pygame.draw.rect(screen, COL_DECOY, plot, 4, border_radius=8)
        screen.blit(small.render("DECOY BURST", True, COL_DECOY), (plot.right - 110, plot.top + 6))
    leg = [("target", COL_TARGET), ("best", COL_BEST), ("mean", COL_MEAN), ("worst", COL_WORST),
           ("cue clean", COL_CLEAN), ("cue decoy", COL_DECOY)]
    if info.get('groups'):
        leg = [("target", COL_TARGET)] + [(g, GROUP_COLS[i % len(GROUP_COLS)]) for i, g in enumerate(info['groups'])] + leg[-2:]
    lx = plot.left + 40
    for name, col in leg:
        pygame.draw.circle(screen, col, (lx, plot.bottom - 12), 4)
        t = small.render(name, True, COL_DIM)
        screen.blit(t, (lx + 8, plot.bottom - 20))
        lx += t.get_width() + 28

    # ---------------- side panel: track with every cursor + current frame bars
    track = pygame.Rect(side.left + 16, side.top + 34, side.width - 32, 56)
    screen.blit(small.render("cursors on the track", True, COL_DIM), (side.left + 16, side.top + 10))
    pygame.draw.rect(screen, COL_BG, track, border_radius=6)
    tx = lambda v: int(track.left + (np.clip(v, -1, 1) + 1) / 2 * track.width)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.35, 0.35, len(agents_x))
    for k, (xv, j) in enumerate(zip(agents_x, jitter)):
        gi = info.get('group_idx')
        ccol = COL_CURSOR if gi is None else GROUP_COLS[int(gi[k]) % len(GROUP_COLS)]
        pygame.draw.circle(screen, ccol, (tx(xv), int(track.centery + j * track.height)), 3)
    pygame.draw.line(screen, COL_TARGET, (tx(info.get('target', 0.0)), track.top), (tx(info.get('target', 0.0)), track.bottom), 3)
    if state_row is not None:
        by = track.bottom + 30
        screen.blit(small.render("current frame (what the newest token holds)", True, COL_DIM), (side.left + 16, by - 22))
        for i, (name, col) in enumerate((("cursor", COL_CURSOR), ("cue", COL_DECOY if decoy else COL_CLEAN), ("tag", COL_DECOY if decoy else COL_CLEAN))):
            row = pygame.Rect(side.left + 70, by + i * 28, side.width - 90, 16)
            pygame.draw.rect(screen, COL_BG, row, border_radius=4)
            mid_x = row.centerx
            v = float(np.clip(state_row[i], -2.0, 2.0)) / 2.0
            bar = pygame.Rect(min(mid_x, mid_x + int(v * row.width / 2)), row.top, abs(int(v * row.width / 2)) or 1, row.height)
            pygame.draw.rect(screen, col, bar, border_radius=4)
            screen.blit(small.render(name, True, COL_TEXT), (side.left + 16, row.top))
    # quality numbers of the best cursor so far (what to look for: see README 'Reading a genome')
    if 'mae' in info and info.get('step', 0) > 0:
        gap = info['mae_decoy'] - info['mae_clean']
        ly = side.top + 34 + 56 + 30 + 3 * 28 + 10
        lines = [
            ("best cursor, mean |error|", None),
            (f"  all frames   {info['mae']:.3f}", COL_TEXT),
            (f"  clean frames {info['mae_clean']:.3f}", COL_CLEAN),
            (f"  decoy frames {info['mae_decoy']:.3f}", COL_DECOY),
            (f"  decoy - clean gap {gap:+.3f}", COL_TEXT),
            ("  gap ~0 = memory works; big = copies the cue", COL_DIM),
        ]
        for text, col in lines:
            if ly + 16 > side.bottom - 4:
                break
            screen.blit(small.render(text, True, col or COL_DIM), (side.left + 16, ly))
            ly += 17
    pygame.draw.rect(screen, COL_DECOY if decoy else COL_CLEAN, side, 2, border_radius=8)

    # ---------------- strip: the sequence the model sees (best cursor's history), oldest left
    screen.blit(small.render("model input window (oldest -> newest): cue height, tag bar; red = decoy", True, COL_DIM), (strip.left + 10, strip.top + 6))
    if history is not None:
        seq = history.shape[0]
        cw = (strip.width - 20) / seq
        base = strip.bottom - 12
        top = strip.top + 28
        kinds = info.get('window_kinds')
        for i in range(seq):
            cx = int(strip.left + 10 + i * cw)
            kind = bool(kinds[i]) if kinds is not None and i < len(kinds) else False
            col = COL_DECOY if kind else COL_CLEAN
            tag_h = int(np.clip((history[i, 2] + 1.5) / 3.0, 0.0, 1.0) * (base - top))
            pygame.draw.rect(screen, (45, 49, 66), (cx, base - tag_h, max(int(cw) - 2, 1), tag_h))
            pygame.draw.circle(screen, col, (cx + int(cw / 2), _y(history[i, 1], top, base - top)), max(2, int(cw / 3)))
    _draw_controls(screen, fonts, ctrl, info)
    pygame.display.flip()