"""Pygame window of the Signal Tracker: keyboard / mouse controls and the draw call (pygame is imported lazily)."""
from __future__ import annotations

from collections import deque

import numpy as np

from . import renderer

__all__ = ["Window"]


class Window:
    """
    Keys : Space = pause, + / - = zoom the time axis (visible steps), Shift + / - = framerate, R = reset the sliders,
           Esc or closing the window = stop the episode.
    Mouse: wheel = zoom; drag the sliders of the control panel = change the stream (decoy fraction / run, noise, drift,
           target speed) while it runs; left click on the plot = kick the hidden target there; right click = decoy burst.
    The window only queues these as events (``consume_events``); the env applies them, so the window stays a view.
    The window keeps the last ``buffer_len`` steps; ``view_len`` of them are drawn (the zoom).
    """

    MIN_VIEW = 16
    BURST_FRAMES = 12

    def __init__(self, fps: int | None = 30, size: tuple[int, int] = (1280, 680), trace_len: int = 240, buffer_len: int = 4096):
        self.fps = fps
        self.size = size
        self.screen = None
        self.clock = None
        self.fonts: dict = {}
        self.running = False
        self.paused = False
        self.stop = False
        self.buffer_len = max(int(buffer_len), int(trace_len))
        self.view_len = int(trace_len)
        self.dirty = False                  # a control changed something while paused: redraw the last frame
        self.speed = 0.12                   # cursor speed per step (set by the env)
        self._last = None
        self.groups = None                  # (names, index array): compare mode, one trace per group of cursors
        self.params: dict[str, float] = {}  # slider values as displayed (the env sets the initial ones)
        self.initial: dict[str, float] = {}
        self.events: list[tuple] = []       # ('param', name, value) | ('kick', value) | ('burst', frames)
        self.drag: str | None = None
        self._flash: list = [None, None, 0]     # kind, value, frames left
        self.reset_trace()

    def reset_trace(self) -> None:
        self.trace = {k: deque(maxlen=self.buffer_len) for k in ('target', 'cue', 'kind', 'best', 'mean', 'worst')}

    def set_groups(self, names: list[str] | None, idx=None) -> None:
        """Compare mode: cursor i belongs to group idx[i]; the plot draws each group's best cursor in its own colour."""
        self.groups = None if names is None else (list(names), np.asarray(idx))

    def set_params(self, values: dict[str, float]) -> None:
        self.params = dict(values)
        self.initial = dict(values)

    def consume_events(self) -> list[tuple]:
        ev, self.events = self.events, []
        return ev

    # ---------------------------------------------------------------- geometry / mouse
    def _geometry(self):
        plot, _, _, ctrl = renderer.layout(*self.screen.get_size())
        return plot.inflate(-24, -24), ctrl, renderer.slider_rects(ctrl)

    @staticmethod
    def _value_at(y: float, inner) -> float:
        return float(min(max(1.0 - 2.0 * (y - inner.top) / max(inner.height, 1), -1.0), 1.0))

    def mouse_state(self) -> dict:
        """Mouse position, whether it is over the plot, the track value it points at, the slider under it, last flash."""
        if not self.running:
            return {}
        import pygame
        pos = pygame.mouse.get_pos()
        inner, ctrl, sliders = self._geometry()
        inside = inner.collidepoint(pos) and pygame.mouse.get_focused()
        hover = next((s['name'] for s in sliders if s['row'].collidepoint(pos)), None)
        flash = (self._flash[0], self._flash[1]) if self._flash[2] > 0 else None
        return {'mouse': pos, 'inside': bool(inside), 'value': self._value_at(pos[1], inner),
                'hover_slider': hover, 'drag': self.drag, 'flash': flash}

    def _set_slider(self, name: str, x: float, sliders) -> None:
        s = next(s for s in sliders if s['name'] == name)
        tr = s['track']
        frac = min(max((x - tr.left) / max(tr.width, 1), 0.0), 1.0)
        value = s['lo'] + frac * (s['hi'] - s['lo'])
        value = round(value * 2.0) / 2.0 if name == 'decoy_run' else round(value, 3)
        if self.params.get(name) != value:
            self.params[name] = value
            self.events.append(('param', name, value))
            self.dirty = True

    def _mouse_down(self, e) -> None:
        inner, ctrl, sliders = self._geometry()
        if e.button == 1:
            hit = next((s['name'] for s in sliders if s['row'].collidepoint(e.pos)), None)
            if hit:
                self.drag = hit
                self._set_slider(hit, e.pos[0], sliders)
            elif inner.collidepoint(e.pos):
                v = self._value_at(e.pos[1], inner)
                self.events.append(('kick', v))
                self._flash = ['kick', v, 10]
                self.dirty = True
        elif e.button == 3 and inner.collidepoint(e.pos):
            self.events.append(('burst', self.BURST_FRAMES))
            self._flash = ['burst', None, 10]
            self.dirty = True

    # ---------------------------------------------------------------- controls
    def zoom(self, direction: int) -> None:
        """direction +1 = zoom in (fewer steps across the plot), -1 = zoom out. x axis only."""
        v = self.view_len / 1.25 if direction > 0 else self.view_len * 1.25
        self.view_len = int(min(max(round(v), self.MIN_VIEW), self.buffer_len))
        self.dirty = True

    def change_fps(self, direction: int) -> None:
        if direction > 0:
            self.fps = None if self.fps is None else min(self.fps * 2, 480)
        else:
            self.fps = 5 if self.fps is None else max(self.fps // 2, 1)
        self.dirty = True

    def reset_sliders(self) -> None:
        for name, value in self.initial.items():
            if self.params.get(name) != value:
                self.params[name] = value
                self.events.append(('param', name, value))
        self.dirty = True

    def initialize(self) -> None:
        import pygame
        pygame.init()
        pygame.display.set_caption("Signal Tracker")
        self.screen = pygame.display.set_mode(self.size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.fonts = {'small': pygame.font.SysFont("consolas,menlo,dejavusansmono", 13),
                      'mid': pygame.font.SysFont("consolas,menlo,dejavusansmono", 18, bold=True)}
        self.running, self.stop = True, False

    def record(self, target: float, cue: float, kind: int, x: np.ndarray, best: int, scores: np.ndarray | None = None) -> None:
        t = self.trace
        if self.groups is not None:
            names, idx = self.groups
            sc = scores if scores is not None else -np.abs(x - target)
            for g in range(len(names)):
                m = np.flatnonzero(idx == g)
                if m.size:
                    t.setdefault(f'g{g}', deque(maxlen=self.buffer_len)).append(float(x[m[int(np.argmax(sc[m]))]]))
        t['target'].append(target); t['cue'].append(cue); t['kind'].append(kind)
        err = np.abs(x - target)
        t['best'].append(float(x[best])); t['mean'].append(float(x.mean())); t['worst'].append(float(x[int(np.argmax(err))]))

    def _handle_events(self) -> None:
        """Drain the pygame queue once (no pause loop in here)."""
        import pygame
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.stop = True
            elif e.type == pygame.MOUSEWHEEL:
                self.zoom(1 if e.y > 0 else -1)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                self._mouse_down(e)
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                self.drag = None
                self.dirty = True
            elif e.type == pygame.MOUSEMOTION and self.drag:
                self._set_slider(self.drag, e.pos[0], self._geometry()[2])
            elif e.type == pygame.KEYDOWN:
                shift = bool(e.mod & pygame.KMOD_SHIFT)
                plus = e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS)
                minus = e.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_UNDERSCORE)
                if e.key == pygame.K_ESCAPE:
                    self.stop = True
                elif e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_r:
                    self.reset_sliders()
                elif plus or minus:
                    d = 1 if plus else -1
                    self.change_fps(d) if shift else self.zoom(d)

    def poll_events(self) -> None:
        """Handle events; while paused keep handling them in a plain loop (zoom / sliders redraw the frozen frame)."""
        import pygame
        self._handle_events()
        while self.paused and not self.stop:
            if self.dirty and self._last is not None:
                self._render(*self._last)
            pygame.time.wait(40)
            self._handle_events()

    def consume_stop(self) -> bool:
        stop, self.stop = self.stop, False
        return stop

    def _render(self, agents_x, state_row, history, info) -> None:
        pygame = __import__('pygame')
        self.screen = pygame.display.get_surface() or self.screen
        view = self.view_len
        tr = {k: list(v)[-view:] for k, v in self.trace.items()}
        tr['cap'] = view
        info = dict(info, view=view, fps=self.fps, ui=self.mouse_state(), params=self.params)
        if self.groups is not None:
            names, idx = self.groups
            fit = info.get('fitness')
            info['groups'] = [f"{nm} {fit[idx == g].max():.3f}" if fit is not None and (idx == g).any() else nm for g, nm in enumerate(names)]
            info['group_idx'] = idx
        renderer.draw(self.screen, self.fonts, tr, agents_x, state_row, history, info)
        if self._flash[2] > 0:
            self._flash[2] -= 1
        self.dirty = self._flash[2] > 0 and self.paused

    def draw(self, agents_x, state_row, history, info) -> None:
        if not self.running:
            self.initialize()
        self._last = (agents_x, state_row, history, info)
        self._render(agents_x, state_row, history, info)
        if self.fps:
            self.clock.tick(self.fps)

    def close(self) -> None:
        if self.running:
            import pygame
            pygame.display.quit()
            self.running = False
            self.screen = None