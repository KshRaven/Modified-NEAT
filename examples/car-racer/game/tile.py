from __future__ import annotations

import pygame
import math

from typing import Optional, Any
from itertools import count
from pygame import Surface

from .config import TileConfig, overwrite_config
from .functional import Position, Type
from .constants import COLOR_START_PRI, COLOR_START_SEC

__all__ = ["Tile"]


COLOR_BOUNDARY = (125, 125, 255)
COLOR_VOID = (0, 0, 0)


class Tile:
    __indexer__ = count()

    def __init__(
        self, type: Type, position: Position, size: int, rotation: int = 0,
        config: TileConfig | None = None, **params: dict[str, Any]
    ):
        if config is None:
            config = TileConfig()

        overwrite_config(config, params)
        self.config = config
        
        self.index = next(self.__indexer__)
        self.type = type
        self.size = int(size)
        self.position = position
        self.position_abs: Position = tuple([v * self.size for v in self.position])

        self._horizontal_invert = self.type == Type.RIGHT

        # Surface colors from config
        self.road_color = config.road_color
        self.grass_color = config.grass_color
        self.curb_red = config.curb_red
        self.curb_white = config.curb_white
        self.gravel_color = config.gravel_color

        # Surface ratios from config
        self.gravel = max(0.0, min(0.20, float(config.gravel))) if config.gravel != 1.0 else 1.0
        self.grass = max(0.0, min(0.20, float(config.grass))) if config.gravel != 1.0 else 0.0
        self.curb = max(0.0, min(0.20, float(config.curb))) if config.gravel != 1.0 else 0.0
        self.empty = self.gravel == 1.0

        self.rotation = rotation % 360

        self.gravel_px = max(0, int(self.size * self.gravel))
        self.grass_px = max(0, int(self.size * self.grass))
        self.curb_px = max(0, int(self.size * self.curb))
        self.boundary_px = 2

        self.road_offset = self.gravel_px + self.grass_px + self.curb_px
        self.road_px = max(2, self.size - 2 * self.road_offset)
        self.road_px_half = self.road_px / 2
        self.curb_offset = (self.size - self.road_px) / 2 - self.curb_px
        self.grass_offset = (self.size - self.road_px) / 2 - (self.curb_px + self.grass_px)
        self.curb_px_fill = self.curb_px * 2 + self.road_px
        self.grass_px_fill = self.grass_px * 2 + self.curb_px_fill
        self.half_size = self.size / 2
        self.center = (self.half_size, self.half_size)
        self.center_abs = tuple(k + self.half_size for k in self.position_abs)
        
        self.curb_pattern_splits = config.curb_pattern_splits
        self.curb_pattern_splits += (2 - self.curb_pattern_splits % 2)
        self.curb_px_split = self.size / self.curb_pattern_splits
        self.curb_pattern_splits_curve = math.ceil(self.half_size / self.curb_px_split)
        self.curb_curve_offset = self.half_size + self.road_px_half + self.curb_px
        self.grass_curve_offset = self.half_size + self.road_px_half + self.curb_px + self.grass_px
        
        start_checker = config.start_checker_size
        self.start_checker_size = int(start_checker) if start_checker is not None else int(max(3, self.road_px // 8))
        self.finish_line_back_offset = int(config.finish_line_back_offset_ratio * self.size)
        self.center_check_abs = tuple(
            ax + (self.half_size * func(math.radians(
                (
                    self.rotation + 90 # NOTE: Up=0, Left=90, Down=180, Right=270
                    + (
                        +90 if self.type == Type.LEFT else -90 if self.type == Type.RIGHT else (
                        180 if (self.rotation // 90) % 2 == 0 else 0
                    ))
                    * (-1 if i == 1 else 1)
                ) % 360
            )))
            for i, (ax, func) in enumerate(zip(self.center_abs, [math.cos, math.sin]))
        )
        self.respawn_point = tuple(
            ax + (self.half_size * func(math.radians(
                (
                    self.rotation + 90 + 180 # NOTE: Up=0, Left=90, Down=180, Right=270
                    + (
                        +90 if self.type == Type.LEFT else -90 if self.type == Type.RIGHT else (
                        180 if (self.rotation // 90) % 2 == 0 else 0
                    ))
                    * (-1 if i == 1 else 1)
                ) % 360
            )))
            for i, (ax, func) in enumerate(zip(self.center_abs, [math.cos, math.sin]))
        )

    def __str__(self):
        return f"Tile({self.type}, {self.position}, {self.rotation})"
    
    def __repr__(self):
        return self.__str__()

    @property
    def orientation(self):
        return self.rotation + (+90 if self.type == Type.LEFT else -90 if self.type == Type.RIGHT else 0)
    # ------------------------------------------------------------------
    # Layer-specific rendering for mask generation
    # ------------------------------------------------------------------

    def _render_layer_mask(self, layer: str) -> Surface:
        """
        Render only a specific layer (gravel, grass, curb, or road) to a surface.
        Used for generating collision masks.
        
        Args:
            layer: One of 'gravel', 'grass', 'curb', or 'road'
            
        Returns:
            Surface containing only the specified layer
        """
        if self.type in (Type.STRAIGHT, Type.START):
            base = self._draw_straight_layer(layer)
        elif self.type in (Type.LEFT, Type.RIGHT):
            base = self._draw_turn_layer(layer)
        else:
            raise ValueError(f"Unsupported segment type: {self.type}")

        # base = scale_image(base, 0.90)
        
        # Apply rotation consistently to all layers, matching render() behavior
        if self.rotation:
            base = pygame.transform.rotate(base, self.rotation)
        
        return base

    def _draw_straight_layer(self, layer: str) -> Surface:
        """
        Draw only specified layer for straight segment.
        """
        base = Surface((self.size, self.size), pygame.SRCALPHA)
        
        if self.gravel != 1.0:
            if layer == 'gravel' and self.gravel_px > 0:
                base.fill(self.gravel_color)
            elif layer == 'grass':
                grass = pygame.Rect(self.grass_offset, 0, self.grass_px_fill, self.size)
                pygame.draw.rect(base, self.grass_color, grass)
            elif layer == 'curb':
                curb_main = pygame.Rect(self.curb_offset, 0, self.curb_px_fill, self.size)
                curb_fill = [
                    pygame.Rect(self.curb_offset, self.curb_px_split * i, self.curb_px_fill, self.curb_px_split)
                    for i in range(0, self.curb_pattern_splits, 2)
                ]
                pygame.draw.rect(base, self.curb_red, curb_main)
                for fill in curb_fill:
                    pygame.draw.rect(base, self.curb_white, fill)
            elif layer == 'road':
                road = pygame.Rect(self.road_offset, 0, self.road_px, self.size)
                pygame.draw.rect(base, self.road_color, road)
        
        return base

    def _arc_ring(self, surface: Surface, color, center: Position, r_in: float, r_out: float,
                  t0: float = 0.0, t1: float = 90.0) -> None:
        """Fill a ring segment between two radii. Angle 0 points right, 90 points up (screen space)."""
        if r_out <= r_in or r_out <= 0:
            return
        n = max(8, int(math.radians(t1 - t0) * r_out / 3))
        ts = [math.radians(t0 + (t1 - t0) * i / n) for i in range(n + 1)]
        outer = [(center[0] + r_out * math.cos(t), center[1] - r_out * math.sin(t)) for t in ts]
        inner = [(center[0] + r_in * math.cos(t), center[1] - r_in * math.sin(t)) for t in reversed(ts)]
        pygame.draw.polygon(surface, color, outer + inner)

    def _pocket(self, color, radius: float) -> Surface:
        """Inner-corner square (pivot = bottom-left tile corner) filled with `color` except a quarter disc of `radius`."""
        side = int(math.ceil(self.road_offset))
        pocket = Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.rect(pocket, color, pygame.Rect(0, self.size - side, side, side))
        pygame.draw.circle(pocket, (0, 0, 0, 0), (0, self.size), radius)
        return pocket

    def _turn_curb_stripes(self, base: Surface) -> None:
        """Red/white kerb on a turn: straight stripes on the outer arms, radial stripes on both bends."""
        r, c, ro = self.road_px_half, self.curb_px, self.road_offset
        h, size, split = self.half_size, self.size, self.curb_px_split
        if c <= 0:
            return
        # Straight stretches (outer side only; the inner side is the bend)
        for i in range(0, self.curb_pattern_splits_curve, 2):
            pygame.draw.rect(base, self.curb_white, pygame.Rect(size - ro, h + split * i, c, split))
            pygame.draw.rect(base, self.curb_white, pygame.Rect(split * i, ro - c, split, c))
        # Outer bend (pivot: tile centre) and inner bend (pivot: bottom-left tile corner)
        for pivot, r0, r1, first in (
            (self.center, r, r + c, 1),
            ((0, size), ro - c, ro, 0),
        ):
            n = max(2, round((math.pi / 2) * (r0 + r1) / 2 / split))
            for i in range(first, n, 2):
                self._arc_ring(base, self.curb_white, pivot, r0, r1, 90 * i / n, 90 * (i + 1) / n)

    def _turn_layer_surface(self, layer: str) -> Surface:
        """One layer of a turn tile, unrotated. Road/curb are real bends; grass/gravel stay square."""
        base = Surface((self.size, self.size), pygame.SRCALPHA)
        size, h, ro, c, r = self.size, self.half_size, self.road_offset, self.curb_px, self.road_px_half
        if self.gravel == 1.0:
            return base
        if layer == 'gravel':
            if self.gravel_px > 0:
                base.fill(self.gravel_color)
        elif layer == 'grass':
            grass_up = pygame.Rect(self.grass_offset, self.grass_offset, self.grass_px_fill, self.grass_curve_offset)
            grass_left = pygame.Rect(0, self.grass_offset, self.grass_curve_offset, self.grass_px_fill)
            pygame.draw.rect(base, self.grass_color, grass_up)
            pygame.draw.rect(base, self.grass_color, grass_left)
        elif layer == 'curb':
            pygame.draw.rect(base, self.curb_red, pygame.Rect(ro, h, size - 2 * ro + c, h))              # entry arm, outer side
            pygame.draw.rect(base, self.curb_red, pygame.Rect(0, ro - c, h, size - 2 * ro + c))          # exit arm, outer side
            pygame.draw.circle(base, self.curb_red, self.center, r + c)                                  # outer bend
            base.blit(self._pocket(self.curb_red, ro - c), (0, 0))                                       # inner bend
            self._turn_curb_stripes(base)
        elif layer == 'road':
            pygame.draw.rect(base, self.road_color, pygame.Rect(ro, h, self.road_px, h))
            pygame.draw.rect(base, self.road_color, pygame.Rect(0, ro, h, self.road_px))
            pygame.draw.circle(base, self.road_color, self.center, self.road_px // 2)
            base.blit(self._pocket(self.road_color, ro), (0, 0))                                         # fills the inner corner out to the bend
        if self._horizontal_invert and layer != 'gravel':
            base = pygame.transform.flip(base, True, False)
        return base

    def _draw_turn_layer(self, layer: str) -> Surface:
        """Draw only the specified layer for a turn segment (used for the surface masks)."""
        return self._turn_layer_surface(layer)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, surface: Surface, pos: Optional[tuple[int, int]] = None, verbose: bool | int = True) -> None:
        if pos is None:
            pos = self.position

        # tile = Surface((self.size, self.size), pygame.SRCALPHA)
        # self._draw_base_layers(tile)

        if self.type in (Type.STRAIGHT, Type.START):
            base = self._draw_straight(surface)
            if self.type == Type.START:
               self._draw_start_line(base)
        elif self.type == Type.LEFT:
            base = self._draw_turn(surface)
        elif self.type == Type.RIGHT:
            base = self._draw_turn(surface)
        else:
            raise ValueError(f"Unsupported segment type: {self.type}")
           
        if self.rotation:
            base = pygame.transform.rotate(base, self.rotation)

        surface.blit(base, pos)
            
        if verbose and not self.empty:
            # Draw boundary on top of all layers with transparency
            # Boundary is drawn before rotation, then rotated with the tile
            boundary = self.draw_boundary(surface)
            
            if self.rotation:
                boundary = pygame.transform.rotate(boundary, self.rotation)
                
            # Blit boundary with alpha blending so transparent parts don't overlay
            surface.blit(boundary, pos, special_flags=pygame.BLEND_ALPHA_SDL2)

    def _draw_base_layers(self, tile: Surface) -> None:
        tile.fill(self.gravel_color if self.gravel_px > 0 else self.grass_color)

        if self.gravel_px > 0:
            inner = pygame.Rect(
                self.gravel_px,
                self.gravel_px,
                self.size - 2 * self.gravel_px,
                self.size - 2 * self.gravel_px,
            )
            if inner.width > 0 and inner.height > 0:
                pygame.draw.rect(tile, self.grass_color, inner)

    # -----------------------------
    # Straight segment
    # -----------------------------

    def _draw_straight(self, tile: Surface):
        base = Surface((self.size, self.size), pygame.SRCALPHA)
        base.fill(self.gravel_color) 

        if self.gravel != 1.0:
            road = pygame.Rect(self.road_offset, 0, self.road_px, self.size)
            curb_main = pygame.Rect(self.curb_offset, 0, self.curb_px_fill, self.size)
            curb_fill = [
                pygame.Rect(self.curb_offset, self.curb_px_split * i, self.curb_px_fill, self.curb_px_split)
                for i in range(0, self.curb_pattern_splits, 2)
            ]
            grass = pygame.Rect(self.grass_offset, 0, self.grass_px_fill, self.size)

            pygame.draw.rect(base, self.grass_color, grass) # Then fill in grass
            pygame.draw.rect(base, self.curb_red, curb_main) # Then fill in the curb
            for fill in curb_fill: pygame.draw.rect(base, self.curb_white, fill)
            pygame.draw.rect(base, self.road_color, road) # Then draw the road at the top most object levelling

            tile.blit(base, self.position_abs)

        return base

    def _draw_start_line(self, tile: Surface):
        """Draw finish line visual (checker pattern) in curb region, positioned behind center."""
        band_h = max(2, self.size // 12)
        band_y = self.size - band_h - self.finish_line_back_offset
        sq = self.start_checker_size

        x0 = self.road_offset
        x1 = self.size - self.road_offset

        for row in range(2):
            y = band_y + row * max(1, band_h // 2)
            x = x0
            i = 0
            while x < x1:
                rect = pygame.Rect(x, y, min(sq, x1 - x), max(1, band_h // 2))
                color = COLOR_START_PRI if (i + row) % 2 == 0 else COLOR_START_SEC
                pygame.draw.rect(tile, color, rect)
                x += sq
                i += 1
    
    def _render_finish_line_mask(self) -> Surface:
        """Render finish line mask for straight tile. Covers full width, positioned behind center."""
        base = Surface((self.size, self.size), pygame.SRCALPHA)
        band_h = max(2, self.size // 12)
        band_y = self.size - band_h - self.finish_line_back_offset
        
        # Mask covers full width of tile (x=0 to x=size)
        for row in range(2):
            y = band_y + row * max(1, band_h // 2)
            rect = pygame.Rect(0, y, self.size, max(1, band_h // 2))
            pygame.draw.rect(base, (255, 255, 255, 255), rect)
        
        return base

    # -----------------------------
    # Turn segment
    # -----------------------------

    def _draw_turn(self, tile: Surface):
        base = Surface((self.size, self.size), pygame.SRCALPHA)
        base.fill(self.gravel_color)

        if self.gravel != 1.0:
            for layer in ('grass', 'curb', 'road'):
                base.blit(self._turn_layer_surface(layer), (0, 0))
            tile.blit(base, self.position_abs)

        return base

    def _render_boundary_mask(self) -> Surface:
        """
        Render only the thin 1-pixel boundary to a mask surface (white pixels).
        Used for generating collision masks in grid.py.
        Creates transparent surface with white pixels only where boundary exists.
        
        Returns:
            Surface with white pixels for boundary, transparent elsewhere
        """
        base = Surface((self.size, self.size), pygame.SRCALPHA)
        
        if self.type in (Type.STRAIGHT, Type.START):
            # Straight tiles: 1-pixel boundaries on left and right edges
            pygame.draw.rect(base, (255, 255, 255, 255), pygame.Rect(0, 0, self.boundary_px, self.size))
            pygame.draw.rect(base, (255, 255, 255, 255), pygame.Rect(self.size - self.boundary_px, 0, self.boundary_px, self.size))
        elif self.type in (Type.LEFT, Type.RIGHT):
            # Turn tiles: 1-pixel boundaries on top and non-turn edges
            offset = 0 if self._horizontal_invert else self.size - self.boundary_px
            pygame.draw.rect(base, (255, 255, 255, 255), pygame.Rect(0, 0, self.size, self.boundary_px))
            pygame.draw.rect(base, (255, 255, 255, 255), pygame.Rect(offset, 0, self.boundary_px, self.size))
        else:
            raise ValueError(f"Unsupported segment type: {self.type}")
        
        if self.rotation:
            base = pygame.transform.rotate(base, self.rotation)
        
        return base

    def draw_boundary(self, base: Surface):
        """
        Render thin 1-pixel boundary overlay on top of tile layers.
        Uses transparency so only the blue boundary shows, no black void.
        
        Args:
            base: Surface parameter (unused, kept for compatibility)
            
        Returns:
            SRCALPHA surface with only the boundary drawn in COLOR_BOUNDARY
        """
        base = Surface((self.size, self.size), pygame.SRCALPHA)
        
        # Draw boundary pixels only (thin 1-pixel edges)
        if self.type in (Type.STRAIGHT, Type.START):
            # Straight tiles: 1-pixel boundaries on left and right edges
            boundary_color = COLOR_BOUNDARY + (255,)  # Add alpha channel
            pygame.draw.rect(base, boundary_color, pygame.Rect(0, 0, self.boundary_px, self.size)) # Left side
            pygame.draw.rect(base, boundary_color, pygame.Rect(self.size - self.boundary_px, 0, self.boundary_px, self.size)) # Right side
        elif self.type in (Type.LEFT, Type.RIGHT):
            # Turn tiles: 1-pixel boundaries on top and left edges
            boundary_color = COLOR_BOUNDARY + (255,)  # Add alpha channel
            offset = 0 if self._horizontal_invert else self.size - self.boundary_px
            pygame.draw.rect(base, boundary_color, pygame.Rect(0, 0, self.size, self.boundary_px)) # Top side
            pygame.draw.rect(base, boundary_color, pygame.Rect(offset, 0, self.boundary_px, self.size)) # Non-turn side
        else:
            raise ValueError(f"Unsupported segment type: {self.type}")
        
        return base