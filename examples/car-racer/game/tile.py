
import pygame
import math

from typing import Optional, Any
from itertools import count
from pygame import Surface

from .config import TileConfig, overwrite_config
from .functional import Type, Position, Color
from .constants import COLOR_START_PRI, COLOR_START_SEC


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

    def _draw_turn_layer(self, layer: str) -> Surface:
        """
        Draw only specified layer for turn segment.
        """
        base = Surface((self.size, self.size), pygame.SRCALPHA)
        
        if self.gravel != 1.0:
            if layer == 'gravel' and self.gravel_px > 0:
                base.fill(self.gravel_color)
            elif layer == 'grass':
                grass_up = pygame.Rect(self.grass_offset, self.grass_offset, self.grass_px_fill, self.grass_curve_offset)
                grass_left = pygame.Rect(0, self.grass_offset, self.grass_curve_offset, self.grass_px_fill)
                pygame.draw.rect(base, self.grass_color, grass_up)
                pygame.draw.rect(base, self.grass_color, grass_left)
            elif layer == 'curb':
                curb_main_up = pygame.Rect(self.curb_offset, self.curb_offset, self.curb_px_fill, self.curb_curve_offset)
                curb_main_left = pygame.Rect(0, self.curb_offset, self.curb_curve_offset, self.curb_px_fill)
                curb_fill_up = [
                    pygame.Rect(self.curb_offset, self.half_size + (self.curb_px_split * i), self.curb_px_fill, self.curb_px_split)
                    for i in range(0, self.curb_pattern_splits_curve, 2)
                ]
                curb_fill_left = [
                    pygame.Rect(self.curb_px_split * i, self.curb_offset, self.curb_px_split, self.curb_px_fill)
                    for i in range(0, self.curb_pattern_splits_curve, 2)
                ]
                curb_fill_arc = [
                    (self.curb_px_split * i / self.curb_pattern_splits_curve,
                     self.curb_px_split * (i + 1) / self.curb_pattern_splits_curve)
                    for i in range(1, self.curb_pattern_splits_curve, 2)
                ]
                pygame.draw.rect(base, self.curb_red, curb_main_up)
                pygame.draw.rect(base, self.curb_red, curb_main_left)
                for fill in curb_fill_up:
                    pygame.draw.rect(base, self.curb_white, fill)
                for fill in curb_fill_left:
                    pygame.draw.rect(base, self.curb_white, fill)
                for i, angles in enumerate(curb_fill_arc):
                    start_angle = -math.radians(90 * i / self.curb_pattern_splits_curve)
                    end_angle = -math.radians(90 * (i + 1) / self.curb_pattern_splits_curve)
                    self.draw_sector(base, self.curb_white, self.center, self.road_px_half + self.curb_px, start_angle, end_angle)
            elif layer == 'road':
                road_up = pygame.Rect(self.road_offset, self.half_size, self.road_px, self.half_size)
                road_left = pygame.Rect(0, self.road_offset, self.half_size, self.road_px)
                pygame.draw.rect(base, self.road_color, road_up)
                pygame.draw.rect(base, self.road_color, road_left)
                pygame.draw.circle(base, self.road_color, self.center, self.road_px // 2)
        
        if self._horizontal_invert and layer != 'gravel':
            base = pygame.transform.flip(base, True, False)
        
        return base

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

    @staticmethod # TODO: Move to .util
    def draw_sector(surface: Surface, color: Color, center: Position, radius: float, start_angle: float, end_angle: float):
        """
        Draw a sector of a circle
        
        Args:
            surface: pygame surface to draw on
            color: RGB color tuple
            center: (x, y) center point
            radius: radius of the circle
            start_angle: start angle in radians
            end_angle: end angle in radians
        """
        # Create a list to hold the points of the sector
        points = [center]
        
        # Calculate number of points based on angle difference
        angle_range = end_angle - start_angle
        # Use more points for larger angles to get smoother curve
        num_points = max(5, int(angle_range * radius / 10))
        
        # Add points along the arc
        for i in range(num_points + 1):
            angle = start_angle + (angle_range * i / num_points)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        
        # Draw the filled polygon
        pygame.draw.polygon(surface, color, points)

    def _draw_turn(self, tile: Surface):
        base = Surface((self.size, self.size), pygame.SRCALPHA)
        base.fill(self.gravel_color)

        if self.gravel != 1.0:
            road_up = pygame.Rect(self.road_offset, self.half_size, self.road_px, self.half_size)
            road_left = pygame.Rect(0, self.road_offset, self.half_size, self.road_px)
            curb_main_up = pygame.Rect(self.curb_offset, self.curb_offset, self.curb_px_fill, self.curb_curve_offset)
            curb_main_left = pygame.Rect(0, self.curb_offset, self.curb_curve_offset, self.curb_px_fill)
            curb_fill_up = [
                pygame.Rect(
                    self.curb_offset, self.half_size + (self.curb_px_split * i), self.curb_px_fill, self.curb_px_split
                )
                for i in range(0, self.curb_pattern_splits_curve, 2)
            ]
            curb_fill_left = [
                pygame.Rect(
                    self.curb_px_split * i, self.curb_offset, self.curb_px_split, self.curb_px_fill
                )
                for i in range(0, self.curb_pattern_splits_curve, 2)
            ]
            curb_fill_arc = [
                (
                    -math.radians(90 * i / self.curb_pattern_splits_curve),
                    -math.radians(90 * (i + 1) / self.curb_pattern_splits_curve)
                )
                for i in range(1, self.curb_pattern_splits_curve, 2)
            ]
            grass_up = pygame.Rect(self.grass_offset, self.grass_offset, self.grass_px_fill, self.grass_curve_offset)
            grass_left = pygame.Rect(0, self.grass_offset, self.grass_curve_offset, self.grass_px_fill)

            pygame.draw.rect(base, self.grass_color, grass_up)
            pygame.draw.rect(base, self.grass_color, grass_left)
            pygame.draw.rect(base, self.curb_red, curb_main_up)
            pygame.draw.rect(base, self.curb_red, curb_main_left)
            # pygame.draw.circle(tile, self.curb_red, self.centre, self.road_px // 2 + self.curb_px + 1)
            for fill in curb_fill_up: pygame.draw.rect(base, self.curb_white, fill)
            for fill in curb_fill_left: pygame.draw.rect(base, self.curb_white, fill)
            for angles in curb_fill_arc: self.draw_sector(
                base, self.curb_white, self.center, self.road_px_half + self.curb_px, *angles
            )
            pygame.draw.rect(base, self.road_color, road_up)
            pygame.draw.rect(base, self.road_color, road_left)
            pygame.draw.circle(base, self.road_color, self.center, self.road_px // 2)

            if self._horizontal_invert:
                base = pygame.transform.flip(base, True, False)

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
