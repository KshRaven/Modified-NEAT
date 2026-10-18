import math
import numpy as np
import pygame

from numba import njit, types
from numba.extending import overload
from pygame import Surface, transform, draw

from .functional import Color, Position


def scale_image(img: Surface, factor: float) -> Surface:
    """
    Scale an image by a given factor.
    
    Args:
        img: Pygame surface to scale
        factor: Scale factor (1.0 = no change, 0.5 = half size, 2.0 = double size)
        
    Returns:
        Scaled pygame surface
    """
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return transform.scale(img, size)


def blit_rotate_center(surface: Surface, image: Surface, top_left: tuple[int, int], angle: float) -> None:
    """
    Blit a rotated image centered at top_left position.
    
    Args:
        surface: Target surface to blit to
        image: Image to rotate and blit
        top_left: Position to center the image at
        angle: Rotation angle in degrees
    """
    rotated_image = transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    surface.blit(rotated_image, new_rect.topleft)


def draw_sector(surface: Surface, color: Color, center: tuple[int, int], 
                radius: float, start_angle: float, end_angle: float) -> None:
    """
    Draw a sector of a circle (pie slice).
    
    Args:
        surface: Pygame surface to draw on
        color: RGB color tuple
        center: (x, y) center point
        radius: Radius of the circle
        start_angle: Start angle in radians
        end_angle: End angle in radians
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
    draw.polygon(surface, color, points)


def clip_rotated_surface(surface: Surface, rotation: int, cell_size: int) -> Surface:
    """
    Rotate a surface and clip it to cell boundaries to prevent overflow.
    
    When pygame.transform.rotate() is called, the resulting surface can be larger
    than the original (e.g., a 100x100 surface rotated 45° becomes ~141x141).
    This causes mask overflow into adjacent cells.
    
    Args:
        surface: Original surface to rotate
        rotation: Rotation angle in degrees
        cell_size: Cell size in pixels (output dimensions)
        
    Returns:
        Surface clipped to exactly cell_size x cell_size
    """
    # Rotate the surface
    rotated = transform.rotate(surface, rotation)
    rot_width, rot_height = rotated.get_size()
    
    # Create a cell-sized output surface
    import pygame
    clipped = Surface((cell_size, cell_size), pygame.SRCALPHA)
    
    # Calculate center offset to align rotated content
    offset_x = (cell_size - rot_width) // 2
    offset_y = (cell_size - rot_height) // 2
    
    # Calculate the intersection rect between rotated surface and cell output
    # This ensures we only copy the portion that fits within the cell
    intersect_x = max(0, -offset_x)
    intersect_y = max(0, -offset_y)
    intersect_width = min(rot_width, rot_width - intersect_x - max(0, offset_x + rot_width - cell_size))
    intersect_height = min(rot_height, rot_height - intersect_y - max(0, offset_y + rot_height - cell_size))
    
    # Calculate destination position in output surface (clamp to cell bounds)
    dest_x = max(0, offset_x)
    dest_y = max(0, offset_y)
    
    # Create a source rectangle from the rotated surface, taking only the visible portion
    source_rect = pygame.Rect(intersect_x, intersect_y, intersect_width, intersect_height)
    
    # Blit only the intersection portion
    if source_rect.width > 0 and source_rect.height > 0:
        clipped.blit(rotated, (dest_x, dest_y), source_rect)
    
    return clipped


def draw_arrow(
        surface: Surface, start: Position, end: Position, 
        width: int = 3, head_length: float = 10, head_angle: float = 30, 
        color: Color = (0, 0, 0),
):
    # Draw the shaft
    draw.line(surface, color, start, end, width)

    # Direction vector
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)

    # Arrowhead angles
    left_angle = angle + math.radians(180 - head_angle)
    right_angle = angle - math.radians(180 - head_angle)

    # Arrowhead points
    left_point = (
        end[0] + head_length * math.cos(left_angle),
        end[1] + head_length * math.sin(left_angle)
    )
    right_point = (
        end[0] + head_length * math.cos(right_angle),
        end[1] + head_length * math.sin(right_angle)
    )

    # Draw arrowhead (triangle)
    draw.polygon(surface, color, [end, left_point, right_point])


# ---------------------------------------------------------------------------
# Numba-compatible index grid
# ---------------------------------------------------------------------------

def grid(dims: tuple[int, ...]) -> tuple[int, list[int | tuple[int, ...]]]:
    """
    Pure-Python fallback / reference implementation.

    NOTE: ``dims`` must be a tuple (not a list) so the numba @overload below can
    specialise on its length at compile time.
    """
    total = math.prod(dims)
    if len(dims) <= 0 or len(dims) > 3: raise ValueError("Unsupported number of dims")
    indices = list(np.ndindex(*dims))
    return total, indices


@overload(grid)
def _grid_overload(dims: tuple[int, ...]):
    if not isinstance(dims, (types.UniTuple, types.Tuple)):
        return None

    ndim = len(dims)
    if ndim <= 0 or ndim > 3:
        raise TypeError("Unsupported number of dims (grid supports 1, 2, or 3 dims)")

    if ndim == 1:
        def impl(dims: tuple[int, ...]):
            n0 = dims[0]
            indices = [i for i in range(n0)]
            return n0, indices
        return impl

    if ndim == 2:
        def impl(dims: tuple[int, ...]):
            n0, n1 = dims[0], dims[1]
            indices = [(i, j) for i in range(n0) for j in range(n1)]
            return n0 * n1, indices
        return impl

    def impl(dims: tuple[int, ...]):
        n0, n1, n2 = dims[0], dims[1], dims[2]
        indices = [(i, j, k) for i in range(n0) for j in range(n1) for k in range(n2)]
        return n0 * n1 * n2, indices
    return impl


_PANEL_CACHE: dict[tuple, Surface] = {}


def draw_panel(
    surface: Surface, rect: pygame.Rect, color: tuple[int, int, int] = (0, 0, 0),
    alpha: int = 120, radius: int = 6,
) -> None:
    """Blit a translucent rounded backdrop. The SRCALPHA surface is cached per (size, colour,
    alpha), so a steady overlay costs one blit per frame."""
    if alpha <= 0 or rect.w <= 0 or rect.h <= 0:
        return
    key = (rect.w, rect.h, tuple(color), int(alpha), radius)
    panel = _PANEL_CACHE.get(key)
    if panel is None:
        if len(_PANEL_CACHE) > 64:
            _PANEL_CACHE.clear()
        panel = Surface((rect.w, rect.h), pygame.SRCALPHA)
        draw.rect(panel, (*color, int(alpha)), panel.get_rect(), border_radius=radius)
        _PANEL_CACHE[key] = panel
    surface.blit(panel, rect.topleft)