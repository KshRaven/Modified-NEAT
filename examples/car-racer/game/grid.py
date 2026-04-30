import sys
import pygame
import numpy as np
import random

from typing import Optional, Any
from collections import deque
from pygame import Surface, Mask

from .config import TileConfig, GridConfig, overwrite_config
from .functional import Type, Direction, Position
from .functional import get_next_direction, get_next_position, direction_to_rotation, get_next_rotation
from .constants import COLOR_EMPTY
from .tile import Tile


class Grid:
    def __init__(
            self,
            config: GridConfig | None = None, tile_config: TileConfig | None = None,
            **params: dict[str, Any],
        ):
        if config is None:
            config = GridConfig()
        if tile_config is not None: # Overwrite
            config.tile = tile_config
        else:
            tile_config = config.tile

        overwrite_config(config, params)
        overwrite_config(tile_config, params)
        for attr, value in vars(tile_config).items():
            if attr.lower() != "params" and "_" not in attr:
                setattr(config, attr, value)
        self.config = config
        self.tile_config = tile_config
        
        self.cols, self.rows = tuple(int(dim) for dim in config.grid_size)
        self.cell_size = int(config.cell_size)
        self.width, self.height = self.cols * self.cell_size, self.rows * self.cell_size
        self.granularity = config.granularity  # TODO: Fix 'cell' mode
        
        self.cells: list[list[Optional[Tile]]] = [
            [None for _ in range(self.cols)] for _ in range(self.rows)
        ]
        self._cells = np.empty((self.rows, self.cols), dtype=object)
        self.continuity: dict[Tile, Tile] = dict()
        self.start_cell: Tile | None = None

        # Mask storage
        self.full_gravel_mask = None
        self.full_grass_mask = None
        self.full_curb_mask = None
        self.full_road_mask = None
        self.full_track_mask = None  # Mask of all non-empty tiles (actual track)
        self.full_finish_mask = None  # Mask of finish line (START tiles)
        self.full_boundary_mask = None  # Mask of thin 1-pixel tile boundaries
        self.full_grid_mask = None  # Mask of entire grid area (for boundary checks). TODO: Implement
        
        # Surface storage for debugging/visualization
        self._surface_gravel = None
        self._surface_grass = None
        self._surface_curb = None
        self._surface_road = None
        self._surface_track = None  # Surface for track visualization
        self._surface_boundary = None  # Surface for boundary visualization
        self._surface_finish = None  # Surface for finish line visualization
        self._surface_grid = None  # Surface for entire grid visualization
        
        self.boundary_limits = (self.cols * self.cell_size, self.rows * self.cell_size)
        self.initialized = False

    def verify_track(self) -> bool:
        """
        Ensures the track is a fully connected loop
        """
        pass

    def _reset(self) -> None:
        """
        Wipe every cell, mask, and cached surface back to their initial-empty state.
        Must be called at the top of build() so that re-randomising (or any manual
        rebuild) never bleeds old tiles into the new layout.
        """
        # ── cell arrays ───────────────────────────────────────────────────────
        self._cells     = np.empty((self.rows, self.cols), dtype=object)
        self.cells      = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.continuity = {}
        self.start_cell = None
        self.initialized = False
    
        # ── collision masks ───────────────────────────────────────────────────
        self.full_gravel_mask = None
        self.full_grass_mask  = None
        self.full_curb_mask   = None
        self.full_road_mask   = None
        self.full_track_mask  = None
        self.full_finish_mask = None
        self.full_boundary_mask = None
        self.full_grid_mask   = None
    
        # ── debug / visualisation surfaces ───────────────────────────────────
        self._surface_gravel = None
        self._surface_grass  = None
        self._surface_curb   = None
        self._surface_road   = None
        self._surface_track  = None
        self._surface_boundary = None
        self._surface_finish = None
        self._surface_grid   = None

    def random(
        self,
        max_attempts: int = 500,
        min_tiles:    int  = 0,     # 0 → auto: max(4, rows + cols)
        seed: Optional[int] = None,
    ) -> None:
        """
        Randomly generate a closed-loop track and call self.build() with the result.
    
        Parameters
        ----------
        max_attempts : int
            Independent backtracking runs before giving up.
        min_tiles : int
            Minimum non-start tiles.  0 (default) → max(4, rows + cols).
        seed : int | None
            Optional RNG seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
    
        sys.setrecursionlimit(max(10_000, self.rows * self.cols * 8))
    
        _min = max(4, self.rows + self.cols) if min_tiles == 0 else max(4, min_tiles)
    
        # ------------------------------------------------------------------ #
        # Helpers                                                              #
        # ------------------------------------------------------------------ #
    
        def _in_bounds(pos: Position) -> bool:
            x, y = pos
            return 0 <= x < self.cols and 0 <= y < self.rows
    
        # ------------------------------------------------------------------ #
        # Entry cell                                                           #
        # ------------------------------------------------------------------ #
        #
        # For a given (start_pos, start_dir) there is exactly ONE predecessor
        # cell — the cell from which a single step in start_dir lands on
        # start_pos.  We call this the *entry cell*.
        #
        # Because get_next_position moves +1 in start_dir, the entry cell is
        # one step in the OPPOSITE direction from start_pos:
        #   opposite(d) = Direction((d.value + 2) % 4)
        #
        #   RIGHT(0) ↔ LEFT(2)
        #   UP(1)    ↔ DOWN(3)
        #
        # The BFS lookahead checks whether entry is still reachable rather than
        # "any neighbour of start", making it direction-aware.
    
        def _entry_cell(start_pos: Position, start_dir: Direction) -> Position:
            opp = Direction((start_dir.value + 2) % 4)
            return get_next_position(start_pos, opp)
    
        # ------------------------------------------------------------------ #
        # BFS lookahead                                                        #
        # ------------------------------------------------------------------ #
    
        def _can_reach_entry(
            current: Position,
            entry:   Position,
            visited: set,
        ) -> bool:
            """
            BFS over free (unvisited) cells from *current*.
            Returns True iff *entry* is still reachable.
    
            Edge cases
            ----------
            current == entry : already there → True immediately.
            entry in visited : occupied → False immediately.
            """
            if current == entry:
                return True
            if entry in visited:
                return False
    
            queue = deque([current])
            seen  = {current}
            while queue:
                pos = queue.popleft()
                for d in Direction:
                    nxt = get_next_position(pos, d)
                    if nxt == entry:
                        return True
                    x, y = nxt
                    if (0 <= x < self.cols and 0 <= y < self.rows
                            and nxt not in visited
                            and nxt not in seen):
                        seen.add(nxt)
                        queue.append(nxt)
            return False
    
        # ------------------------------------------------------------------ #
        # Backtracker                                                          #
        # ------------------------------------------------------------------ #
    
        def _backtrack(
            layout:    list,
            cur_pos:   Position,
            cur_dir:   Direction,
            visited:   set,
            start_pos: Position,
            start_dir: Direction,
            entry:     Position,
        ) -> list | None:
            """
            cur_pos / cur_dir = head of path so far.
            next_pos          = get_next_position(cur_pos, cur_dir)
                                always fixed — Type only picks exit direction.
    
            Close condition (both must hold)
            ---------------------------------
            1. next_pos == start_pos       (position)
            2. cur_dir  == start_dir       (direction — START is a STRAIGHT tile)
            3. len(layout) >= _min         (minimum length)
            """
            next_pos = get_next_position(cur_pos, cur_dir)
    
            # ── close attempt ────────────────────────────────────────────────
            if next_pos == start_pos:
                if cur_dir == start_dir and len(layout) >= _min:
                    return layout
                # Position correct but direction wrong (or too short) — dead end
                return None
    
            # ── hard guards ──────────────────────────────────────────────────
            if not _in_bounds(next_pos) or next_pos in visited:
                return None
    
            # ── direction-aware BFS lookahead ─────────────────────────────────
            if not _can_reach_entry(next_pos, entry, visited | {next_pos}):
                return None
    
            # ── try all exit turns in random order ────────────────────────────
            moves = [Type.STRAIGHT, Type.LEFT, Type.RIGHT]
            random.shuffle(moves)
    
            visited.add(next_pos)
            for move in moves:
                next_dir = get_next_direction(cur_dir, move)
                result = _backtrack(
                    layout + [move],
                    next_pos, next_dir,
                    visited,
                    start_pos, start_dir,
                    entry,
                )
                if result is not None:
                    visited.discard(next_pos)
                    return result
            visited.discard(next_pos)
    
            return None
    
        # ------------------------------------------------------------------ #
        # Random start-position generator                                      #
        # ------------------------------------------------------------------ #
    
        def _random_start() -> tuple:
            """
            Edge constraint:
            Horizontal (LEFT / RIGHT) → column in [1, cols-2]
            Vertical   (UP   / DOWN)  → row    in [1, rows-2]
            """
            d = random.choice(list(Direction))
            if d in (Direction.LEFT, Direction.RIGHT):
                col = random.randint(1, self.cols - 2)
                row = random.randint(0, self.rows - 1)
            else:
                col = random.randint(0, self.cols - 1)
                row = random.randint(1, self.rows - 2)
            return (col, row), d
    
        # ------------------------------------------------------------------ #
        # Main loop                                                            #
        # ------------------------------------------------------------------ #
    
        for _ in range(max_attempts):
            start_pos, start_dir = _random_start()
    
            entry = _entry_cell(start_pos, start_dir)
            if not _in_bounds(entry):
                continue                # entry cell off-grid, pick another start
    
            visited = {start_pos}
            layout  = _backtrack(
                [], start_pos, start_dir,
                visited,
                start_pos, start_dir,
                entry,
            )
    
            if layout is not None:
                self.build(start_pos, start_dir, layout)
                return
    
        raise RuntimeError(
            f"Grid.random(): could not generate a valid loop after "
            f"{max_attempts} attempts on a {self.cols}×{self.rows} grid. "
            f"Try a larger grid, reduce min_tiles, or increase max_attempts."
        )

    def build(self, init_pos: Position, init_dir: Direction, layout: list[Type]) -> None:
        """
        """
        self.initialized = False
        
        def out_of_bounds(pos: Position) -> bool:
            """Checks if a position is out of bounds of the grid"""
            x, y = pos
            return x < 0 or y < 0 or x >= self.cols or y >= self.rows
        
        def can_loop(pos: Position, direction: Direction) -> bool:
            """Advanced function that verifies if direction will not lead to a incomplete track / loop (i.e. a block)"""
            pass

        curr_pos, curr_dir, curr_rot = init_pos, init_dir, direction_to_rotation(init_dir)

        self._cells[*curr_pos[::-1]] = self.start_cell = prev_tile = Tile(
            Type.START, curr_pos, self.cell_size,
            rotation=curr_rot.value - 90,
            config=self.tile_config
        )

        curr_tile = None
        continuity = {}
        for turn in layout:
            next_dir = get_next_direction(curr_dir, turn)
            next_pos = get_next_position(curr_pos, curr_dir)
            # print(f"Placing tile at {next_pos} with direction {next_dir} from current position {curr_pos} and direction {curr_dir}")
            if out_of_bounds(next_pos):
                print(self._cells)
                raise ValueError(f"Track goes out of bounds at position {next_pos} with direction {curr_dir}.")
            if can_loop(next_pos, next_dir):
                raise ValueError(f"Track leads to a block at position {next_pos} with direction {next_dir}.")

            if turn == Type.START: turn = Type.STRAIGHT
            next_rot = get_next_rotation(curr_rot, turn)
            turn_phase_offset = -90 if turn == Type.LEFT else 90 if turn == Type.RIGHT else 0
            self._cells[*next_pos[::-1]] = curr_tile = Tile(
                turn, next_pos, self.cell_size,
                rotation=next_rot.value - 90 + turn_phase_offset,
                config=self.tile_config
            )
            curr_pos = next_pos
            curr_dir = next_dir
            curr_rot = next_rot
            continuity[prev_tile] = curr_tile
            prev_tile = curr_tile
        if curr_tile is not None:
            continuity[curr_tile] = self.start_cell
        for pos, tile in np.ndenumerate(self._cells):
            if tile not in continuity:
                self._cells[pos] = None
        # print(self._cells)

        # Create empty tile config
        empty_config = TileConfig(params={
            'gravel': 1.0,
            'grass': 0.0,
            'curb': 0.0,
            'road_color': self.tile_config.road_color,
            'grass_color': COLOR_EMPTY,
            'curb_red': self.tile_config.curb_red,
            'curb_white': self.tile_config.curb_white,
            'gravel_color': COLOR_EMPTY,
        })
        for pos, tile in np.ndenumerate(self._cells):
            if tile is None:
                self._cells[pos] = Tile(
                    Type.STRAIGHT, pos[::-1], self.cell_size,
                    rotation=0,
                    config=empty_config
                )
        
        # if not self.verify_track():
        #     raise ValueError("Track is not a fully connected loop.")
        self.cells = self._cells.tolist()
        self.continuity = continuity
        
        # Generate collision masks after grid is fully built
        self._generate_masks()

        self.initialized = True

    def render(self, surface: Surface, origin: tuple[int, int] = (0, 0), verbose: int | bool = False) -> None:
        verbose = 3
        ox, oy = origin
        for y, row in enumerate(self.cells):
            for x, segment in enumerate(row):
                if segment is not None:
                    segment.render(
                        surface, (ox + x * self.cell_size, oy + y * self.cell_size), 
                        verbose=True, # verbose == 1
                    )
        
        if verbose != 0: # verbose >= 2 and surf is not None:
            surfaces = (
                self._surface_road, self._surface_curb, self._surface_grass, self._surface_gravel, 
                self._surface_track, self._surface_boundary,
            )
            if verbose < 0: verbose = min(len(surfaces), max(0, len(surfaces) + verbose))
            else: verbose -= 1
            surf = surfaces[verbose]
            # Render track visualization as semi-transparent yellow overlay
            surf_viz = surf.copy()
            surf_viz.fill((0, 255, 0), special_flags=pygame.BLEND_MULT)
            surface.blit(surf_viz, origin, special_flags=pygame.BLEND_ALPHA_SDL2)
            pass
      
    # ---------- Mask Generation ----------

    def _generate_masks(self) -> None:
        """
        Generate collision masks for all surface types (gravel, grass, curb, road, and track).
        Called after track is fully built. Masks are stored as Mask objects.
        
        Rotated tiles are properly clipped to their cell boundaries to prevent
        mask overflow into adjacent cells and beyond grid boundaries.
        """
        grid_width = self.cols * self.cell_size
        grid_height = self.rows * self.cell_size
        
        # Create surfaces for each layer
        self._surface_gravel = Surface((grid_width, grid_height), pygame.SRCALPHA)
        self._surface_grass = Surface((grid_width, grid_height), pygame.SRCALPHA)
        self._surface_curb = Surface((grid_width, grid_height), pygame.SRCALPHA)
        self._surface_road = Surface((grid_width, grid_height), pygame.SRCALPHA)
        self._surface_track = Surface((grid_width, grid_height), pygame.SRCALPHA)
        
        # Render each tile's layers onto the corresponding surface
        # for y, row in enumerate(self.cells):
        #     for x, tile in enumerate(row):
        for (y, x), tile in np.ndenumerate(self._cells):
            if tile is not None:
                tile_x, tile_y = tile.position_abs
                
                # Create a clipping rect for this tile's cell to prevent rotated overflow
                cell_clip = pygame.Rect(tile_x, tile_y, self.cell_size, self.cell_size)
                
                # Render track (non-empty tiles)
                if not tile.empty:
                    track_surf = Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    track_surf.fill((255, 255, 255, 255))  # Fully opaque white
                    self._surface_track.blit(track_surf, (tile_x, tile_y))
                
                # Render each layer
                gravel_surf = tile._render_layer_mask('gravel')
                # if tile.rotation and tile.type not in (Type.STRAIGHT, Type.START):
                #     gravel_surf = self._clip_rotated_surface(gravel_surf, tile.rotation, self.cell_size)
                self._surface_gravel.blit(gravel_surf, (tile_x, tile_y))
                
                grass_surf = tile._render_layer_mask('grass')
                # if tile.rotation:
                #     grass_surf = self._clip_rotated_surface(grass_surf, tile.rotation, self.cell_size)
                self._surface_grass.blit(grass_surf, (tile_x, tile_y))
                
                curb_surf = tile._render_layer_mask('curb')
                # if tile.rotation:
                #     curb_surf = self._clip_rotated_surface(curb_surf, tile.rotation, self.cell_size)
                self._surface_curb.blit(curb_surf, (tile_x, tile_y))
                
                road_surf = tile._render_layer_mask('road')
                # if tile.rotation:
                #     road_surf = self._clip_rotated_surface(road_surf, tile.rotation, self.cell_size)
                self._surface_road.blit(road_surf, (tile_x, tile_y))

        # Extract raw masks from surfaces
        raw_gravel_mask = pygame.mask.from_surface(self._surface_gravel)
        raw_grass_mask = pygame.mask.from_surface(self._surface_grass)
        raw_curb_mask = pygame.mask.from_surface(self._surface_curb)
        raw_road_mask = pygame.mask.from_surface(self._surface_road)
        
        # Create subtractive masks where each layer excludes all inner layers.
        # This matches the rendering layer order: gravel → grass → curb → road
        # Each mask should represent ONLY that layer, with inner layers removed.
        self.full_road_mask = raw_road_mask  # Road is the innermost, stays as is
        self.full_curb_mask = self._subtract_mask(raw_curb_mask, raw_road_mask)
        self.full_grass_mask = self._subtract_mask(
            self._subtract_mask(raw_grass_mask, raw_curb_mask), 
            raw_road_mask
        )
        self.full_gravel_mask = self._subtract_mask(
            self._subtract_mask(
                self._subtract_mask(raw_gravel_mask, raw_grass_mask),
                raw_curb_mask
            ),
            raw_road_mask
        )
        
        # Recreate visualization surfaces from subtractive masks
        # so they show the actual visible layers (with inner layers removed)
        self._surface_road = self._mask_to_surface(self.full_road_mask) if self.full_road_mask else None
        self._surface_curb = self._mask_to_surface(self.full_curb_mask) if self.full_curb_mask else None
        self._surface_grass = self._mask_to_surface(self.full_grass_mask) if self.full_grass_mask else None
        self._surface_gravel = self._mask_to_surface(self.full_gravel_mask) if self.full_gravel_mask else None
        
        self.full_track_mask = pygame.mask.from_surface(self._surface_track)
        
        # Generate boundary mask (thin 1-pixel tile boundaries)
        self._surface_boundary = Surface((grid_width, grid_height), pygame.SRCALPHA)
        for (y, x), tile in np.ndenumerate(self._cells):
            if tile is not None and not tile.empty:
                tile_x, tile_y = tile.position_abs
                
                # Render boundary mask using tile's method
                boundary_mask_surf = tile._render_boundary_mask()
                
                # # Apply rotation if needed
                # if tile.rotation:
                #     boundary_mask_surf = pygame.transform.rotate(boundary_mask_surf, tile.rotation)
                #     # Calculate center offset for rotated surface to align with tile center
                #     rot_width, rot_height = boundary_mask_surf.get_size()
                #     offset_x = (rot_width - self.cell_size) // 2
                #     offset_y = (rot_height - self.cell_size) // 2
                #     # Adjust tile position for the rotated surface's larger size
                #     blit_x = tile_x - offset_x
                #     blit_y = tile_y - offset_y
                # else:
                blit_x, blit_y = tile_x, tile_y
                
                self._surface_boundary.blit(boundary_mask_surf, (blit_x, blit_y))
        self.full_boundary_mask = pygame.mask.from_surface(self._surface_boundary)
        
        # Generate finish line mask (START tiles)
        self._surface_finish = Surface((grid_width, grid_height), pygame.SRCALPHA)
        for (y, x), tile in np.ndenumerate(self._cells):
            if tile is not None and tile.type == Type.START:
                tile_x, tile_y = tile.position_abs
                
                # Render finish line mask using tile's method
                if tile.type == Type.START:
                    # START tiles are straight, so use straight mask method
                    finish_line_mask_surf = tile._render_finish_line_mask()
                
                # Apply rotation if needed
                if tile.rotation:
                    finish_line_mask_surf = pygame.transform.rotate(finish_line_mask_surf, tile.rotation)
                    # Calculate center offset for rotated surface to align with tile center
                    rot_width, rot_height = finish_line_mask_surf.get_size()
                    offset_x = (rot_width - self.cell_size) // 2
                    offset_y = (rot_height - self.cell_size) // 2
                    # Adjust tile position for the rotated surface's larger size
                    blit_x = tile_x - offset_x
                    blit_y = tile_y - offset_y
                else:
                    blit_x, blit_y = tile_x, tile_y
                
                self._surface_finish.blit(finish_line_mask_surf, (blit_x, blit_y))
        self.full_finish_mask = pygame.mask.from_surface(self._surface_finish)
        
        # Generate grid boundary mask (entire grid area is valid)
        self._surface_grid = Surface((grid_width, grid_height), pygame.SRCALPHA)
        self._surface_grid.fill((255, 255, 255, 255))  # Fully opaque white for entire grid
        self.full_grid_mask = pygame.mask.from_surface(self._surface_grid)

    def _clip_rotated_surface(self, surface: Surface, rotation: int, cell_size: int) -> Surface:
        """
        Rotate a surface and clip it to cell boundaries to prevent overflow.
        
        When pygame.transform.rotate() is called, the resulting surface can be larger
        than the original (e.g., a 100x100 surface rotated 45° becomes ~141x141).
        This causes mask overflow into adjacent cells.
        
        This method:
        1. Rotates the surface
        2. Creates a cell-sized surface
        3. Centers and blits the visible portion only
        4. Returns the clipped result
        
        Args:
            surface: Original surface to rotate
            rotation: Rotation angle in degrees
            cell_size: Cell size in pixels (output dimensions)
            
        Returns:
            Surface clipped to exactly cell_size x cell_size
        """
        # Rotate the surface
        rotated = pygame.transform.rotate(surface, rotation)
        rot_width, rot_height = rotated.get_size()
        
        # Create a cell-sized output surface
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

    def _subtract_mask(self, mask_outer: Mask, mask_inner: Mask) -> Mask:
        """
        Create a subtractive mask where inner mask is erased from outer mask.
        Used for layering collision masks properly.
        
        Args:
            mask_outer: The outer mask to subtract from
            mask_inner: The inner mask to erase
            
        Returns:
            New mask representing outer - inner
        """
        if mask_outer is None or mask_inner is None:
            return mask_outer
        
        # Create a copy of the outer mask
        result = mask_outer.copy()
        # Erase the inner mask from it
        result.erase(mask_inner, (0, 0))
        return result

    def _mask_to_surface(self, mask: Mask) -> Surface:
        """
        Convert a mask to a surface with white pixels where mask is set (for visualization).
        
        Args:
            mask: The mask to convert
            
        Returns:
            Surface with transparent background and white pixels where mask is True
        """
        if mask is None:
            return None
        
        width, height = mask.get_size()
        surf = Surface((width, height), pygame.SRCALPHA)
        
        # Set white pixels where mask is set
        for x in range(width):
            for y in range(height):
                if mask.get_at((x, y)):
                    surf.set_at((x, y), (255, 255, 255, 255))
        
        return surf

    def _get_object_mask(self, width: int, height: int) -> Mask:
        """
        Create a mask for an object given its bounding box dimensions.
        
        Args:
            width: Object width in pixels
            height: Object height in pixels
            
        Returns:
            Mask representing the object
        """
        surf = Surface((width, height), pygame.SRCALPHA)
        surf.fill((255, 255, 255, 255))  # Fully opaque white
        return pygame.mask.from_surface(surf)

    def _check_collision_pixel(self, mask: Mask, x: int, y: int, width: int, height: int) -> bool:
        """
        Check if object mask overlaps with given mask using pixel-level precision.
        
        Args:
            mask: Target Mask to check against
            x: Object x position
            y: Object y position
            width: Object width
            height: Object height
            
        Returns:
            True if there's any overlap, False otherwise
        """
        if mask is None:
            return False
        
        # Clamp to grid boundaries
        obj_x = max(0, min(int(x), self.boundary_limits[0] - 1))
        obj_y = max(0, min(int(y), self.boundary_limits[1] - 1))
        obj_width = min(int(width), self.boundary_limits[0] - obj_x)
        obj_height = min(int(height), self.boundary_limits[1] - obj_y)
        
        if obj_width <= 0 or obj_height <= 0:
            return False
        
        obj_mask = self._get_object_mask(obj_width, obj_height)
        offset = (obj_x, obj_y)
        return mask.overlap(obj_mask, offset) is not None

    def _check_collision_pixel_mask(self, mask0: Mask, mask1: Mask, offset: tuple[int, int]) -> bool:
        """
        Check if object mask overlaps with given mask using pixel-level precision.
        
        Args:
            mask0: First Mask to check
            mask1: Second Mask to check
            offset: Offset tuple (x, y) for the overlap check
        """
        if mask0 is None or mask1 is None:
            return False

        return mask0.overlap(mask1, offset) is not None

    def _check_collision_grid_cell(self, layer_type: str, x: int, y: int, width: int, height: int) -> bool:
        """
        Check if object overlaps with layer using grid-cell-based collision.
        Faster but coarser than pixel-level collision.
        
        Args:
            layer_type: One of 'gravel', 'grass', 'curb', 'road'
            x: Object x position
            y: Object y position
            width: Object width
            height: Object height
            
        Returns:
            True if any grid cell containing object has the specified layer
        """
        # Get grid cell range covered by object bounding box
        min_col = max(0, int(x // self.cell_size))
        max_col = min(self.cols - 1, int((x + width) // self.cell_size))
        min_row = max(0, int(y // self.cell_size))
        max_row = min(self.rows - 1, int((y + height) // self.cell_size))
        
        # Check each affected cell for the layer type
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if row < self.rows and col < self.cols:
                    tile = self.cells[row][col]
                    if tile is None:
                        continue
                    
                    # Simplified check: if layer dimensions are non-zero, layer exists
                    if layer_type == 'gravel' and tile.gravel_px > 0:
                        return True
                    elif layer_type == 'grass' and tile.grass_px > 0:
                        return True
                    elif layer_type == 'curb' and tile.curb_px > 0:
                        return True
                    elif layer_type == 'road' and tile.road_px > 0:
                        return True
        
        return False

    def get_surface_at(self, x: float, y: float) -> Optional[str]:
        """
        Determine which surface type a point is on (if any).
        Useful for direct surface queries.
        
        Args:
            x: Query x position
            y: Query y position
            
        Returns:
            'road', 'curb', 'grass', 'gravel', or None if no surface
        """
        if not self.initialized:
            return None
        
        grid_x, grid_y = int(x), int(y)
        if grid_x < 0 or grid_y < 0 or grid_x >= self.boundary_limits[0] or grid_y >= self.boundary_limits[1]:
            return None
        
        # Check in order of layers (top to bottom in rendering)
        if self.full_road_mask and self.full_road_mask.get_at((grid_x, grid_y)):
            return 'road'
        elif self.full_curb_mask and self.full_curb_mask.get_at((grid_x, grid_y)):
            return 'curb'
        elif self.full_grass_mask and self.full_grass_mask.get_at((grid_x, grid_y)):
            return 'grass'
        elif self.full_gravel_mask and self.full_gravel_mask.get_at((grid_x, grid_y)):
            return 'gravel'
        
        return None

    # ---------- Collision functions ----------

    def on_gravel(self, x: float, y: float, width: float, height: float, mask: Mask = None) -> bool:
        """
        Check whether an object is touching gravel.
        
        Args:
            x: Object x position
            y: Object y position
            width: Object width
            height: Object height
            mask: Optional mask to check against
            
        Returns:
            True if object overlaps with gravel region
        """
        if not self.initialized:
            return False
        
        if self.granularity == 'cell':
            return self._check_collision_grid_cell('gravel', x, y, width, height)
        else:
            if mask is None:
                return self._check_collision_pixel(self.full_gravel_mask, x, y, width, height)
            else:
                return self._check_collision_pixel_mask(self.full_gravel_mask, mask, (round(x), round(y)))

    def on_grass(self, x: float, y: float, width: float, height: float, mask: Mask = None) -> bool:
        """
        Check whether an object is touching grass.
        
        Args:
            x: Object x position
            y: Object y position
            width: Object width
            height: Object height
            mask: Optional mask to check against
            
        Returns:
            True if object overlaps with grass region
        """
        if not self.initialized:
            return False
        
        if self.granularity == 'cell':
            return self._check_collision_grid_cell('grass', x, y, width, height)
        else:
            if mask is None:
                return self._check_collision_pixel(self.full_grass_mask, x, y, width, height)
            else:
                return self._check_collision_pixel_mask(self.full_grass_mask, mask, (round(x), round(y)))

    def on_curb(self, x: float, y: float, width: float, height: float, mask: Mask = None) -> bool:
        """
        Check whether an object is touching curb.
        
        Args:
            x: Object x position
            y: Object y position
            width: Object width
            height: Object height
            mask: Optional mask to check against
            
        Returns:
            True if object overlaps with curb region
        """
        if not self.initialized:
            return False
        
        if self.granularity == 'cell':
            return self._check_collision_grid_cell('curb', x, y, width, height)
        else:
            if mask is None:
                return self._check_collision_pixel(self.full_curb_mask, x, y, width, height)
            else:
                return self._check_collision_pixel_mask(self.full_curb_mask, mask, (round(x), round(y)))

    def on_road(self, x: float, y: float, width: float, height: float, mask: Mask = None) -> bool:
        """
        Check whether an object is touching road.
        
        Args:
            x: Object x position
            y: Object y position
            width: Object width
            height: Object height
            mask: Optional mask to check against
            
        Returns:
            True if object overlaps with road region
        """
        if not self.initialized:
            return False
        
        if self.granularity == 'cell':
            return self._check_collision_grid_cell('road', x, y, width, height)
        else:
            if mask is None:
                return self._check_collision_pixel(self.full_road_mask, x, y, width, height)
            else:
                return self._check_collision_pixel_mask(self.full_road_mask, mask, (round(x), round(y)))

    def on_track(self, x: float, y: float, width: float, height: float, mask: Mask = None) -> bool:
        """
        Check whether an object is on the track (non-empty tiles).
        
        Args:
            x: Object x position
            y: Object y position
            width: Object width
            height: Object height
            granularity: 'pixel' for pixel-level (precise), 'cell' for grid-cell-based (faster)
            mask: Optional mask to check against
        
        Returns:
            True if object overlaps with track region
        """
        if not self.full_track_mask:
            return False
        
        if self.granularity == 'cell':
            # Check if any cell containing the object is a track tile
            min_col = max(0, int(x // self.cell_size))
            max_col = min(self.cols - 1, int((x + width) // self.cell_size))
            min_row = max(0, int(y // self.cell_size))
            max_row = min(self.rows - 1, int((y + height) // self.cell_size))
            
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    if row < self.rows and col < self.cols:
                        tile = self.cells[row][col]
                        if tile is not None and not tile.empty:
                            return True
            return False
        else:
            if mask is None:
                return self._check_collision_pixel(self.full_track_mask, x, y, width, height)
            else:
                return self._check_collision_pixel_mask(self.full_track_mask, mask, (round(x), round(y)))

    def on_finish(self, x: float, y: float, width: float, height: float, mask: Mask = None) -> bool:
        """
        Check whether an object is on the finish line (START tiles).
        
        Args:
            x: Object x position
            y: Object y position
            width: Object width
            height: Object height
            mask: Optional mask to check against
            
        Returns:
            True if object overlaps with finish line region
        """
        if not self.initialized or self.full_finish_mask is None:
            return False
        
        if self.granularity == 'cell':
            # Check if any cell containing the object is a START tile
            min_col = max(0, int(x // self.cell_size))
            max_col = min(self.cols - 1, int((x + width) // self.cell_size))
            min_row = max(0, int(y // self.cell_size))
            max_row = min(self.rows - 1, int((y + height) // self.cell_size))
            
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    if row < self.rows and col < self.cols:
                        tile = self.cells[row][col]
                        if tile is not None and tile.type == Type.START:
                            return True
            return False
        else:
            if mask is None:
                return self._check_collision_pixel(self.full_finish_mask, x, y, width, height)
            else:
                return self._check_collision_pixel_mask(self.full_finish_mask, mask, (round(x), round(y)))

    def on_grid(self, x: float, y: float, width: float, height: float, mask: Mask = None) -> bool:
        """
        Check whether an object is within the grid boundaries.
        Returns True if object is inside the valid grid area.
        
        Args:
            x: Object x position
            y: Object y position
            width: Object width
            height: Object height
            mask: Optional mask to check against
            
        Returns:
            True if object is fully within grid boundaries
        """
        if not self.initialized:
            return True  # No masks generated, assume within bounds
        
        if self.granularity == 'cell':
            # Check if any part of object falls within grid cells
            min_col = max(0, int(x // self.cell_size))
            max_col = min(self.cols - 1, int((x + width) // self.cell_size))
            min_row = max(0, int(y // self.cell_size))
            max_row = min(self.rows - 1, int((y + height) // self.cell_size))
            
            # Object is within grid if all cells are within bounds
            return (
                x >= 0 and 
                x + width <= self.boundary_limits[0] and 
                y >= 0 and 
                y + height <= self.boundary_limits[1]
            )
        else:
            if mask is None:
                return self._check_collision_pixel(self.full_grid_mask, x, y, width, height)
            else:
                return self._check_collision_pixel_mask(self.full_grid_mask, mask, (round(x), round(y)))

    def on_boundary(self, x: float, y: float, width: float, height: float, mask: Mask = None) -> bool:
        """
        Check whether an object is touching the thin 1-pixel tile boundaries.
        
        Args:
            x: Object x position
            y: Object y position
            width: Object width
            height: Object height
            mask: Optional mask to check against
            
        Returns:
            True if object overlaps with boundary region
        """
        if not self.initialized or self.full_boundary_mask is None:
            return False
        
        if self.granularity == 'cell':
            # Check if any cell containing the object has a boundary
            min_col = max(0, int(x // self.cell_size))
            max_col = min(self.cols - 1, int((x + width) // self.cell_size))
            min_row = max(0, int(y // self.cell_size))
            max_row = min(self.rows - 1, int((y + height) // self.cell_size))
            
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    if row < self.rows and col < self.cols:
                        tile = self.cells[row][col]
                        if tile is not None and not tile.empty:
                            return True
            return False
        else:
            if mask is None:
                return self._check_collision_pixel(self.full_boundary_mask, x, y, width, height)
            else:
                return self._check_collision_pixel_mask(self.full_boundary_mask, mask, (round(x), round(y)))
