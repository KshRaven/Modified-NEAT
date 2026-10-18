import random
from collections import deque
from typing import Any, Optional

import numpy as np
from numpy import ndarray as Array
from pygame import Surface, draw

from .config import GridConfig, TileConfig, overwrite_config
from .functional import Position, Direction, CellType, ALL_DIRECTIONS, opposite, get_next_position
from .tile import Cell
from .constants import NP_INT, NP_FLOAT


# ------------------------------------------------------------------
# Numba Helper Functions
# ------------------------------------------------------------------



class Grid:
    """
    A simple Cols x Rows grid, used internally as one: the agent always moves
    from one discrete cell to an adjacent discrete cell (e.g. (1,'A') -> (2,'A')).
    There is no continuous/pixel coordinate space anywhere in this class.
    """

    def __init__(
        self,
        config: GridConfig | None = None, tile_config: TileConfig | None = None,
        **params: dict[str, Any],
    ):
        if config is None:
            config = GridConfig()
        if tile_config is not None:
            config.tile = tile_config
        else:
            tile_config = config.tile

        overwrite_config(config, params)
        overwrite_config(tile_config, params)
        self.config = config
        self.tile_config = tile_config

        self.cols, self.rows = tuple(int(dim) for dim in config.grid_size)
        self.cell_size = int(config.cell_size)
        self.width, self.height = self.cols * self.cell_size, self.rows * self.cell_size
        self.static = config.static

        self.cells:  Array[tuple[int, int], Cell] = np.empty((self.rows, self.cols), dtype=object)
        # Expanded topology: cell centers are at odd/odd coordinates.  Values
        # are 0 for passages and 1 for walls/boundaries until cell types are
        # written at the centers.
        self._grid: Array = np.ones((self.rows * 2 + 1, self.cols * 2 + 1), dtype=NP_INT)
        self.start_cell: Position               = (0, 0)
        self.goal_cell: Position                = (0, 0)
        self.trap_cells: set[Position]          = set()
        self.dist_to_goal: Array | None         = None      # (rows, cols) int, BFS steps to goal
        self.euclidean_to_goal: Array | None    = None      # (rows, cols) float
        self.max_goal_distance: float           = 1.0
        self.best_path: list[Position]          = []        # shortest start->goal path (cells, in order)
        self.all_paths: list[list[Position]]    = []        # shortest path first
        self.path_cells: set[Position]          = set()     # cells occurring in any mapped path
        self.best_path_cells: set[Position]     = set()
        self.checkpoint_cells: set[Position]    = set()     # bends, junctions, endpoints, traps
        self.checkpoint_order: list[Position]   = []

        self.initialized = False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def cell(self, pos: Position) -> Cell:
        r, c = pos
        return self.cells[r, c]

    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_move(self, pos: Position, direction: Direction) -> bool:
        """True iff there is no wall blocking `direction` from `pos` (and the neighbor is in-bounds)."""
        if not self.in_bounds(pos):
            return False
        nxt = get_next_position(pos, direction)
        if not self.in_bounds(nxt):
            return False
        return not self.cell(pos).walls[direction]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def randomize(
            self, seed: Optional[int] = None, trap_ratio: float = 0.35,
            goal_region_ratio: float = 0.34, braid_ratio: float = 0.08,
            checkpoints: int = 10
    ) -> None:
        """(Re-)generate a random maze: perfect maze via randomized DFS, optional
        braiding for loops, a goal placed in the center region, and traps at
        a subset of remaining dead ends."""
        rng = random.Random(seed)
        self.initialized = False

        self.cells = np.empty((self.rows, self.cols), dtype=object)
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r, c] = Cell((r, c), self.cell_size, config=self.tile_config)

        self._carve_maze(rng)
        if braid_ratio > 0:
            self._braid(rng, braid_ratio)

        self.start_cell = self._random_edge_cell(rng)
        self.goal_cell = self._random_center_cell(rng, goal_region_ratio, exclude={self.start_cell})

        self.cell(self.start_cell).type = CellType.START
        self.cell(self.goal_cell).type = CellType.GOAL

        self._place_traps(rng, trap_ratio)
        self._compute_distances()
        self._compute_paths()
        self._place_checkpoints(checkpoints)
        self._build_expanded_grid()

        self.initialized = True

    def _carve_maze(self, rng: random.Random) -> None:
        """Randomized depth-first-search backtracker -> perfect (tree) maze."""
        start = (rng.randrange(self.rows), rng.randrange(self.cols))
        stack = [start]
        self.cell(start).visited = True

        while stack:
            pos = stack[-1]
            neighbors = []
            for d in ALL_DIRECTIONS:
                nxt = get_next_position(pos, d)
                if self.in_bounds(nxt) and not self.cell(nxt).visited:
                    neighbors.append((d, nxt))

            if not neighbors:
                stack.pop()
                continue

            d, nxt = rng.choice(neighbors)
            self.cell(pos).walls[d] = False
            self.cell(nxt).walls[opposite(d)] = False
            self.cell(nxt).visited = True
            stack.append(nxt)

    def _braid(self, rng: random.Random, ratio: float) -> None:
        """Knock down a fraction of remaining walls between dead ends to add loops/branches,
        so the maze isn't a single guaranteed path (keeps the task from being trivially solvable
        by pure wall-following)."""
        dead_ends = [(r, c) for r in range(self.rows) for c in range(self.cols)
                     if self.cell((r, c)).is_dead_end]
        rng.shuffle(dead_ends)
        n = int(len(dead_ends) * ratio)
        for pos in dead_ends[:n]:
            walled = [d for d in ALL_DIRECTIONS if self.cell(pos).walls[d]]
            rng.shuffle(walled)
            for d in walled:
                nxt = get_next_position(pos, d)
                if self.in_bounds(nxt):
                    self.cell(pos).walls[d] = False
                    self.cell(nxt).walls[opposite(d)] = False
                    break

    def _random_edge_cell(self, rng: random.Random) -> Position:
        edge = rng.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            return (0, rng.randrange(self.cols))
        if edge == "bottom":
            return (self.rows - 1, rng.randrange(self.cols))
        if edge == "left":
            return (rng.randrange(self.rows), 0)
        return (rng.randrange(self.rows), self.cols - 1)

    def _random_center_cell(self, rng: random.Random, region_ratio: float, exclude: set[Position]) -> Position:
        """Pick a cell within the central `region_ratio` fraction of the grid (both axes),
        so the genome can't just hug a wall to the far corner and win by accident."""
        region_ratio = min(max(region_ratio, 0.05), 1.0)
        row_margin = int(self.rows * (1 - region_ratio) / 2)
        col_margin = int(self.cols * (1 - region_ratio) / 2)
        row_lo, row_hi = row_margin, max(row_margin, self.rows - 1 - row_margin)
        col_lo, col_hi = col_margin, max(col_margin, self.cols - 1 - col_margin)

        candidates = [
            (r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)
            if (r, c) not in exclude
        ]
        if not candidates:
            candidates = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in exclude]
        return rng.choice(candidates)

    def _place_traps(self, rng: random.Random, trap_ratio: float) -> None:
        self.trap_cells = set()
        dead_ends = [
            (r, c) for r in range(self.rows) for c in range(self.cols)
            if self.cell((r, c)).is_dead_end and (r, c) not in (self.start_cell, self.goal_cell)
        ]
        rng.shuffle(dead_ends)
        n = int(len(dead_ends) * trap_ratio)
        for pos in dead_ends[:n]:
            self.cell(pos).type = CellType.TRAP
            self.trap_cells.add(pos)

    def _compute_distances(self) -> None:
        """BFS shortest-path (in cells) from every cell to the goal cell — used for
        potential-based reward shaping."""
        dist = np.full((self.rows, self.cols), -1, dtype=NP_INT)
        q = deque([self.goal_cell])
        dist[self.goal_cell] = 0
        while q:
            pos = q.popleft()
            for d in ALL_DIRECTIONS:
                if self.can_move(pos, d):
                    nxt = get_next_position(pos, d)
                    if dist[nxt] == -1:
                        dist[nxt] = dist[pos] + 1
                        q.append(nxt)
        # Any unreachable cell (shouldn't happen in a fully-connected maze) gets the max distance.
        max_d = int(dist.max()) if dist.max() >= 0 else 1
        dist[dist == -1] = max_d + 1
        self.dist_to_goal = dist
        goal_row, goal_col = self.goal_cell
        rows, cols = np.indices((self.rows, self.cols), dtype=NP_FLOAT)
        self.euclidean_to_goal = np.hypot(rows - goal_row, cols - goal_col).astype(NP_FLOAT)
        self.max_goal_distance = float(self.dist_to_goal.max()) or 1.0

    def goal_distance(self, pos: Position, use_euclidean: bool = False) -> float:
        """Return maze-aware BFS distance or straight-line distance to goal."""
        r, c = pos
        return float(self.euclidean_to_goal[r, c] if use_euclidean else self.dist_to_goal[r, c])

    def max_goal_distance_for(self, use_euclidean: bool = False) -> float:
        """Return the maximum distance used to normalize the selected metric."""
        if use_euclidean:
            return float(self.euclidean_to_goal.max()) or 1.0
        return self.max_goal_distance

    def _compute_paths(self) -> None:
        """Enumerate simple start->goal paths, with the shortest path first."""
        paths: list[list[Position]] = []
        limit = max(1, int(self.config.max_paths))

        def visit(pos: Position, path: list[Position], visited: set[Position]) -> None:
            if len(paths) >= limit:
                return
            if pos == self.goal_cell:
                paths.append(path.copy())
                return
            for direction in ALL_DIRECTIONS:
                if not self.can_move(pos, direction):
                    continue
                nxt = get_next_position(pos, direction)
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    visit(nxt, path, visited)
                    path.pop()
                    visited.remove(nxt)

        visit(self.start_cell, [self.start_cell], {self.start_cell})
        paths.sort(key=len)
        self.all_paths = paths
        self.best_path = paths[0] if paths else [self.start_cell]
        self.best_path_cells = set(self.best_path)
        self.path_cells = set().union(*(set(path) for path in paths)) if paths else set()

    def _compute_best_path(self) -> None:
        """Compatibility wrapper for callers using the old private method."""
        self._compute_paths()

    def _place_checkpoints(self, n: int) -> None:
        """Mark every bend, junction, endpoint, and trap as a checkpoint.

        The explicit count remains as a fallback for long straight corridors,
        preserving the old reward-density control without hiding navigation
        structure from the LiDAR state.
        """
        self.checkpoint_cells = set()
        self.checkpoint_order = []
        for r in range(self.rows):
            for c in range(self.cols):
                pos = (r, c)
                degree = sum(self.can_move(pos, direction) for direction in ALL_DIRECTIONS)
                open_dirs = [d for d in ALL_DIRECTIONS if self.can_move(pos, d)]
                bend = degree == 2 and open_dirs[0] != opposite(open_dirs[1])
                if degree != 2 or bend or self.cell(pos).type == CellType.TRAP:
                    self.checkpoint_cells.add(pos)

        interior = [pos for pos in self.best_path[1:-1] if pos not in self.checkpoint_cells]
        n = min(max(0, int(n)), len(interior))
        if n:
            idxs = np.linspace(0, len(interior) - 1, num=n)
            self.checkpoint_cells.update(interior[int(round(i))] for i in idxs)

        self.checkpoint_order = [pos for pos in self.best_path if pos in self.checkpoint_cells]

    def _build_expanded_grid(self) -> None:
        """Build the wall/path topology around typed odd/odd cell centers."""
        expanded = np.ones((self.rows * 2 + 1, self.cols * 2 + 1), dtype=NP_INT)
        for r in range(self.rows):
            for c in range(self.cols):
                expanded[2 * r + 1, 2 * c + 1] = self.cell((r, c)).type.value + 2
                for direction in ALL_DIRECTIONS:
                    if self.can_move((r, c), direction):
                        nr, nc = get_next_position((r, c), direction)
                        expanded[r + nr + 1, c + nc + 1] = 0
        self._grid = expanded

    # ------------------------------------------------------------------
    # return_grid array builders
    # ------------------------------------------------------------------

    def base_array_continuous(self) -> Array:
        """(rows, cols, 7) float in [-1, 1]: wall_up, wall_right, wall_down, wall_left,
        is_start, is_goal, is_trap. This is the STATIC base — no player position baked in
        (Agents.get_state() stamps that in per-genome)."""
        arr = np.full((self.rows, self.cols, 7), -1.0, dtype=NP_FLOAT)
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.cells[r, c]
                for i, d in enumerate(ALL_DIRECTIONS):
                    arr[r, c, i] = 1.0 if cell.walls[d] else -1.0
                arr[r, c, 4] = 1.0 if cell.type == CellType.START else -1.0
                arr[r, c, 5] = 1.0 if cell.type == CellType.GOAL else -1.0
                arr[r, c, 6] = 1.0 if cell.type == CellType.TRAP else -1.0
        return arr

    # Number of distinct wall configurations per cell: 4 independent wall bits -> 2**4.
    WALL_BITMASK_OPTIONS: int = 16

    @property
    def cell_type_options(self) -> int:
        """Number of distinct CellType values (0=EMPTY,1=START,2=GOAL,3=TRAP, ...)."""
        return len(CellType)

    @property
    def discrete_cell_vocab_size(self) -> int:
        """Number of distinct static (wall_bitmask, cell_type) codes a single cell can
        take, i.e. the vocab_size for an nn.Embedding over `base_array_discrete()`."""
        return self.WALL_BITMASK_OPTIONS * self.cell_type_options

    @property
    def compact_discrete_vocab_size(self) -> int:
        """Number of distinct combined codes the compact discrete per-genome state
        (row, col, 4 wall bits, 2 goal signs) can take, i.e. the vocab_size for an
        nn.Embedding over `Agents._get_compact_state()`'s discrete output."""
        return self.rows * self.cols * self.WALL_BITMASK_OPTIONS * 3 * 3

    def base_array_discrete(self) -> Array:
        """(rows, cols) int: a single combined code per cell —
        code = wall_bitmask * cell_type_options + cell_type_code, where
        wall_bitmask is in [0, 15] (bit0=UP,bit1=RIGHT,bit2=DOWN,bit3=LEFT) and
        cell_type_code is in [0, cell_type_options). One scalar per cell, directly
        usable as an nn.Embedding index (vocab_size = discrete_cell_vocab_size)."""
        arr = np.zeros((self.rows, self.cols), dtype=NP_INT)
        n_types = self.cell_type_options
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.cells[r, c]
                bitmask = 0
                for i, d in enumerate(ALL_DIRECTIONS):
                    if cell.walls[d]:
                        bitmask |= (1 << i)
                arr[r, c] = bitmask * n_types + cell.type.value
        return arr

    def expanded_array_discrete(self) -> Array:
        """Return the expanded integer topology, including typed cell centers."""
        return self._grid.copy()

    def expanded_array_continuous(self) -> Array:
        """Return the expanded topology normalized to [-1, 1]."""
        out = np.full(self._grid.shape, -1.0, dtype=NP_FLOAT)
        out[self._grid == 0] = -0.5  # passage
        out[self._grid == 2] = 0.0   # normal cell
        out[self._grid == 3] = -1.0  # start
        out[self._grid == 4] = 1.0   # goal
        out[self._grid == 5] = 0.5   # trap
        return out

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, surface: Surface, origin: tuple[int, int] = (0, 0)) -> None:
        ox, oy = origin
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r, c].render(surface)
