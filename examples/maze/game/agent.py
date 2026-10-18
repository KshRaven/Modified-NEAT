from collections import deque
from typing import Any

import numpy as np
from numpy import ndarray as Array

from .config import AgentConfig, overwrite_config
from .functional import (
    Direction, CellType, ALL_DIRECTIONS, get_next_position,
)
from .constants import NP_INT, NP_FLOAT
from .grid import Grid
from .player import Players


# ------------------------------------------------------------------
# Numba Helper Functions
# ------------------------------------------------------------------



# 8 principal directions (row, col) deltas, used only for the continuous compact
# sensor readings. Diagonal "sensing" requires BOTH adjacent cardinal walls to be
# open (no cutting through a wall corner), keeping it physically consistent with
# the grid.
_SENSOR_DELTAS: list[tuple[int, int]] = [
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
]

LIDAR_WALL          = 0
LIDAR_PATH          = 1
LIDAR_ALTERNATE     = 2
LIDAR_CHECKPOINT    = 3
LIDAR_START         = 4
LIDAR_GOAL          = 5
LIDAR_TRAP          = 6
LIDAR_DEAD_END      = 7
LIDAR_TYPES         = 8
LIDAR_DISTANCE_BINS = 10


class Agents:
    """
    The dots. Every genome/player lives at an integer (row, col) grid cell —
    there is no continuous/pixel position anywhere in the simulation. Each
    `update()` call resolves to AT MOST one grid-cell step per genome.

    All 4 modes are implemented here (nowhere else):
      config.discrete_states  -> observation encoding (get_state)
      config.discrete_actions -> how `actions` is interpreted (update)
      config.return_grid      -> whole-grid array observation instead of compact features
      config.reset_on_death   -> respawn at start cell on trap death, vs continue in place

    When `discrete_states` is True, `get_state()` returns integer features or grid
    cell IDs (shape (N, *grid_dims) in grid mode, (N, 10) in compact mode), so the
    output is directly usable as an nn.Embedding index. `self.max_options` holds the corresponding
    vocab_size once `reset()` has been called; it's None whenever `discrete_states`
    is False (continuous observations don't have a vocab size).
    """

    def __init__(self, grid: Grid, players: Players, config: AgentConfig | None = None, **params: dict[str, Any]):
        if config is None:
            config = AgentConfig()
        overwrite_config(config, params)
        self.config = config

        self.grid       = grid
        self.players    = players

        self.seq_len: int | None            = config.seq_len
        self.discrete_states: bool          = config.discrete_states
        self.discrete_actions: bool         = config.discrete_actions
        self.return_grid: bool              = config.return_grid
        self.reset_on_death: bool           = config.reset_on_death
        self.action_cutoff: float           = config.action_cutoff
        self.sensor_count: int              = max(1, min(8, config.sensor_count))
        self.sensor_max_cells: int          = config.sensor_max_cells
        self.use_euclidean_distance: bool   = config.use_euclidean_distance
        self.wall_bump_ends_episode: bool   = config.wall_bump_ends_episode

        self.total              = 1
        self.row: Array         = np.zeros((self.total,), dtype=NP_INT)
        self.col: Array         = np.zeros((self.total,), dtype=NP_INT)
        self.last_bumped: Array = np.full((self.total,), False, dtype=bool)
        self.last_moved: Array  = np.full((self.total,), False, dtype=bool)
        self.paths: list[deque] = []  # per-genome trail of (row, col), for rendering only
        self.visited_checkpoints: list[set] = []  # per-genome set of checkpoint cells already touched

        self._base_grid_arr: Array | None = None  # cached static return_grid base layer

        # vocab_size for the combined discrete-state code (nn.Embedding num_embeddings).
        # Only meaningful when discrete_states is True; set for real in reset().
        self.max_options: np.int32 | None = None

    # ======================================================================
    # Reset
    # ======================================================================

    def reset(self, total: int):
        if not self.grid.initialized:
            raise RuntimeError("Grid must be initialized before Agents.reset().")

        self.total = total
        sr, sc = self.grid.start_cell
        self.row = np.full((self.total,), sr, dtype=NP_INT)
        self.col = np.full((self.total,), sc, dtype=NP_INT)
        self.last_bumped = np.full((self.total,), False, dtype=bool)
        self.last_moved = np.full((self.total,), False, dtype=bool)
        self.paths = [deque([(sr, sc)], maxlen=200) for _ in range(self.total)]
        self.visited_checkpoints = [set() for _ in range(self.total)]
        # Genomes that start ON a checkpoint cell shouldn't get free-farmed
        # rewards for it, but they also shouldn't be able to re-earn it later —
        # treat the start cell as already "visited" if it happens to be one.
        if (sr, sc) in self.grid.checkpoint_cells:
            for visited in self.visited_checkpoints:
                visited.add((sr, sc))

        start_dist = float(self.grid.dist_to_goal[sr, sc])
        self.players.reset(total, start_dist)

        if self.return_grid:
            self._base_grid_arr = (
                self.grid.expanded_array_discrete()
                if self.discrete_states else
                self.grid.expanded_array_continuous()
            )

        if self.discrete_states:
            # +1 factor of 2 in the grid case accounts for the stamped-in is_player bit.
            vocab_size = (
                max(int(self._base_grid_arr.max()) + 2, 10)
                if self.return_grid else
                max(LIDAR_DISTANCE_BINS, LIDAR_TYPES)
            )
            self.max_options = np.int32(vocab_size)
        else:
            self.max_options = None

    # ======================================================================
    # Movement resolution
    # ======================================================================

    def _resolve_directions(self, actions: Array) -> list[Direction | None]:
        """Turn the raw action array into a single Direction (or None = no move) per genome."""
        directions: list[Direction | None] = []
        if self.discrete_actions:
            idx = np.asarray(actions).reshape(self.total, -1)[:, 0]
            for i in range(self.total):
                v = int(round(float(idx[i])))
                v = max(0, min(3, v))
                directions.append(Direction(v))
        else:
            # Continuous-action mode uses four unbounded direction scores
            # (UP, RIGHT, DOWN, LEFT), not a bounded 2D movement vector.
            logits = np.asarray(actions).reshape(self.total, -1)[:, :4]
            for i in range(self.total):
                directions.append(Direction(int(np.argmax(logits[i]))))
        return directions

    def update(self, actions: Array, verbose: bool = False):
        # TODO: Numbafy
        active      = self.players.active
        directions  = self._resolve_directions(actions)

        bumped              = np.full((self.total,), False, dtype=bool)
        moved               = np.full((self.total,), False, dtype=bool)
        reached_goal        = np.full((self.total,), False, dtype=bool)
        hit_trap            = np.full((self.total,), False, dtype=bool)
        reached_checkpoint  = np.full((self.total,), False, dtype=bool)
        best_path_checkpoint = np.full((self.total,), False, dtype=bool)
        deviated_path       = np.full((self.total,), False, dtype=bool)
        entered_dead_end    = np.full((self.total,), False, dtype=bool)

        for i in range(self.total):
            if not active[i]:
                continue
            d = directions[i]
            if d is None:
                continue  # deliberately held still this step — no bump, no move

            pos = (int(self.row[i]), int(self.col[i]))
            if self.grid.can_move(pos, d):
                nr, nc = get_next_position(pos, d)
                self.row[i], self.col[i] = nr, nc
                moved[i] = True
                self.paths[i].append((nr, nc))

                cell_type = self.grid.cell((nr, nc)).type
                if cell_type == CellType.GOAL:
                    reached_goal[i] = True
                elif cell_type == CellType.TRAP:
                    hit_trap[i] = True

                if (nr, nc) in self.grid.checkpoint_cells and (nr, nc) not in self.visited_checkpoints[i]:
                    self.visited_checkpoints[i].add((nr, nc))
                    reached_checkpoint[i] = True
                    best_path_checkpoint[i] = (nr, nc) in self.grid.best_path_cells
                if (nr, nc) not in self.grid.best_path_cells:
                    deviated_path[i] = True
                if (nr, nc) not in self.grid.path_cells:
                    entered_dead_end[i] = True
            else:
                bumped[i] = True

        self.last_bumped, self.last_moved = bumped, moved

        dist_to_goal = self.grid.dist_to_goal[self.row, self.col].astype(NP_FLOAT)
        self.players.update(
            dist_to_goal, reached_goal, hit_trap, bumped, moved,
            reached_checkpoint=reached_checkpoint,
            best_path_checkpoint=best_path_checkpoint,
            deviated_path=deviated_path,
            entered_dead_end=entered_dead_end,
        )

        # ── reset_on_death: respawn at start if the genome still has lives left ──
        respawn = hit_trap & (self.players.lives > 0) & self.reset_on_death
        if respawn.any():
            sr, sc = self.grid.start_cell
            self.row[respawn] = sr
            self.col[respawn] = sc
            for i in np.nonzero(respawn)[0]:
                self.paths[i].append((sr, sc))

        self.players.restart()

        if verbose:
            for i in range(self.total):
                if reached_goal[i]:
                    print(f"Genome {i} reached the goal!")
                if hit_trap[i]:
                    print(f"Genome {i} hit a trap (lives left: {self.players.lives[i]})")

    # ======================================================================
    # Observations
    # ======================================================================

    def get_state(self) -> Array:
        if self.return_grid: return self._get_grid_state()
        else:                return self._get_compact_state()

    def _get_grid_state(self) -> Array:
        """One expanded topology array per genome, with its position stamped in."""
        base = self._base_grid_arr
        if self.discrete_states:
            # 0=passage, 1=wall/boundary, 2..5=cell types, 6=player.
            out = np.repeat(base[None, :, :], self.total, axis=0).astype(NP_INT)
            for i in range(self.total):
                out[i, 2 * self.row[i] + 1, 2 * self.col[i] + 1] = len(CellType) + 2
            return out
        else:
            out = np.repeat(base[None, :, :], self.total, axis=0).astype(NP_FLOAT)
            for i in range(self.total):
                out[i, 2 * self.row[i] + 1, 2 * self.col[i] + 1] = 1.0
            return out[:, None, :, :]

    def _cell_lidar_type(self, pos: tuple[int, int]) -> int:
        cell = self.grid.cell(pos)
        if cell.type == CellType.START:
            return LIDAR_START
        if cell.type == CellType.GOAL:
            return LIDAR_GOAL
        if cell.type == CellType.TRAP:
            return LIDAR_TRAP
        if pos in self.grid.checkpoint_cells:
            degree = sum(self.grid.can_move(pos, d) for d in ALL_DIRECTIONS)
            return LIDAR_DEAD_END if degree <= 1 else LIDAR_CHECKPOINT
        return LIDAR_PATH

    def _cast_lidar(self, row: int, col: int, direction: Direction) -> tuple[int, int]:
        """Return (cell distance, target type) for one cardinal beam."""
        pos = (row, col)
        if not self.grid.can_move(pos, direction):
            return 0, LIDAR_WALL

        saw_alternate = False
        distance = 0
        while self.grid.can_move(pos, direction):
            pos = get_next_position(pos, direction)
            distance += 1
            target_type = self._cell_lidar_type(pos)
            if target_type != LIDAR_PATH:
                if target_type in (LIDAR_CHECKPOINT, LIDAR_DEAD_END) and pos not in self.grid.best_path_cells:
                    target_type = LIDAR_ALTERNATE
                return distance, target_type
            if pos not in self.grid.best_path_cells:
                saw_alternate = True

        return distance, LIDAR_ALTERNATE if saw_alternate else LIDAR_DEAD_END

    @staticmethod
    def _distance_bin(distance: int, maximum: int) -> int:
        if distance <= 0:
            return 0
        return min(9, max(1, int(np.ceil(distance / max(1, maximum) * 9))))

    @staticmethod
    def _type_value(type_code: int) -> float:
        return {
            LIDAR_WALL: -1.0, LIDAR_PATH: -0.5, LIDAR_ALTERNATE: -0.25,
            LIDAR_CHECKPOINT: 0.0, LIDAR_START: -1.0, LIDAR_GOAL: 1.0,
            LIDAR_TRAP: 0.5, LIDAR_DEAD_END: 0.25,
        }[type_code]

    @staticmethod
    def _cell_type_value(cell_type: CellType) -> float:
        return {
            CellType.EMPTY: 0.0, CellType.START: -1.0,
            CellType.GOAL: 1.0, CellType.TRAP: 0.5,
        }[cell_type]

    def _get_compact_state(self) -> Array:
        # Fixed layout: four LiDAR distances, four target types, current cell
        # type, and distance-to-goal.
        out = np.zeros((self.total, 10), dtype=NP_INT if self.discrete_states else NP_FLOAT)
        lidar_maximum = max(self.grid.rows, self.grid.cols)
        goal_maximum = self.grid.max_goal_distance_for(self.use_euclidean_distance)
        for i in range(self.total):
            row, col = int(self.row[i]), int(self.col[i])
            for j, direction in enumerate(ALL_DIRECTIONS):
                distance, target_type = self._cast_lidar(row, col, direction)
                if self.discrete_states:
                    out[i, j] = self._distance_bin(distance, lidar_maximum)
                    out[i, 4 + j] = target_type
                else:
                    out[i, j] = (min(distance, lidar_maximum) / lidar_maximum) * 2.0 - 1.0
                    out[i, 4 + j] = self._type_value(target_type)
            cell_type = self.grid.cell((row, col)).type
            out[i, 8] = self._cell_type_value(cell_type) if not self.discrete_states else cell_type.value
            goal_distance = self.grid.goal_distance(
                (row, col), use_euclidean=self.use_euclidean_distance,
            )
            out[i, 9] = (
                self._distance_bin(goal_distance, goal_maximum)
                if self.discrete_states else
                (goal_distance / goal_maximum) * 2.0 - 1.0
            )
        return out
