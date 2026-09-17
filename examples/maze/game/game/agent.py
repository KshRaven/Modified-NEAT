from collections import deque
from typing import Any

import numpy as np
from numpy import ndarray as Array

from .config import AgentConfig, overwrite_config
from .functional import (
    Direction, CellType, ALL_DIRECTIONS, get_next_position, vector_to_direction,
)
from .constants import NP_INT, NP_FLOAT
from .grid import Grid
from .player import Players


# 8 principal directions (row, col) deltas, used only for the continuous compact
# sensor readings. Diagonal "sensing" requires BOTH adjacent cardinal walls to be
# open (no cutting through a wall corner), keeping it physically consistent with
# the grid.
_SENSOR_DELTAS: list[tuple[int, int]] = [
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
]


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
        self.wall_bump_ends_episode: bool   = config.wall_bump_ends_episode

        self.total              = 1
        self.row: Array         = np.zeros((self.total,), dtype=NP_INT)
        self.col: Array         = np.zeros((self.total,), dtype=NP_INT)
        self.last_bumped: Array = np.full((self.total,), False, dtype=bool)
        self.last_moved: Array  = np.full((self.total,), False, dtype=bool)
        self.paths: list[deque] = []  # per-genome trail of (row, col), for rendering only
        self.visited_checkpoints: list[set] = []  # per-genome set of checkpoint cells already touched

        self._base_grid_arr: Array | None = None  # cached static return_grid base layer

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
                self.grid.base_array_discrete() if self.discrete_states
                else self.grid.base_array_continuous()
            )

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
            vecs = np.asarray(actions).reshape(self.total, -1)[:, :2]
            for i in range(self.total):
                dx, dy = float(vecs[i, 0]), float(vecs[i, 1])  # dx=col axis, dy=row axis
                directions.append(vector_to_direction(dx, dy, cutoff=self.action_cutoff))
        return directions

    def update(self, actions: Array, verbose: bool = False):
        active = self.players.active
        directions = self._resolve_directions(actions)

        bumped = np.full((self.total,), False, dtype=bool)
        moved = np.full((self.total,), False, dtype=bool)
        reached_goal = np.full((self.total,), False, dtype=bool)
        hit_trap = np.full((self.total,), False, dtype=bool)
        reached_checkpoint = np.full((self.total,), False, dtype=bool)

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
            else:
                bumped[i] = True

        self.last_bumped, self.last_moved = bumped, moved

        dist_to_goal = self.grid.dist_to_goal[self.row, self.col].astype(NP_FLOAT)
        self.players.update(dist_to_goal, reached_goal, hit_trap, bumped, moved, reached_checkpoint)

        if self.wall_bump_ends_episode:
            self.players.disqualified[bumped & active] = True

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
        """One full grid array PER genome, with that genome's own position stamped in."""
        base = self._base_grid_arr
        rows, cols = self.grid.rows, self.grid.cols

        if self.discrete_states:
            # (N, rows, cols, 3) int: [wall_bitmask, cell_type, is_player_here]
            out = np.zeros((self.total, rows, cols, 3), dtype=NP_INT)
            out[:, :, :, :2] = base[None, :, :, :]
            for i in range(self.total):
                out[i, self.row[i], self.col[i], 2] = 1
        else:
            # (N, rows, cols, 8) float32 in [-1, 1]: base 7 channels + player-occupancy channel
            out = np.full((self.total, rows, cols, 8), -1.0, dtype=NP_FLOAT)
            out[:, :, :, :7] = base[None, :, :, :]
            for i in range(self.total):
                out[i, self.row[i], self.col[i], 7] = 1.0
        return np.permute_dims(out, axes=(0, 3, 1, 2))

    def _cast_sensors(self, row: int, col: int) -> list[float]:
        """Wall-distance sensor readings (in whole cells) along up to 8 principal directions."""
        readings = []
        step = max(1, 8 // self.sensor_count)
        deltas = _SENSOR_DELTAS[::step][: self.sensor_count]
        for dr, dc in deltas:
            r, c = row, col
            dist = 0
            for _ in range(self.sensor_max_cells):
                cardinal_ok = True
                if dr != 0:
                    d_v = Direction.DOWN if dr > 0 else Direction.UP
                    cardinal_ok &= self.grid.can_move((r, c), d_v)
                if dc != 0:
                    d_h = Direction.RIGHT if dc > 0 else Direction.LEFT
                    cardinal_ok &= self.grid.can_move((r, c), d_h)
                if not cardinal_ok:
                    break
                r, c = r + dr, c + dc
                dist += 1
            readings.append(dist / float(self.sensor_max_cells))
        return readings

    def _get_compact_state(self) -> Array:
        rows, cols = self.grid.rows, self.grid.cols
        gr, gc = self.grid.goal_cell
        max_dist = float(self.grid.dist_to_goal.max()) or 1.0

        if self.discrete_states:
            # (N, 8) int: row, col, wall_up, wall_right, wall_down, wall_left,
            #             goal_row_sign(0/1/2), goal_col_sign(0/1/2)  -- all non-negative, Embedding-ready
            out = np.zeros((self.total, 8), dtype=NP_INT)
            for i in range(self.total):
                r, c = int(self.row[i]), int(self.col[i])
                cell = self.grid.cell((r, c))
                out[i, 0] = r
                out[i, 1] = c
                for j, d in enumerate(ALL_DIRECTIONS):
                    out[i, 2 + j] = 1 if cell.walls[d] else 0
                out[i, 6] = int(np.sign(gr - r)) + 1
                out[i, 7] = int(np.sign(gc - c)) + 1
            return out
        else:
            # (N, 5 + sensor_count) float32 in [-1, 1]
            feat_n = 5 + self.sensor_count
            out = np.zeros((self.total, feat_n), dtype=NP_FLOAT)
            for i in range(self.total):
                r, c = int(self.row[i]), int(self.col[i])
                dist = float(self.grid.dist_to_goal[r, c])
                out[i, 0] = (r / max(1, rows - 1)) * 2 - 1
                out[i, 1] = (c / max(1, cols - 1)) * 2 - 1
                out[i, 2] = (dist / max_dist) * 2 - 1
                out[i, 3] = np.clip((gr - r) / max(1, rows - 1), -1.0, 1.0)
                out[i, 4] = np.clip((gc - c) / max(1, cols - 1), -1.0, 1.0)
                out[i, 5:] = np.array(self._cast_sensors(r, c), dtype=NP_FLOAT) * 2 - 1
            return out
