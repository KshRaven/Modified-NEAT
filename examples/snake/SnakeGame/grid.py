
from .functional import INT, FLOAT, BOOL, ARRAY_1D, Direction, GridEnum
from .player import Players, PLAYERS_TYPE

from numba import prange, njit
from numba.experimental import jitclass
from numpy import ndarray as CPUArray

import pygame
import math
import random
import numpy as np


BLOB = np.uint16

def get_positions(mask: CPUArray[tuple[int, ...], bool]):
    coordinates: dict[int, tuple[int, ...]] = {coord[0]: coord[1:] for coord in zip(*np.nonzero(mask))}
    positions = [coordinates.get(p) for p in prange(len(mask))]
    return positions

# @njit(nogil=True)
def get_free_positions(mask: CPUArray):
    # Ensure the mask is 3-dimensional. shape(players, width, height)
    if mask.ndim != 3:
        raise ValueError(f"Invalid mask shape")
    # Ensure all masks are the same
    if np.any(~np.all(mask == mask[[0]], axis=(1, 2))):
        raise ValueError(f"All players' masks must be the same")
    # Get indices where mask is True
    positions: list[tuple[int, ...]] = [tuple(index.tolist()) for index in np.argwhere(mask[0])]
    return positions

def get_new_positions(positions: list[tuple[int, int] | None], directions: CPUArray[tuple[int], int] | list[int]):
    new_positions: list[tuple[int, int] | None] = []
    for p in prange(len(positions)):
        pos = positions[p]
        if pos is not None:
            x, y = pos
            drc = directions[p]
            x, y = x + round(np.cos(drc * 0.5 * np.pi)), y + round(np.sin(drc * 0.5 * np.pi))
            new_positions.append((x, y))
        else:
            new_positions.append(None)
    return new_positions

def set_positions(array: CPUArray, positions: list[tuple[int, int] | None], value: int, mask: CPUArray | None,
                  validation: bool = False):
    players = len(array)
    if players != len(positions):
        raise ValueError(f"Invalid number of positions")
    for p in prange(players):
        fill = True
        if mask is not None:
            fill &= mask[p].item()
        coordinates = positions[p]
        if fill and coordinates is not None:
            x, y = coordinates
            if validation and array[p, x, y] != GridEnum.Empty.value:
                raise ValueError(f"Position must be empty")
            array[p, x, y] = value

def check_collision(array: CPUArray, mask: CPUArray, value: int) -> CPUArray[tuple[int], bool]:
    collisions: CPUArray[tuple[int], int] = np.count_nonzero((array == value) & mask, axis=(1, 2))
    assert np.all(collisions < 2)
    return collisions.astype(bool)

def get_distance(positions0: list[tuple[int, int] | None], positions1: list[tuple[int, int] | None], split=False):
    if len(positions0) != len(positions1):
        raise ValueError(f"Invalid position lists")
    total = len(positions0)
    distance: CPUArray[tuple[int]] | CPUArray[tuple[int, int]] = \
        np.full((total,) if not split else (total, 2), 0., dtype=float)
    for p in prange(len(positions0)):
        pos0, pos1 = positions0[p], positions1[p]
        if pos0 is None or pos1 is None:
            if not split:
                distance[p] = np.nan
            else:
                distance[p, :] = np.nan
        else:
            x0, y0 = pos0
            x1, y1 = pos1
            if not split:
                distance[p] = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            else:
                distance[p, 0] = x1 - x0
                distance[p, 1] = y1 - y0
    return distance if not split else (distance[..., 0], distance[..., 1])

# @jitclass([
#     ('MAX_VEL', FLOAT),
#     ('RADIUS', FLOAT),
#     ('MAX_DRAW', INT),
#     ('LAUNCH_ANGLE', INT),
#     ('players', PLAYERS_TYPE),
#     ('total', INT),
#     ('window_width', INT),
#     ('window_height', INT),
#     ('original_x', INT),
#     ('original_y', INT),
#     ('x', ARRAY_1D),
#     ('y', ARRAY_1D),
#     ('x_vel', ARRAY_1D),
#     ('y_vel', ARRAY_1D),
# ])
class Grid(object):
    def __init__(
            self, players: Players,
            shape: tuple[int, int],
            init_direction: int | None = None,
            init_length: int = 1,
    ):
        if init_direction is None:
            init_direction = random.randint(0, 3)
        init_direction = min(max(init_direction % 4, 0), 3)
        init_length = max(init_length, 1)
        for size in shape:
            assert size > init_length * 2
        self.INIT_DIR = int(init_direction)
        self.INIT_LEN = init_length

        self.players = players
        self.total = self.players.total
        self.shape = shape
        self.width, self.height = shape
        self.max_distance = math.sqrt(self.width ** 2 + self.height ** 2)
        self.init_x, self.init_y = self.width // 2 + 1, self.height // 2 + 1
        self.factor_x = round(np.cos(0.5 * math.pi * self.INIT_DIR))
        self.factor_y = round(np.sin(0.5 * math.pi * self.INIT_DIR))
        self.grid = np.full(
            (self.total, self.width+2, self.height+2),
            fill_value=GridEnum.Empty.value, dtype=np.int64
        )
        self.food_locations: list[tuple[int, int]] | None = None
        self.boundary_mask = np.full_like(self.grid, fill_value=False, dtype=bool)
        self.head_mask = np.full_like(self.grid, fill_value=False, dtype=bool)
        self.direction: CPUArray[tuple[int], int] = np.full(self.total, fill_value=self.INIT_DIR, dtype=int)
        self.snake_length = np.full((self.total,), fill_value=self.INIT_LEN, dtype=int)
        self.hiatus = np.full((self.total,), fill_value=0, dtype=int)
        self.prev_distance: CPUArray = np.full((self.total,), fill_value=1.0, dtype=float)

        self.reset()

    @property
    def filled(self) -> CPUArray:
        return self.grid != GridEnum.Empty.value

    @property
    def empty(self) -> CPUArray:
        return self.grid == GridEnum.Empty.value

    @property
    def completed(self) -> CPUArray[tuple[int], bool]:
        return np.all(self.grid != GridEnum.Empty.value, axis=(1, 2))

    def _set_food(self, mask: CPUArray[tuple[int], bool] | None):
        locations_total = len(self.food_locations)
        next_indices = [
            self.snake_length[p] - self.INIT_LEN
            for p in range(self.total)
        ]
        next_positions = [
            self.food_locations[index] if index < locations_total else None
            for index in next_indices
        ]
        set_positions(self.grid, next_positions, GridEnum.Food.value, mask, validation=True)

    def reset(self, complete=True):
        restart = self.players.to_restart()
        complete |= self.players.total != self.total # or np.all(restart == True).item()

        # Recreate masks if necessary and food locations
        if complete:
            self.total = self.players.total
            self.grid = np.full(
                (self.total, self.width+2, self.height+2),
                fill_value=GridEnum.Empty.value, dtype=np.int64
            )
            self.direction: CPUArray[tuple[int], int] = np.full(self.total, fill_value=self.INIT_DIR, dtype=int)
            self.snake_length = np.full((self.total,), fill_value=self.INIT_LEN, dtype=int)
            self.hiatus = np.full((self.total,), fill_value=0, dtype=int)
            self.prev_distance = np.full((self.total,), fill_value=1.0, dtype=float)

            # Boundary mask
            self.boundary_mask = np.full_like(self.grid, fill_value=False, dtype=bool)
            player_indices = np.arange(self.players.total)
            limits = [list(range(dim_size)) for dim_size in self.grid.shape[1:]]
            for dim in prange(2):
                indices = [
                    np.array(indices if dim_idx == dim else [0, -1])
                    for dim_idx, indices in enumerate(limits)
                ]
                self.boundary_mask[np.ix_(player_indices, *indices)] = True

            # Snake head mask
            self.head_mask = np.full_like(self.grid, fill_value=False, dtype=bool)
            self.head_mask[:, self.init_x, self.init_y] = True

        # Reset or restart the blob and boundary locations
        if complete:
            self.grid[:]                    = GridEnum.Empty.value
            self.grid[self.boundary_mask]   = GridEnum.Boundary.value
            self.grid[self.head_mask]       = GridEnum.SnakeHead.value
            self.snake_length[:]            = self.INIT_LEN
            self.hiatus[:]                  = 0
            self.prev_distance[:]           = 1.0 # self.max_distance
        else:
            _restart = restart[:, None, None] & np.ones_like(self.grid, dtype=bool)
            self.grid[_restart]                      = GridEnum.Empty.value
            self.grid[_restart & self.boundary_mask] = GridEnum.Boundary.value
            self.grid[_restart & self.head_mask]     = GridEnum.SnakeHead.value
            self.snake_length[restart]               = self.INIT_LEN
            self.hiatus[restart]                     = 0
            self.prev_distance[restart]              = 1.0 # self.max_distance
        for i in prange(self.INIT_LEN):
            pos_x, pos_y = int(self.init_x - (self.factor_x * (i+1))), int(self.init_y - (self.factor_y * (i+1)))
            if complete:
                self.grid[:, pos_x, pos_y] = GridEnum.SnakeHead.value + 50 + i
            else:
                self.grid[restart, pos_x, pos_y] = GridEnum.SnakeHead.value + 50 + i
        if self.food_locations is None:
            self.food_locations = get_free_positions(self.empty)
            random.shuffle(self.food_locations)
        self._set_food(None if complete else restart)

    def restart(self):
        self.reset(complete=False)

    def get_body_mask(self, index: int) -> CPUArray[tuple[int, int, int], bool]:
        if index < 0:
            index = GridEnum.SnakeHead.value + 50 + self.snake_length + index
            index[index == (GridEnum.SnakeHead.value + 50 - 1)] = GridEnum.SnakeHead.value
            index = index[:, None, None]
        else:
            index = GridEnum.SnakeHead.value + (0 if index == 0 else 50 - 1) + index
        # TODO: Might want to add check to ensure limits of negative indices
        if np.any(index < GridEnum.SnakeHead.value):
            raise ValueError(f"A player has a snake with no body or index not within limits")
        return self.grid == index

    def move(self, action: CPUArray[tuple[int], int], convolutional=True):
        grid, prev_grid = self.grid, self.grid.copy()
        movement = action - 1 # Should be integer in interval [-1, +1]
        head, tail = self.get_body_mask(0), self.get_body_mask(-1)
        head_positions, tail_positions = get_positions(head), get_positions(tail)
        self.direction -= movement
        self.direction[self.direction == -1] = 3
        self.direction[self.direction == +4] = 0
        new_head_positions = get_new_positions(head_positions, self.direction)
        # TODO: Make the snake move even when inactive
        # active = self.players.active & (~self.players.completed)
        set_positions(grid, new_head_positions, GridEnum.SnakeHead.value, None, validation=False) # Move Head. NOTE: active -> None

        food_mask = prev_grid == GridEnum.Food.value
        food_collision = check_collision(grid, food_mask, GridEnum.SnakeHead.value) # NOTE: The check only counts collisions so having 2 head locations won't raise error
        # _active = active[:, None, None]
        old_body_mask = (prev_grid >= GridEnum.SnakeHead.value) # & _active
        grid[old_body_mask] += 1 # Move the rest of the body
        grid[head] = GridEnum.SnakeHead.value + 50 # Ensure first body part location
        grid[tail & (~food_collision)[:, None, None]] = GridEnum.Empty.value # Extend snake on eating food. & _active
        self.snake_length[food_collision] += 1
        self.hiatus[food_collision] = 0
        self.hiatus[~food_collision] += 1
        old_body_mask[tail] = False # The tail has moved so snake can't collide with it
        body_collision = check_collision(grid, old_body_mask, GridEnum.SnakeHead.value)
        boundary_mask = self.boundary_mask # & _active
        wall_collision = check_collision(grid, boundary_mask, GridEnum.SnakeHead.value)
        self._set_food(food_collision) # Place new food when eaten
        new_food_positions = get_positions(grid == GridEnum.Food.value)

        distances = get_distance(new_head_positions, new_food_positions) / self.max_distance
        distances[np.isnan(distances)] = 1.0 # self.max_distance
        moved_closer = (self.prev_distance - distances) > 0
        self.prev_distance = distances.copy()
        self.players.update(food_collision, wall_collision, body_collision, self.completed,
                            distances, self.hiatus, moved_closer
                            ) # , True)

        # TODO: Restart immediately after player update because of getting the next state
        self.restart()

        if not convolutional:
            dist_x, dist_y = get_distance(new_head_positions, new_food_positions, True)
            dist_x[np.isnan(dist_x)] = self.width
            dist_y[np.isnan(dist_y)] = self.height
            dist_x, dist_y = dist_x / self.width, dist_y / self.height
            dir_x, dir_y = np.cos(0.5 * np.pi * self.direction).round(), np.sin(0.5 * np.pi * self.direction).round()
            scalars = [dist_x, dist_y, dir_x, dir_y]

            danger = []
            # just_died = wall_collision | body_collision
            # not_just_died = ~just_died
            # active &= not_just_died
            # _active = active[:, None, None]
            pseudo_body_mask = (grid >= GridEnum.SnakeHead.value) # & _active
            for t in range(3):
                temp_grid = grid.copy()
                direction = self.direction.copy() - (t-1)
                direction[direction == -1] = 3
                direction[direction == +4] = 0
                pseudo_head_positions = get_new_positions(new_head_positions, direction)
                set_positions(temp_grid, pseudo_head_positions, GridEnum.SnakeHead.value, None, validation=False) # NOTE: active -> not_just_died -> None
                temp_grid[pseudo_body_mask] += 1
                pseudo_body_mask[self.get_body_mask(-1)] = False
                _body_collision = check_collision(temp_grid, pseudo_body_mask, GridEnum.SnakeHead.value)
                _wall_collision = check_collision(temp_grid, boundary_mask, GridEnum.SnakeHead.value)
                dir_danger = (_body_collision | _wall_collision).astype(float)
                # dir_danger[just_died] = -1
                danger.append(dir_danger)

            result = scalars + danger
            return result
        else:
            return (self.grid - GridEnum.Food.value) / GridEnum.Food.value * 2


if __name__ == "__main__":
    from functional import Direction
    import random

    test_players = Players(lives=5)
    test_grid = Grid(test_players, (5, 5), Direction.RIGHT.value, 2)

    test_players.reset(1)
    test_grid.reset()
    print(f"GRID =>\n{test_grid.grid[0]}")

    print(test_players)
    print(f"active: {test_players.active}")

    for x in range(10):
        print(f"-----{x}-----")
        move = np.full((test_players.total,), x % 2 * 2, dtype=int) # np.random.randint(0, 3, (test_players.total,))
        res = test_grid.move(move)
        state = np.stack(res, axis=-1)
        print(f"GRID =>{move}\n{test_grid.grid[0]}\n{state},{test_players.lives[0]}")
        test_grid.restart()
        test_players.restart()

    for i in range(test_players.total):
        test_players.lives[i] = test_players.lives_total
    test_grid.restart()
    for _ in range(30):
        move = np.random.randint(0, 3, (test_players.total,))
        test_grid.move(move)
        if test_players.scores[0] > 0:
            print(f"GRID =>{move}\n{test_grid.grid[0]}")
        test_grid.restart()
        test_players.restart()
    print(f"GRID =>\n{test_grid.grid[0]}")
    print(test_players.completed)

    pass