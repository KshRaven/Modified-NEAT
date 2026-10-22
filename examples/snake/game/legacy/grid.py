
from .functional import INT, FLOAT, BOOL, ARRAY_1D, Direction, GridEnum
from .player import Players, PLAYERS_TYPE

from numba import prange, njit
from numba.experimental import jitclass
from numpy import ndarray as CPUArray
from typing import Iterable

import pygame
import math
import random
import numpy as np


BLOB = np.uint16

def get_positions(mask: CPUArray[tuple[int, ...], bool]):
    coordinates: dict[int, tuple[int, ...]] = {coord[0]: coord[1:] for coord in zip(*np.nonzero(mask))}
    positions = [coordinates.get(p) for p in prange(len(mask))]
    return positions

@njit(nogil=True)
def get_free_positions(mask: CPUArray, independent=True):
    # Ensure the mask is 3-dimensional. shape(players, width, height)
    if mask.ndim != 3:
        raise ValueError(f"Invalid mask shape")
    if independent:
        players_total = mask.shape[0]
        positions: list[CPUArray[tuple[int], np.dtype[int]] | None] = []
        for p in prange(players_total):
            # Get indices where mask is True for this player
            ys, xs = np.where(mask[p])

            if len(xs) == 0:
                # no available positions
                positions.append(None)
            else:
                # pick a random available index
                idx = np.random.randint(len(xs))
                positions.append(np.array([ys[idx], xs[idx]], np.int64))
    else:
        # # Ensure all masks are the same
        # if np.all(mask == np.expand_dims(mask[0], 0)):
        #     raise ValueError(f"All players' masks must be the same")
        # # Get indices where mask is True
        # positions = [index.astype(np.int64) for index in np.argwhere(mask[0])]
        raise RuntimeError()
    return positions

def get_new_positions(positions: list[Iterable[int] | None], directions: CPUArray[tuple[int], int] | list[int]):
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
        try:
            if fill and coordinates is not None:
                x, y = coordinates
                if validation and array[p, x, y] != GridEnum.Empty.value:
                    raise ValueError(f"Position must be empty. Found value '{array[p, x, y]}'")
                array[p, x, y] = value
        except IndexError as e:
            return p, e
    return None

def check_collision(array: CPUArray, mask: CPUArray, value: int) -> CPUArray[tuple[int], bool]:
    collisions: CPUArray[tuple[int], int] = np.count_nonzero((array == value) & mask, axis=(1, 2))
    assert np.all(collisions < 2)
    return collisions.astype(bool)

def get_rel_distance(positions0: list[tuple[int, int] | None], positions1: list[tuple[int, int] | None], split=False):
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

@njit
def calc_collision_distance(player_grid_mask: CPUArray, index: int, axis: int, position: int, velocity: int):
    assert player_grid_mask.ndim == 2
    if velocity != 0:
        array = np.take(player_grid_mask, index, axis=axis)
        array = array[(position + velocity)::velocity]
        if np.any(array):
            return np.argmax(array) + 1
    return 0.

def get_lim_distance(grid_mask: CPUArray, positions: list[tuple[int, int] | None], velocities: CPUArray):
    total = len(grid_mask)
    if total != len(positions) or total != len(velocities):
        raise ValueError(f"Invalid player counts")
    distance: CPUArray[tuple[int], float] = np.full((total,), 0., dtype=np.float64)
    for p in prange(total):
        position = positions[p]
        if position is None:
            # distance[p] = np.nan
            raise ValueError("Cannot have a missing coordinate when calculating limit distance")
        else:
            stack = [] # Get both axes and return max distance since only one axis will have a value that is non-zero
            for axis, value in enumerate(position): # x = axis 0, y = axis 1
                counter_axis = 1 - axis # The axis to reduce
                counter_index = position[counter_axis] # The coordinate/index on the axis to get a 1D array from
                dist = calc_collision_distance(grid_mask[p], counter_index, counter_axis, value, round(velocities[p, axis].item()))
                stack.append(abs(dist))
            distance[p] = max(stack)
    return distance

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
            timeout: int | None = 100,
            var_thresh: float | None = 0.15,
    ):
        if init_direction is None:
            init_direction = random.randint(0, 3)
        init_direction = min(max(init_direction % 4, 0), 3)
        init_length = max(init_length, 1)
        for size in shape:
            assert size > init_length * 2
        self.INIT_DIR = int(init_direction)
        self.INIT_LEN = init_length
        self.TIMEOUT = timeout
        self.VAR_THRESH = var_thresh

        self.players = players
        self.total = self.players.total
        self.shape = shape
        self.width, self.height = shape
        self.max_rel_distance = math.sqrt(self.width ** 2 + self.height ** 2)
        self.init_x, self.init_y = self.width // 2 + 1, self.height // 2 + 1
        self.factor_x = round(np.cos(0.5 * math.pi * self.INIT_DIR))
        self.factor_y = round(np.sin(0.5 * math.pi * self.INIT_DIR))

        self.grid = np.full(
            (self.total, self.width+2, self.height+2),
            fill_value=GridEnum.Empty.value, dtype=np.int64
        )
        self.player_indices = np.arange(self.total)
        self.food_locations: list[tuple[int, int]] | None = None
        self.boundary_mask = np.full_like(self.grid, fill_value=False, dtype=bool)
        self.head_mask = np.full_like(self.grid, fill_value=False, dtype=bool)
        self.direction = np.full(self.total, fill_value=self.INIT_DIR, dtype=int)
        self.action_count = np.full((self.total, 3), fill_value=0, dtype=int)
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
        # locations_total = len(self.food_locations)
        # next_indices = [
        #     self.snake_length[p] - self.INIT_LEN
        #     for p in range(self.total)
        # ]
        # next_positions = [
        #     self.food_locations[index] if index < locations_total else None
        #     for index in next_indices
        # ]
        next_positions = get_free_positions(self.empty, independent=True)
        set_positions(self.grid, next_positions, GridEnum.Food.value, mask, validation=True)

    def reset(self, complete=True):
        restart = self.players.to_restart()
        complete |= self.players.total != self.total # or np.all(restart == True).item()

        # Recreate masks if necessary and food locations
        if complete:
            self.total = self.players.total
            self.INIT_DIR = random.randint(0, 3)
            self.factor_x = round(math.cos(0.5 * math.pi * self.INIT_DIR))
            self.factor_y = round(math.sin(0.5 * math.pi * self.INIT_DIR))
            self.grid = np.full(
                (self.total, self.width+2, self.height+2),
                fill_value=GridEnum.Empty.value, dtype=np.int64
            )
            self.player_indices = np.arange(self.total)
            self.direction = np.full(self.total, fill_value=self.INIT_DIR, dtype=int)
            self.action_count = np.full((self.total, 3), fill_value=0, dtype=int)
            self.snake_length = np.full((self.total,), fill_value=self.INIT_LEN, dtype=int)
            self.hiatus = np.full((self.total,), fill_value=0, dtype=int)
            self.prev_distance = np.full((self.total,), fill_value=1.0, dtype=float)

            # Boundary mask
            self.boundary_mask = np.full_like(self.grid, fill_value=False, dtype=bool)
            player_indices = np.arange(self.players.total)
            limits = [list(range(dim_size)) for dim_size in self.grid.shape[1:]]
            for dim in prange(self.grid.ndim - 1):
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
            self.direction[:]               = self.INIT_DIR
            self.action_count[:]            = 0
            self.snake_length[:]            = self.INIT_LEN
            self.hiatus[:]                  = 0
            self.prev_distance[:]           = 1.0 # self.max_distance
        else:
            _restart = restart[:, None, None] & np.ones_like(self.grid, dtype=bool)
            self.grid[_restart]                      = GridEnum.Empty.value
            self.grid[_restart & self.boundary_mask] = GridEnum.Boundary.value
            self.grid[_restart & self.head_mask]     = GridEnum.SnakeHead.value
            self.direction[restart]                  = self.INIT_DIR
            self.snake_length[restart]               = self.INIT_LEN
            self.hiatus[restart]                     = 0
            self.prev_distance[restart]              = 1.0 # self.max_distance
        for i in prange(self.INIT_LEN):
            pos_x, pos_y = int(self.init_x - (self.factor_x * (i+1))), int(self.init_y - (self.factor_y * (i+1)))
            if complete:
                self.grid[:, pos_x, pos_y] = GridEnum.SnakeBody.value + i
            else:
                self.grid[restart, pos_x, pos_y] = GridEnum.SnakeBody.value + i
                setup = True
        # TODO: Might want to scrap standardized movements because player's snake states might not allow at some point in time
        if self.food_locations is None or complete: # TODO: Make this optional
            self.food_locations = get_free_positions(self.empty)
            random.shuffle(self.food_locations)
        self._set_food(None if complete else restart)

    def restart(self):
        self.reset(complete=False)

    def get_body_mask(self, index: int) -> CPUArray[tuple[int, int, int], bool]:
        if index < 0:
            index = GridEnum.SnakeBody.value + self.snake_length + index
            index[index == (GridEnum.SnakeBody.value - 1)] = GridEnum.SnakeHead.value
            index = index[:, None, None]
        else:
            index = GridEnum.SnakeHead.value if index == 0 else GridEnum.SnakeBody.value + index
        # TODO: Might want to add check to ensure limits of negative indices
        if np.any(index < GridEnum.SnakeHead.value):
            raise ValueError(f"A player has a snake with no body or index not within limits")
        return self.grid == index

    def move(self, action: CPUArray[tuple[int], int], convolutional: bool = True, verbose: int | bool = False):
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
        _no_food_collision = ~food_collision[:, None, None]
        # _active = active[:, None, None]
        old_body_mask = (prev_grid >= GridEnum.SnakeHead.value) # & _active
        new_head = self.get_body_mask(0) # At this stage there should be 2 heads on the grid, assuming it moved
        assert(np.all(np.count_nonzero(new_head, axis=(1, 2)) == 2))
        grid[old_body_mask & (~new_head)] += 1 # Move the rest of the body. Exclude head since it might have eaten itself
        grid[head] = GridEnum.SnakeBody.value # Ensure first body part location
        tail_moved = tail & _no_food_collision
        grid[tail_moved & (~new_head)] = GridEnum.Empty.value # Extend snake on eating food. & _active
        self.snake_length[food_collision] += 1
        self.hiatus[food_collision] = 0
        self.hiatus[~food_collision] += 1
        old_body_mask[tail_moved] = False # The tail has moved so snake can't collide with it if no food eaten
        body_collision = check_collision(grid, old_body_mask, GridEnum.SnakeHead.value)
        if verbose:
            if np.any(food_collision & body_collision):
                raise RuntimeError("Cannot collide with food and body")
            new_head = self.get_body_mask(0) # Should only be one head at this point and it should be at the new location
            check = np.count_nonzero(new_head, axis=(1, 2)) != 1
            check_index = np.argmax(check)
            if np.any(check):
                print(f"Head positions = {head_positions[check_index]}")
                print(f"Actions = {action[check_index]}")
                print(f"Directions = {self.direction[check_index]}")
                print(f"New head positions = {new_head_positions[check_index]}")
                print(f"Previous Grid = \n{prev_grid[check_index]}")
                print(f"Current Grid = \n{grid[check_index]}")
                raise RuntimeError("A snake head has not moved.")
        boundary_mask = self.boundary_mask # & _active
        wall_collision = check_collision(grid, boundary_mask, GridEnum.SnakeHead.value)
        if verbose:
            if np.any(body_collision | wall_collision | food_collision):
                paused = True
        if self.TIMEOUT is not None:
            move_food = (self.hiatus % self.TIMEOUT) == 0
            grid[food_mask & _no_food_collision & (move_food[:, None, None])] = GridEnum.Empty.value
            self._set_food(food_collision | move_food) # Place new food when eaten
            new_food_positions = get_positions(grid == GridEnum.Food.value)

        self.action_count[self.player_indices, action] += 1
        if self.VAR_THRESH is not None:
            max_count = self.action_count.max(axis=-1)  # shape: (players,)
            thresholds = np.ceil(max_count * self.VAR_THRESH)  # shape: (players,)
            low_action_usage = (self.action_count < thresholds[:, None]).any(axis=-1)
        else:
            low_action_usage = np.zeros_like(self.hiatus, dtype=bool)

        distances = get_rel_distance(new_head_positions, new_food_positions) / self.max_rel_distance
        distances[np.isnan(distances)] = 1.0 # self.max_distance
        moved_closer = (self.prev_distance - distances) > 0
        self.prev_distance = distances.copy()
        self.players.update(
            food_collision, wall_collision, body_collision, self.completed,
            distances, self.hiatus, moved_closer, low_action_usage,
        ) # , True)

        # TODO: Restart immediately after player update because of getting the next state
        if verbose:
            if np.any(self.players.disqualified):
                stop = True
        self.restart()
        if verbose:
            self.debug_grid_state(grid, prev_grid)

        if not convolutional:
            grid = self.grid
            # TODO: Attempt to scale values between -1 and 1
            # dir_x, dir_y = np.cos(0.5 * np.pi * self.direction).round(), np.sin(0.5 * np.pi * self.direction).round()
            snake_dir = self.direction / 3

            base = [snake_dir]
            food = []
            goal = []
            danger = []
            # just_died = wall_collision | body_collision
            # not_just_died = ~just_died
            # active &= not_just_died
            # _active = active[:, None, None]
            pseudo_food_mask = grid == GridEnum.Food.value
            _pseudo_body_mask = (grid >= GridEnum.SnakeHead.value) # Get fake old body locations
            pseudo_head, pseudo_tail = self.get_body_mask(0), self.get_body_mask(-1)
            pseudo_old_head_positions = get_positions(pseudo_head)
            pseudo_food_positions = get_positions(pseudo_food_mask)
            dim_dist_max = max(self.width, self.height)
            for t in range(3):
                pseudo_grid = self.grid.copy()
                pseudo_direction = self.direction.copy() - (t-1)
                pseudo_direction[pseudo_direction == -1] = 3
                pseudo_direction[pseudo_direction == +4] = 0
                pseudo_components = np.stack([
                    np.cos(0.5 * np.pi * pseudo_direction).round(), np.sin(0.5 * np.pi * pseudo_direction).round()
                ], axis=-1)
                pseudo_head_positions = get_new_positions(pseudo_old_head_positions, pseudo_direction)
                fault = set_positions( # Move the snake head first 
                    pseudo_grid, pseudo_head_positions, GridEnum.SnakeHead.value, None, validation=False
                ) # NOTE: active -> not_just_died -> None
                if fault is not None:
                    index, e = fault
                    print(f"\n\nFaulty index = {index}")
                    print(f"Current position = {pseudo_old_head_positions[index]}")
                    print(f"Check position = {pseudo_head_positions[index]}")
                    print(f"Direction = {pseudo_direction[index]}")
                    print(f"Action = {t}")
                    print(f"Died = {(body_collision | wall_collision)[index]}")
                    print(f"Disq = {self.players.disqualified[index]}")
                    print(f"Lives = {self.players.lives[index]}")
                    print(f"Grid = \n", grid[index])
                    print(f"Pseudo grid = \n", pseudo_grid[index])
                    raise e

                food_dist = get_rel_distance(pseudo_head_positions, pseudo_food_positions, False)
                food_dist[np.isnan(food_dist)] = self.max_rel_distance
                food_dist /= self.max_rel_distance

                goal_dist = get_lim_distance(pseudo_food_mask, pseudo_old_head_positions, pseudo_components)
                goal_dist /= dim_dist_max

                pseudo_food_collision = check_collision(pseudo_grid, pseudo_food_mask, GridEnum.SnakeHead.value)
                _pseudo_food_collision = pseudo_food_collision[:, None, None]

                pseudo_body_mask = _pseudo_body_mask.copy() # Get copy since it might be modified
                # pseudo_new_head = pseudo_grid == GridEnum.SnakeHead.value # Get the 2 head locations
                # pseudo_grid[pseudo_body_mask & (~pseudo_new_head)] += 1 # Update body excluding if head on body
                # pseudo_grid[pseudo_new_head] = GridEnum.SnakeBody.value # Set first body part
                pseudo_tail_moved = pseudo_tail & (~_pseudo_food_collision)
                # pseudo_grid[pseudo_tail_moved & (~pseudo_new_head)] = GridEnum.Empty.value # Update tail if not eaten
                pseudo_body_mask[pseudo_tail_moved] = False # Update tail in mask

                # _body_collision = check_collision(pseudo_grid, pseudo_body_mask, GridEnum.SnakeHead.value)
                # _wall_collision = check_collision(pseudo_grid, boundary_mask, GridEnum.SnakeHead.value)

                full_mask = pseudo_body_mask | boundary_mask
                danger_dist = get_lim_distance(full_mask, pseudo_old_head_positions, pseudo_components)
                danger_dist /= dim_dist_max
                goal_dist[danger_dist < goal_dist] *= -1

                # dir_danger = (_body_collision | _wall_collision).astype(float)
                # dir_danger[pseudo_food_collision] = -1
                # # dir_danger[just_died] = -1

                food.append(food_dist)
                goal.append(goal_dist)
                danger.append(danger_dist)

            result = base + food + goal + danger
            return result
        else:
            normalized_grid = (self.grid - GridEnum.Food.value) / GridEnum.Food.value * 2
            if verbose:
                if np.any(self.players.frames_done == 0):
                    stop = True
            return normalized_grid

    def debug_grid_state(self, grid: CPUArray,  prev_grid: CPUArray):
        g = grid
        GE = GridEnum
        allowed = (
            (g == GE.Empty.value) |
            (g == GE.Boundary.value) |
            (g == GE.Food.value) |
            (g == GE.SnakeHead.value) |
            (g >= GE.SnakeBody.value)
        )
        if not allowed.all():
            bad_idx = np.argwhere(~allowed)
            print("=== INVALID GRID VALUES FOUND ===")
            print("unique grid values:", np.unique(g))
            print("first 20 bad cells (player,x,y,value):")
            for p,x,y in bad_idx[:20]:
                print(p, x, y, "val=", int(g[p,x,y]))
            # dump more context for the first offending player
            p0 = int(bad_idx[0,0])
            print("--- player", p0, "---")
            print("snake_length:", int(self.snake_length[p0]))
            print("hiatus:", int(self.hiatus[p0]))
            print("unique values for player:", np.unique(g[p0]))
            print("prev_grid slice for player:\n", prev_grid[p0])
            print("curr_grid slice for player:\n", grid[p0])
            # optional: raise so you get a full trace
            raise RuntimeError("Invalid grid values detected - see console")

        # per-player invariants
        for p in range(self.total):
            pl = g[p]
            head_count = int(np.sum(pl == GE.SnakeHead.value))
            body_count = int(np.sum(pl >= GE.SnakeBody.value))
            if head_count != 1 or body_count != int(self.snake_length[p]):
                print(f"*** Invariant fail for player {p}: heads={head_count}, bodies={body_count}, snake_length={self.snake_length[p]}")
                print("unique:", np.unique(pl))
                print("head positions:", np.argwhere(pl == GE.SnakeHead.value))
                print("body positions:", np.argwhere(pl >= GE.SnakeBody.value))
                # dump masks/collisions and small context
                raise RuntimeError(f"Invariant failed for player {p}")



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