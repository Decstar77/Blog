import copy
import numpy as np

from mapgen import EMPTY, WALL, FLOOR, PLAYER, ENEMY


class MoveResult:
    OUT_OF_BOUNDS = "out_of_bounds"
    HIT_WALL      = "hit_wall"
    HIT_EMPTY     = "hit_empty"
    MOVED         = "moved"
    ATE_ENEMY     = "ate_enemy"
    NO_PLAYER     = "no_player"


class MapController:
    def __init__(self, grid):
        self.grid = copy.deepcopy(grid)
        self.height, self.width = self.grid.shape
        self._player_pos = self._find_player()

    def _find_player(self):
        ys, xs = np.where(self.grid == PLAYER)
        if len(xs) == 0:
            return None
        return (int(xs[0]), int(ys[0]))

    def player_position(self):
        return self._player_pos

    def enemy_count(self):
        return int(np.sum(self.grid == ENEMY))

    def enemy_positions(self):
        ys, xs = np.where(self.grid == ENEMY)
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    def all_enemies_eaten(self):
        return self.enemy_count() == 0

    def tile_at(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        return int(self.grid[y, x])
    
    def tile_at_tuple(self, t):
        if not (0 <= t[0] < self.width and 0 <= t[1] < self.height):
            return None
        return int(self.grid[t[1], t[0]])

    def shape(self):
        return self.grid.shape

    def snapshot(self):
        return self.grid.copy()

    def _move(self, dx, dy):
        if self._player_pos is None:
            return MoveResult.NO_PLAYER

        px, py = self._player_pos
        nx, ny = px + dx, py + dy

        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return MoveResult.OUT_OF_BOUNDS

        target = self.grid[ny, nx]

        if target == WALL:
            return MoveResult.HIT_WALL
        if target == EMPTY:
            return MoveResult.HIT_EMPTY

        ate = (target == ENEMY)

        self.grid[py, px] = FLOOR
        self.grid[ny, nx] = PLAYER
        self._player_pos = (nx, ny)

        return MoveResult.ATE_ENEMY if ate else MoveResult.MOVED

    def move_player_left(self):
        return self._move(-1, 0)

    def move_player_right(self):
        return self._move(1, 0)

    def move_player_up(self):
        return self._move(0, -1)

    def move_player_down(self):
        return self._move(0, 1)
