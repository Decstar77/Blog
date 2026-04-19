import numpy as np
import random

EMPTY  = 0
WALL   = 1
FLOOR  = 2
PLAYER = 3
ENEMY  = 4

class Cellular:
    def count_wall_neighbors(self, grid, x, y):
        count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                    if grid[ny, nx] == WALL:
                        count += 1
                else:
                    count += 1  # out-of-bounds counts as wall
        return count

    def smooth(self, grid):
        new_grid = np.copy(grid)
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if self.count_wall_neighbors(grid, x, y) >= 5:
                    new_grid[y, x] = WALL
                else:
                    new_grid[y, x] = FLOOR
        return new_grid

    def place_spawns(self, grid):
        floor_tiles = [(x, y)
                    for y in range(grid.shape[0])
                    for x in range(grid.shape[1])
                    if grid[y, x] == FLOOR]
        if not floor_tiles:
            return

        random.shuffle(floor_tiles)

        px, py = floor_tiles[0]
        grid[py, px] = PLAYER

        n_enemies = max(1, len(floor_tiles) // 20)
        for ex, ey in floor_tiles[1:n_enemies + 1]:
            grid[ey, ex] = ENEMY

    def fill_borders(self, grid):
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

    def thin_walls_to_border(self, grid):
        """Walls not touching a floor (8-dir) become empty, leaving a 1-tile wall ring."""
        height, width = grid.shape
        new_grid = grid.copy()
        for y in range(height):
            for x in range(width):
                if grid[y, x] != WALL:
                    continue
                keep = False
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and grid[ny, nx] == FLOOR:
                            keep = True
                            break
                    if keep:
                        break
                if not keep:
                    new_grid[y, x] = EMPTY
        return new_grid

    def flood_fill(self, grid, start_x, start_y):
        """Return the set of (x, y) floor positions reachable from start (4-directional)."""
        height, width = grid.shape
        visited = set()
        stack = [(start_x, start_y)]
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            if not (0 <= x < width and 0 <= y < height):
                continue
            if grid[y, x] != FLOOR:
                continue
            visited.add((x, y))
            stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return visited

    def remove_disconnected_floors(self, grid):
        """Keep only floors reachable from a random starting tile; island floors become walls."""
        floor_tiles = [(x, y)
                    for y in range(grid.shape[0])
                    for x in range(grid.shape[1])
                    if grid[y, x] == FLOOR]
        if not floor_tiles:
            return

        sx, sy = random.choice(floor_tiles)
        reachable = self.flood_fill(grid, sx, sy)

        for x, y in floor_tiles:
            if (x, y) not in reachable:
                grid[y, x] = WALL


    def remove_wall_islands(self, grid):
        """Replace isolated 2x2 wall blocks (no adjacent external wall) with floor."""
        height, width = grid.shape
        to_floor = set()

        for y in range(height - 1):
            for x in range(width - 1):
                # Must be a full 2x2 wall block
                if not all(grid[y + dy, x + dx] == WALL
                        for dy in range(2) for dx in range(2)):
                    continue

                block = {(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)}
                isolated = True
                for by in range(y - 1, y + 3):
                    for bx in range(x - 1, x + 3):
                        if (bx, by) in block:
                            continue
                        # Out-of-bounds counts as wall — block touches the border
                        if not (0 <= by < height and 0 <= bx < width):
                            isolated = False
                            break
                        if grid[by, bx] == WALL:
                            isolated = False
                            break
                    if not isolated:
                        break

                if isolated:
                    to_floor.update(block)

        for cx, cy in to_floor:
            grid[cy, cx] = FLOOR

    def generate(self, width=64, height=64, fill_ratio=0.45, smooth_passes=5):
        grid = np.zeros((height, width), dtype=int)
        for y in range(height):
            for x in range(width):
                grid[y, x] = WALL if random.random() < fill_ratio else FLOOR

        for _ in range(smooth_passes):
            grid = self.smooth(grid)
        self.fill_borders(grid)
        self.remove_disconnected_floors(grid)
        self.remove_wall_islands(grid)
        grid = self.thin_walls_to_border(grid)
        self.place_spawns(grid)
        return grid

class Drunk:
    def choose_direction4(self, x, y, sizeX, sizeY):
        dir = random.randint(1, 5)
        if dir == 1 and x - 1 >= 0:
            return (-1, 0)
        elif dir == 2 and x + 1 < sizeX:
            return (1, 0)
        elif dir == 3 and y - 1 >= 0:
            return (0, -1)
        elif dir == 4 and y + 1 < sizeY:
            return (0, 1)
        return (0, 0)


    def choose_direction8(self, x, y, sizeX, sizeY):
        dir = random.randint(1, 9)
        if dir == 1 and x - 1 >= 0:
            return (-1, 0)
        elif dir == 2 and x + 1 < sizeX:
            return (1, 0)
        elif dir == 3 and y - 1 >= 0:
            return (0, -1)
        elif dir == 4 and y + 1 < sizeY:
            return (0, 1)
        elif dir == 5 and x - 1 >= 0 and y - 1 >= 0:
            return (-1, -1)
        elif dir == 6 and x + 1 < sizeX and y - 1 >= 0:
            return (1, -1)
        elif dir == 7 and x - 1 >= 0 and y + 1 < sizeY:
            return (-1, 1)
        elif dir == 8 and x + 1 < sizeX and y + 1 < sizeY:
            return (1, 1)
        return (0, 0)


    def add_corridor_walls(self, grid):
        height, width = grid.shape
        to_wall = set()
        for y in range(height):
            for x in range(width):
                if grid[y, x] == FLOOR:
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if grid[ny, nx] == EMPTY:
                                    to_wall.add((ny, nx))
        for ny, nx in to_wall:
            grid[ny, nx] = WALL

    def remove_wall_islands(self, grid, size=2):
        """Replace isolated NxN wall blocks (no adjacent external wall) with floor."""
        height, width = grid.shape
        to_floor = set()

        for y in range(height - size + 1):
            for x in range(width - size + 1):

                # Check if NxN block is all WALL
                if not all(grid[y + dy, x + dx] == WALL
                        for dy in range(size) for dx in range(size)):
                    continue

                # Build the block set
                block = {(x + dx, y + dy) for dy in range(size) for dx in range(size)}

                isolated = True

                # Check surrounding border (1 tile padding)
                for by in range(y - 1, y + size + 1):
                    for bx in range(x - 1, x + size + 1):
                        if (bx, by) in block:
                            continue

                        # Out of bounds = not isolated (touches edge)
                        if not (0 <= by < height and 0 <= bx < width):
                            isolated = False
                            break

                        # Found neighboring wall = not isolated
                        if grid[by, bx] == WALL:
                            isolated = False
                            break

                    if not isolated:
                        break

                if isolated:
                    to_floor.update(block)

        # Apply changes
        for cx, cy in to_floor:
            grid[cy, cx] = FLOOR

        return grid

    def fill_borders(self, grid):
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

    def place_spawns(self, grid):
        floor_tiles = [(x, y)
                    for y in range(grid.shape[0])
                    for x in range(grid.shape[1])
                    if grid[y, x] == FLOOR]
        if not floor_tiles:
            return

        random.shuffle(floor_tiles)

        px, py = floor_tiles[0]
        grid[py, px] = PLAYER

        n_enemies = max(1, len(floor_tiles) // 20)
        for ex, ey in floor_tiles[1:n_enemies + 1]:
            grid[ey, ex] = ENEMY

    def generate(self, width=64, height=64, steps=2000):
        grid = np.zeros((height, width), dtype=int)

        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        grid[y, x] = FLOOR

        for _ in range(steps):
            dx, dy = self.choose_direction4(x, y, width, height)
            x += dx
            y += dy
            grid[y, x] = FLOOR

        self.add_corridor_walls(grid)
        self.remove_wall_islands(grid, 1)
        self.remove_wall_islands(grid, 2)
        self.remove_wall_islands(grid, 3)
        self.remove_wall_islands(grid, 4)
        self.fill_borders(grid)
        self.place_spawns(grid)
        return grid

MIN_LEAF_SIZE = 10  # minimum partition size before stopping splits
MIN_ROOM_DIM  = 4   # minimum room interior dimension

class Room:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def center(self):
        return (self.x + self.width // 2, self.y + self.height // 2)

    def floor_tiles(self):
        """All interior (non-wall border) positions as (x, y) tuples."""
        tiles = []
        for ry in range(self.y + 1, self.y + self.height - 1):
            for rx in range(self.x + 1, self.x + self.width - 1):
                tiles.append((rx, ry))
        return tiles

class BSPNode:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.left  = None
        self.right = None
        self.room  = None  # set only on leaf nodes

    def is_leaf(self):
        return self.left is None and self.right is None

    def split(self):
        """Attempt to split this leaf into two children. Returns True on success."""
        if not self.is_leaf():
            return False

        # Bias split direction by aspect ratio; fall back to random
        if self.width > self.height * 1.25:
            horizontal = False
        elif self.height > self.width * 1.25:
            horizontal = True
        else:
            horizontal = random.random() < 0.5

        if horizontal:
            max_pos = self.height - MIN_LEAF_SIZE
            if max_pos < MIN_LEAF_SIZE:
                return False
            pos = random.randint(MIN_LEAF_SIZE, max_pos)
            self.left  = BSPNode(self.x, self.y,       self.width, pos)
            self.right = BSPNode(self.x, self.y + pos, self.width, self.height - pos)
        else:
            max_pos = self.width - MIN_LEAF_SIZE
            if max_pos < MIN_LEAF_SIZE:
                return False
            pos = random.randint(MIN_LEAF_SIZE, max_pos)
            self.left  = BSPNode(self.x,       self.y, pos,             self.height)
            self.right = BSPNode(self.x + pos, self.y, self.width - pos, self.height)

        return True

class BSP:
    def build_bsp(self, node, depth=0, max_depth=5):
        if depth >= max_depth:
            return
        if node.split():
            self.build_bsp(node.left,  depth + 1, max_depth)
            self.build_bsp(node.right, depth + 1, max_depth)

    def carve_rooms(self, node, grid):
        """Paint rooms into the grid at every leaf node."""
        if node.is_leaf():
            max_w = node.width  - 2
            max_h = node.height - 2
            if max_w < MIN_ROOM_DIM or max_h < MIN_ROOM_DIM:
                return

            rw = random.randint(MIN_ROOM_DIM, max_w)
            rh = random.randint(MIN_ROOM_DIM, max_h)

            x_slack = node.width  - rw - 2
            y_slack = node.height - rh - 2
            rx = node.x + 1 + (random.randint(0, x_slack) if x_slack > 0 else 0)
            ry = node.y + 1 + (random.randint(0, y_slack) if y_slack > 0 else 0)

            node.room = Room(rx, ry, rw, rh)

            for y in range(ry, ry + rh):
                for x in range(rx, rx + rw):
                    on_border = (y == ry or y == ry + rh - 1 or
                                x == rx or x == rx + rw - 1)
                    if on_border:
                        if grid[y, x] == EMPTY:
                            grid[y, x] = WALL
                    else:
                        grid[y, x] = FLOOR
        else:
            if node.left:  self.carve_rooms(node.left,  grid)
            if node.right: self.carve_rooms(node.right, grid)

    def pick_room(self, node):
        """Return one representative room from this subtree."""
        if node.is_leaf():
            return node.room
        left  = self.pick_room(node.left)  if node.left  else None
        right = self.pick_room(node.right) if node.right else None
        if left is None:  return right
        if right is None: return left
        return random.choice([left, right])

    def dig_h(self, grid, x1, x2, y):
        """Carve a horizontal tunnel; walls and empty space become floor."""
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
                continue
            if grid[y, x] in (WALL, EMPTY):
                grid[y, x] = FLOOR

    def dig_v(self, grid, y1, y2, x):
        """Carve a vertical tunnel; walls and empty space become floor."""
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
                continue
            if grid[y, x] in (WALL, EMPTY):
                grid[y, x] = FLOOR

    def connect_siblings(self, node, grid):
        """Recursively connect each pair of sibling subtrees with an L-shaped corridor."""
        if node.is_leaf():
            return
        self.connect_siblings(node.left,  grid)
        self.connect_siblings(node.right, grid)

        a = self.pick_room(node.left)
        b = self.pick_room(node.right)
        if a is None or b is None:
            return

        ax, ay = a.center()
        bx, by = b.center()

        if random.random() < 0.5:
            self.dig_h(grid, ax, bx, ay)
            self.dig_v(grid, ay, by, bx)
        else:
            self.dig_v(grid, ay, by, ax)
            self.dig_h(grid, ax, bx, by)

    def add_corridor_walls(self, grid):
        """Add walls around all floor tiles that border empty space (8-directional)."""
        height, width = grid.shape
        to_wall = set()
        for y in range(height):
            for x in range(width):
                if grid[y, x] == FLOOR:
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if grid[ny, nx] == EMPTY:
                                    to_wall.add((ny, nx))
        for ny, nx in to_wall:
            grid[ny, nx] = WALL

    def collect_rooms(self, node):
        """Gather all rooms from leaf nodes."""
        if node.is_leaf():
            return [node.room] if node.room else []
        rooms = []
        if node.left:  rooms.extend(self.collect_rooms(node.left))
        if node.right: rooms.extend(self.collect_rooms(node.right))
        return rooms

    def enemy_count(self, room):
        """Scale enemy count with interior area: ~1 per 8 floor tiles, minimum 1."""
        interior_area = (room.width - 2) * (room.height - 2)
        return max(1, interior_area // 8)

    def place_spawns(self, grid, rooms):
        """Place one player spawn and area-scaled enemy spawns per room."""
        if not rooms:
            return

        random.shuffle(rooms)

        # Player in first room
        candidates = [(x, y) for (x, y) in rooms[0].floor_tiles()
                    if grid[y, x] == FLOOR]
        if candidates:
            px, py = random.choice(candidates)
            grid[py, px] = PLAYER

        # Enemies in remaining rooms — more enemies in larger rooms
        for room in rooms[1:]:
            candidates = [(x, y) for (x, y) in room.floor_tiles()
                        if grid[y, x] == FLOOR]
            if not candidates:
                continue
            max_n = self.enemy_count(room)
            n = min(random.randint(1, max_n), len(candidates))
            for ex, ey in random.sample(candidates, n):
                grid[ey, ex] = ENEMY

    def generate(self, width=64, height=64, max_depth=5):
        grid = np.zeros((height, width), dtype=int)

        root = BSPNode(0, 0, width, height)
        self.build_bsp(root, max_depth=max_depth)
        self.carve_rooms(root, grid)
        self.connect_siblings(root, grid)
        self.add_corridor_walls(grid)

        rooms = self.collect_rooms(root)
        self.place_spawns(grid, rooms)

        return grid

from tqdm import tqdm

def generate_map_data( filePath, cellular : Cellular, drunk : Drunk, bsp : BSP, count = 10000, size = 32):
    print("Building dataset...")
    grids = []
    for _ in enumerate(tqdm(range(count))):
        grids.append( cellular.generate(size, size, 0.6, 10) )
        grids.append( drunk.generate(size, size, 2500) )
        grids.append( bsp.generate(size, size, 5) )

    print(f"Saving dataset{ filePath }")
    np.savez_compressed(filePath, *grids)