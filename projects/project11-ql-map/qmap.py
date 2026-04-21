import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F
import mapgen as mg
import mapcontroller as mc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

#EMPTY  = 0
#WALL   = 1
#FLOOR  = 2
#PLAYER = 3
#ENEMY  = 4

random.seed(32)

game_map_data = mg.Cellular().generate(32, 32, 0.6, 10)
game_map = mc.MapController(game_map_data)

COLORS = [
    '#1a1a2e',  # 0 empty air    - near-black navy
    '#4a4a5a',  # 1 solid wall   - dark slate
    '#c8a96e',  # 2 floor        - warm sand
    '#00e676',  # 3 player spawn - bright green
    '#ff1744',  # 4 enemy spawn  - vivid red
]

LABELS = ['Empty', 'Wall', 'Floor', 'Player', 'Enemy']
def display_map(grid):
    cmap = ListedColormap(COLORS)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(COLORS) - 1,
              interpolation='nearest', aspect='equal')

    legend = [mpatches.Patch(color=COLORS[i], label=f'{i} – {LABELS[i]}')
              for i in range(len(COLORS))]
    ax.legend(handles=legend, loc='upper right', fontsize=9,
              framealpha=0.85, edgecolor='#888')

    ax.set_title("Map", fontsize=11)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    plt.show()

def reward_for_tile(tile):
    if ( tile == mg.EMPTY ): return -100
    elif ( tile == mg.WALL ):  return -1
    elif ( tile == mg.FLOOR ): return 0
    elif ( tile == mg.PLAYER ): assert False
    elif ( tile == mg.ENEMY ): return 1

def populate_q(q, pos):
    right   = reward_for_tile( game_map.tile_at(pos[0] + 1, pos[1]) )
    left    = reward_for_tile( game_map.tile_at(pos[0] - 1, pos[1]) )
    up      = reward_for_tile( game_map.tile_at(pos[0], pos[1] - 1) )
    down    = reward_for_tile( game_map.tile_at(pos[0], pos[1] + 1) )
    q[pos] = {
        "right" : right,
        "left" : left,
        "up" : up,
        "down" : down
    }

while game_map.all_enemies_eaten() == False:
    q = {}
    populate_q(q, game_map.player_position())
    state = q[game_map.player_position()]

    print(state)
    display_map(game_map.grid)
    
    choice = "right"
    if state["right"] == state["left"] == state["up"] == state["down"] == 0:
        choice = random.choice(["right", "left", "down", "up"])
    else:
        choice = max(state, key=state.get)

    if ( choice == "right"): game_map.move_player_right()
    elif ( choice == "left"): game_map.move_player_left()
    elif ( choice == "down"): game_map.move_player_down()
    elif ( choice == "up"): game_map.move_player_up()
    else: assert False






