import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import mapgen as mg

def load_map(path):
    grid = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line:
                grid.append([int(c) for c in line])
    return np.array(grid, dtype=int)

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

if __name__ == '__main__':
    cell = mg.Cellular()
    map = cell.generate(32, 32, 0.6, 10)
    display_map(map)
