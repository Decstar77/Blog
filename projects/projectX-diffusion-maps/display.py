import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


def load_map(path):
    grid = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line:
                grid.append([int(c) for c in line])
    return np.array(grid, dtype=int)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python display.py <map_file>')
        sys.exit(1)
    display_map(sys.argv[1])
