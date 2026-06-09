"""Interactively view a .glb mesh/model.

Usage:
    python view.py path/to/model.glb

Controls (trimesh viewer):
    drag        - rotate
    ctrl+drag   - pan
    scroll      - zoom
    w           - toggle wireframe
    a           - toggle XYZ axis
    q / esc     - quit
"""

import argparse
import sys
from pathlib import Path

import trimesh

def view(path: Path) -> None:
    path = "data/glbs/glbs_2k/000-000/" + str(path)
    scene = trimesh.load(path, force="scene")

    geom_count = len(scene.geometry)
    vert_count = sum(len(g.vertices) for g in scene.geometry.values())
    face_count = sum(len(g.faces) for g in scene.geometry.values()
                     if hasattr(g, "faces"))

    scene.show(smooth=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively view a .glb model.")
    parser.add_argument("glb", type=str, help="Path to the .glb file")
    args = parser.parse_args()
    view(args.glb)


if __name__ == "__main__":
    main()
