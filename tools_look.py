"""Render a block in context from a schematic (textured, 4 views) to check it against the game.

python tools_look.py <schematic> <block id substring> [radius] [index] -o out.png
"""
import argparse
import math
import numpy as np
from modded2model import instance
from modded2model.assets import Assets
from modded2model.schematic import load_structure, Structure
from modded2model.voxelize import Voxelizer
from modded2model.blender import TexturedBuilder
from modded2model.preview_render import render_textured
from tools_render import write_png

D = r"C:/Users/zlwaj/AppData/Roaming/PrismLauncher/instances/AeronauticsModpack"
_vox = None


def look(schem, block, radius=0, index=0, out=None, views=((30, 25), (120, 25), (210, 25), (300, -30)), size=360, backface=False):
    global _vox
    if _vox is None:
        _vox = Voxelizer(Assets(instance.asset_paths(D)), 16)
    s = load_structure(instance.find_schematic(D, schem))
    hits = [b for b in s.blocks if block in b[3].name]
    if not hits:
        raise SystemExit(f"no {block} in {schem}")
    cx, cy, cz, st = hits[index % len(hits)]
    print(st.name, dict(st.properties), s.block_data.get((cx, cy, cz), {}) and list(s.block_data[(cx, cy, cz)])[:12])
    near = [b for b in s.blocks if max(abs(b[0] - cx), abs(b[1] - cy), abs(b[2] - cz)) <= radius]
    sub = Structure(s.size, near, {k: v for k, v in s.block_data.items() if any(k == b[:3] for b in near)})
    f = TexturedBuilder(_vox, sub).build()
    imgs = [render_textured(f, _vox.assets, math.radians(y), math.radians(p), size, backface=backface) for y, p in views]
    img = np.concatenate(imgs, 1).astype(np.uint8)
    if out:
        write_png(out, img)
    return st, f


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("schematic"); p.add_argument("block")
    p.add_argument("radius", nargs="?", type=int, default=0); p.add_argument("index", nargs="?", type=int, default=0)
    p.add_argument("-o", default="exports/previews/check/look.png")
    a = p.parse_args()
    look(a.schematic, a.block, a.radius, a.index, a.o)
