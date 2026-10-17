"""Show what Minecraft-Modded2Model reads from the jars for a block.

python -m modded2model.inspect create:shaft axis=x -a <mods folder> -a <vanilla jar>
"""
import argparse

import numpy as np

from . import codeshapes
from .assets import Assets, _split_id
from .schematic import BlockState
from .voxelize import Voxelizer


def main(argv=None):
    p = argparse.ArgumentParser(prog="modded2model.inspect")
    p.add_argument("block", help="block id, e.g. create:shaft")
    p.add_argument("props", nargs="*", help="properties as key=value")
    p.add_argument("-a", "--assets", action="append", required=True)
    a = p.parse_args(argv)

    assets = Assets(a.assets)
    state = BlockState(a.block, tuple(sorted(kv.split("=", 1) for kv in a.props)))
    state = BlockState(state.name, tuple(tuple(x) for x in state.props))
    ns, path = _split_id(a.block)

    print(f"blockstate assets/{ns}/blockstates/{path}.json:", "found" if assets.blockstate(a.block) else "missing")
    if a.block in codeshapes.ALIASES:
        print("shape source: alias of", codeshapes.ALIASES[a.block])
    elif a.block in codeshapes.JAR_MODELS:
        print("shape source: jar models", codeshapes.JAR_MODELS[a.block][0])
    elif a.block in codeshapes.RULES:
        print("shape source: built-in rule ->", codeshapes.RULES[a.block](state.properties))
    else:
        parts = assets.resolve(state) or []
        print("shape source: blockstate models")
        for model, xr, yr in parts:
            print(f"  model x={xr} y={yr}, {len(model['elements'])} elements")
            for el in model["elements"]:
                rot = el.get("rotation", {})
                extra = f" rotated {rot['angle']} about {rot['axis']}" if rot.get("angle") else ""
                print(f"    {el['from']} -> {el['to']}{extra}")

    shape = Voxelizer(assets, 16).block_shape(state)
    if shape.any():
        idx = np.argwhere(shape)
        print(f"voxels: {int(shape.sum())} of 4096, bounds {idx.min(0).tolist()} -> {(idx.max(0) + 1).tolist()}")
    else:
        print("voxels: empty")


if __name__ == "__main__":
    main()
