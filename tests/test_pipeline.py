"""End-to-end check: fake mod jar + vanilla jar + structure file -> watertight STL.

Run: python tests/test_pipeline.py [path/to/vanilla-client.jar]
"""
import json
import os
import sys
import tempfile
import zipfile
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modded2model import nbt  # noqa: E402
from modded2model.__main__ import main  # noqa: E402


def make_mod_jar(path):
    shaft = {"elements": [{"from": [6, 0, 6], "to": [10, 16, 10]}]}
    cog = {"parent": "testmod:block/shaft", "elements": [
        {"from": [6, 0, 6], "to": [10, 16, 10]},
        {"from": [2, 6, 2], "to": [14, 10, 14], "rotation": {"angle": 22.5, "axis": "y", "origin": [8, 8, 8]}},
    ]}
    axis_variants = {"axis=y": {"model": "testmod:block/shaft"},
                     "axis=x": {"model": "testmod:block/shaft", "x": 90, "y": 90},
                     "axis=z": {"model": "testmod:block/shaft", "x": 90}}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("assets/testmod/models/block/shaft.json", json.dumps(shaft))
        z.writestr("assets/testmod/models/block/cog.json", json.dumps(cog))
        z.writestr("assets/testmod/blockstates/shaft.json", json.dumps({"variants": axis_variants}))
        z.writestr("assets/testmod/blockstates/cog.json", json.dumps({"variants": {"": {"model": "testmod:block/cog"}}}))


def make_structure(path, with_vanilla):
    palette = [{"Name": "testmod:shaft", "Properties": {"axis": "x"}},
               {"Name": "testmod:cog"},
               {"Name": "testmod:mystery_block"}]
    blocks = [{"pos": [x, 1, 1], "state": 0} for x in range(4)]
    blocks += [{"pos": [4, 1, 1], "state": 1}, {"pos": [0, 0, 0], "state": 2}]
    if with_vanilla:
        palette.append({"Name": "minecraft:oak_stairs",
                        "Properties": {"facing": "east", "half": "bottom", "shape": "straight", "waterlogged": "false"}})
        blocks += [{"pos": [x, 0, 1], "state": 3} for x in range(1, 5)]
    with open(path, "wb") as f:
        f.write(nbt.dumps({"size": [5, 2, 3], "palette": palette, "blocks": blocks}))


def check_watertight(stl_path):
    data = open(stl_path, "rb").read()
    n = int.from_bytes(data[80:84], "little")
    rec = np.frombuffer(data[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n)
    v = np.round(rec["v"], 4)
    edges = Counter()
    for tri in v:
        pts = [tuple(p) for p in tri]
        for i in range(3):
            edges[(pts[i], pts[(i + 1) % 3])] += 1
    # Closed shells: every edge is matched by the same number of reverse edges. Exact exports
    # hold one closed shell per model element, and touching shells may share edges.
    bad = [e for e, c in edges.items() if edges.get((e[1], e[0]), 0) != c]
    return n, len(bad)


if __name__ == "__main__":
    vanilla = sys.argv[1] if len(sys.argv) > 1 else None
    with tempfile.TemporaryDirectory() as d:
        jar = os.path.join(d, "testmod.jar")
        struct_path = os.path.join(d, "build.nbt")
        out = os.path.join(d, "build.stl")
        make_mod_jar(jar)
        make_structure(struct_path, with_vanilla=bool(vanilla))
        args = [struct_path, "-a", jar, "-o", out, "--res", "16"]
        if vanilla:
            args += ["-a", vanilla]
        main(args)
        n, bad = check_watertight(out)
        print(f"Edge check: {n} triangles, {bad} bad edges")
        assert bad == 0, "mesh is not watertight"
        main(args[:4] + [os.path.join(d, "b.3mf")] + args[5:])
        assert zipfile.ZipFile(os.path.join(d, "b.3mf")).read("3D/3dmodel.model")
        print("OK")
