"""Reference model test: mm_pickup_truck_1 from the AeronauticsModpack Prism instance.

Run: python tests/test_pickup_truck.py [instance folder]
Skips if the instance isn't on this machine.
"""
import os
import sys
import tempfile

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modded2model import instance, printprep  # noqa: E402
from modded2model.__main__ import main  # noqa: E402
from modded2model.assets import Assets  # noqa: E402
from modded2model.schematic import load_structure  # noqa: E402
from modded2model.voxelize import Voxelizer, rasterize_mesh  # noqa: E402
from test_pipeline import check_watertight  # noqa: E402

DEFAULT = os.path.join(os.environ.get("APPDATA", ""), "PrismLauncher", "instances", "AeronauticsModpack")
NAME = "mm_pickup_truck_1"


def run(inst):
    structure = load_structure(instance.find_schematic(inst, NAME))
    vox = Voxelizer(Assets(instance.asset_paths(inst)), 16)
    grid = vox.voxelize(structure)

    # Every block resolves to a real shape.
    assert not vox.unknown, f"unknown blocks: {vox.unknown}"
    assert not vox.no_geometry, f"blocks drawn as cubes: {vox.no_geometry}"

    # The four Offroad wheels hang below the chassis, so the model is taller than the blocks.
    assert grid.shape[1] > structure.size[1] * 16, "wheels missing"

    # Tires are written as smooth meshes, not voxels; count them as solid for connectivity
    # since the slicer merges them with whatever they overlap.
    assert len(vox.smooth_meshes) == 4, "expected four tires"
    joined = grid.copy()
    for tris in vox.smooth_meshes:
        rasterize_mesh(joined, tris)

    # One main body; only a few tiny parts (cranks, a lever) may stand apart.
    lab, n = ndimage.label(joined, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(lab.ravel())[1:]
    loose = int(sizes.sum() - sizes.max())
    print(f"{n} pieces, {loose} loose voxels of {int(sizes.sum())}")
    assert loose < 0.01 * sizes.sum(), "copycats or parts floating away from the body"

    # Mirrored copycats on the left and right sides produce a mirror-symmetric body.
    solid = printprep.solidify(grid, 8)
    mirror = solid[::-1]
    overlap = (solid & mirror).sum() / solid.sum()
    print(f"left/right symmetry {overlap:.1%}")
    assert overlap > 0.9, "left and right sides differ"

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "truck.stl")
        main([NAME, "--instance", inst, "-o", out])
        tris, bad = check_watertight(out)
        print(f"{tris} triangles, {bad} bad edges")
        assert bad == 0


if __name__ == "__main__":
    inst = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.isdir(inst):
        print("SKIP: instance not found")
        sys.exit(0)
    run(inst)
    print("OK")
