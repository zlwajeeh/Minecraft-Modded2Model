"""--sub-level all: a schematic's main build with every Aeronautics sub-level placed where it was
saved, whether or not it is attached to anything.

Sub-levels saved unrotated are merged onto the grid as usual (schematic._merge_aligned). A
rotated one is a ship of its own at an angle: in game its blocks are at
    world = R @ local + position        (blocks; R from the saved orientation quaternion)
GLB exports place it exactly like that (a static contraptions.MovingPart, drawn turned). Prints
are built on the block grid, so there it is turned to the nearest quarter turn instead, block
states turned to match; blocks with no state for the new way round keep their shape, turned
(contraptions.keep_turned_shapes).
"""
import math

import numpy as np

from . import contraptions, nbt
from .schematic import _parse, load_structure


def quat_matrix(q):
    """3x3 rotation matrix of a saved {w, x, y, z} orientation."""
    w, x, y, z = (float(q.get(k, 0.0)) for k in "wxyz")
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def nearest_quarter_turn(r):
    """The rotation made of quarter turns closest to r (a signed permutation matrix, det +1)."""
    import itertools
    best, score = np.eye(3, dtype=int), -9.0
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3), int)
            for row, (col, s) in enumerate(zip(perm, signs)):
                m[row, col] = s
            if round(np.linalg.det(m)) == 1 and float((m * r).sum()) > score:
                best, score = m, float((m * r).sum())
    return best


def _turned_by(a, b):
    """Degrees between two rotation matrices."""
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def place_all(path, contraption_mode="world", exact=False):
    """(structure, notes): the main build, its merged sub-levels and every rotated sub-level,
    with every piece's moving contraptions attached. exact: GLB (rotated ships at their real
    angle); otherwise rotated ships are turned to the nearest quarter turn."""
    root = nbt.load(path)
    raw = root.get("sub_levels", [])
    main = load_structure(path, pick_ship=False)
    notes = []
    if contraption_mode != "none":  # the main build's own contraptions, and its merged sub-levels'
        found, to_local = contraptions.for_schematic(path, main, root)
        notes += contraptions.attach(main, found, to_local, contraption_mode, exact=exact)
    shift = np.array(getattr(main, "grid_shift", (0, 0, 0)), float)  # schematic -> main grid
    taken = {b[:3] for b in main.blocks}
    turned = dict(getattr(main, "turned", {}))
    moving = list(getattr(main, "moving", []))
    for i in getattr(main, "skipped", []):
        sl = _parse(raw[i])
        if not sl.blocks:
            continue
        r = quat_matrix(raw[i].get("orientation", {}))
        p = np.array([raw[i].get("position", {}).get(k, 0.0) for k in "xyz"], float) + shift
        first = np.array(sl.blocks[0][:3])
        if contraption_mode != "none":  # in the ship's own grid, before it is placed
            notes += contraptions.attach(sl, contraptions.in_schematic(raw[i]), contraptions._same,
                                         contraption_mode, exact=exact)
        moved = np.array(sl.blocks[0][:3]) - first  # attach may have shifted the ship's grid
        if exact:
            mp = contraptions.MovingPart(sl, tuple(p - moved), tuple(p), r, kind="sub-level", static=True)
            moving.append(mp)
            notes.append(f"Sub-level {i}: {len(sl.blocks)} blocks, placed where it was saved")
            continue
        q = nearest_quarter_turn(r)
        own = getattr(sl, "turned", {})
        for x, y, z, st in sl.blocks:
            c = np.array([x, y, z], float) - moved + 0.5  # block centre in the ship's saved frame
            pos = tuple(int(v) for v in np.rint(q @ c + p - 0.5))
            if pos in taken:  # where a ship overlaps the build, the build stays
                continue
            taken.add(pos)
            new = contraptions.rotate_state(st, q)
            main.blocks.append((*pos, new))
            if (x, y, z) in sl.block_data:
                main.block_data[pos] = sl.block_data[(x, y, z)]
            original, m = own.get((x, y, z), (st, np.eye(3, dtype=int)))
            if not (q @ m == np.eye(3, dtype=int)).all():
                turned[pos] = (original, q @ m)
        off = _turned_by(r, q)
        notes.append(f"Sub-level {i}: {len(sl.blocks)} blocks, placed where it was saved"
                     + (f" (turned {off:.0f} degrees to sit on the print's block grid)" if off > 0.5 else ""))
    main.turned, main.moving, main.skipped = turned, moving, []
    _fit(main)
    return main, notes


def _corners(mp):
    """Where the corners of a placed part's box end up (blocks)."""
    size = np.array(mp.structure.size, float)
    box = np.array([[x, y, z] for x in (0, size[0]) for y in (0, size[1]) for z in (0, size[2])])
    c = np.array(mp.center, float)
    return (box + np.array(mp.offset, float) - c) @ np.asarray(mp.matrix).T + c


def _fit(s):
    """Move everything into positive coordinates and size the structure to hold it all."""
    pts = [b[:3] for b in s.blocks] + [tuple(np.floor(p).astype(int)) for mp in s.moving for p in _corners(mp)]
    if not pts:
        return
    lo = np.min(pts, 0)
    d = [int(v) for v in -np.minimum(lo, 0)]
    if any(d):
        s.blocks = [(x + d[0], y + d[1], z + d[2], st) for x, y, z, st in s.blocks]
        s.block_data = {(x + d[0], y + d[1], z + d[2]): v for (x, y, z), v in s.block_data.items()}
        s.turned = {(x + d[0], y + d[1], z + d[2]): v for (x, y, z), v in s.turned.items()}
        for mp in s.moving:
            mp.offset = tuple(o + v for o, v in zip(mp.offset, d))
            mp.center = tuple(o + v for o, v in zip(mp.center, d))
    hi = np.max([b[:3] for b in s.blocks], 0) + 1 if s.blocks else np.zeros(3, int)
    for mp in s.moving:
        hi = np.maximum(hi, np.ceil(_corners(mp).max(0)).astype(int))
    s.size = tuple(int(max(a, b)) for a, b in zip(hi, s.size))
