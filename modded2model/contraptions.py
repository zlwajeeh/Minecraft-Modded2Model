"""Moving Create contraptions: blocks carried by a bearing, propeller bearing, windmill and so on.

While a contraption is assembled its blocks aren't blocks in the world any more: they are saved
inside an entity (create:stationary_contraption, aeronautics:propeller_bearing_contraption, ...)
as a palette plus a block list, relative to the contraption's anchor (the block in front of the
bearing), with the bearing's axis and current angle. This reads them back and places them.

Rotation follows Create (ControlledContraptionEntity.applyRotation -> catnip VecHelper.rotate):
positions turn about the centre of the anchor block, by `Angle` degrees about `Axis`:
  X: (x, y cos - z sin, z cos + y sin)
  Y: (x cos + z sin, y, z cos - x sin)
  Z: (x cos - y sin, y cos + x sin, z)

Angle modes:
  "world"     as it was when the world was saved. Blocks off the grid can't go in a print grid,
              so prints snap to the nearest quarter turn; GLB exports turn the meshes exactly.
  "snap"      the nearest quarter turn, so every block stays on the grid (blocks turned to match).
  "assembled" angle 0: as the contraption was built.
"""
import glob
import math
import os
from dataclasses import dataclass, field

import numpy as np

from . import nbt
from .schematic import AIR, BlockState, Structure

DIRS = {"north": (0, 0, -1), "south": (0, 0, 1), "west": (-1, 0, 0), "east": (1, 0, 0),
        "up": (0, 1, 0), "down": (0, -1, 0)}
FACING_INDEX = ["down", "up", "north", "south", "west", "east"]  # Minecraft's Direction order
PLOT_BASE = 20480000  # Sable keeps ship blocks in plots starting here (x and z)


@dataclass
class Contraption:
    kind: str  # the entity id, e.g. aeronautics:propeller_bearing_contraption
    anchor: tuple  # (x, y, z) of the anchor block, in the save's coordinates
    axis: str  # "x", "y" or "z"
    angle: float  # degrees, as saved
    blocks: list = field(default_factory=list)  # (dx, dy, dz, BlockState) relative to the anchor
    block_data: dict = field(default_factory=dict)  # (dx, dy, dz) -> block entity NBT


def _signed(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def _unpack_pos(v):
    """Minecraft's BlockPos.asLong: x in bits 38-63, z in 12-37, y in 0-11."""
    if isinstance(v, dict):
        return int(v.get("X", 0)), int(v.get("Y", 0)), int(v.get("Z", 0))
    if isinstance(v, (list, tuple)):
        return int(v[0]), int(v[1]), int(v[2])
    return _signed(v >> 38, 26), _signed(v, 12), _signed(v >> 12, 26)


def _vec(v):
    if isinstance(v, dict):
        return int(v.get("X", 0)), int(v.get("Y", 0)), int(v.get("Z", 0))
    return int(v[0]), int(v[1]), int(v[2])


def parse(entity):
    """A Contraption from an entity's NBT, or None if it isn't a block-carrying contraption."""
    c = entity.get("Contraption")
    if not isinstance(c, dict) or "Anchor" not in c:
        return None
    blocks = c.get("Blocks") or {}
    palette = [BlockState(e["Name"], tuple(sorted((k, str(v)) for k, v in e.get("Properties", {}).items())))
               for e in blocks.get("Palette", [])]
    axis = str(entity.get("Axis") or "").lower()
    if axis not in ("x", "y", "z"):
        facing = c.get("Facing")
        d = DIRS[FACING_INDEX[facing]] if isinstance(facing, int) and 0 <= facing < 6 else (0, 1, 0)
        axis = "xyz"[[abs(v) for v in d].index(1)]
    out = Contraption(str(entity.get("id")), _vec(c["Anchor"]), axis, float(entity.get("Angle", 0.0)))
    for b in blocks.get("BlockList", []):
        i = b.get("State", 0)
        if not 0 <= i < len(palette) or palette[i].name in AIR:
            continue
        pos = _unpack_pos(b.get("Pos", 0))
        out.blocks.append((*pos, palette[i]))
        if b.get("Data"):
            out.block_data[pos] = b["Data"]
    return out if out.blocks else None


# -- finding them ----------------------------------------------------------------------------------
def _entity_dir(world, dimension="overworld"):
    from .world import region_dir
    return os.path.join(os.path.dirname(region_dir(world, dimension)), "entities")


def in_world(world, lo=None, hi=None, dimension="overworld"):
    """Contraptions saved in a world whose anchor lies in the box lo..hi (x, z; y too if given),
    or all of them. Reads only the entity region files the box touches."""
    from .world import Region
    folder = _entity_dir(world, dimension)
    files = sorted(glob.glob(os.path.join(folder, "r.*.*.mca")))
    if lo is not None:
        rx0, rz0, rx1, rz1 = lo[0] >> 9, lo[2] >> 9, hi[0] >> 9, hi[2] >> 9
        files = [f for f in files
                 if rx0 <= int(os.path.basename(f).split(".")[1]) <= rx1
                 and rz0 <= int(os.path.basename(f).split(".")[2]) <= rz1]
    out = []
    for f in files:
        reg = Region(f)
        if len(reg.data) < 8192:
            continue
        for i in range(1024):
            raw = reg.raw(i % 32, i // 32)
            if raw is None or b"Contraption" not in raw:  # cheap test before parsing the chunk
                continue
            for e in nbt.loads(raw).get("Entities", []):
                c = parse(e)
                if c is None:
                    continue
                if lo is not None and not all(min(a, b) <= v <= max(a, b)
                                              for v, a, b in zip(c.anchor, lo, hi)):
                    continue
                out.append(c)
    return out


def on_ship(world, ship):
    """Contraptions mounted on a Sable ship (they are saved at the ship's plot coordinates)."""
    x0, z0 = plot_origin(ship)
    size = 16 << int(ship.data["plot"].get("log_size", 7))
    return in_world(world, (x0, -10 ** 6, z0), (x0 + size - 1, 10 ** 6, z0 + size - 1))


def plot_origin(ship):
    """Where a ship's plot starts in the save's coordinates (x, z)."""
    plot = ship.data["plot"]
    size = 16 << int(plot.get("log_size", 7))
    return PLOT_BASE + int(plot.get("plot_x", 0)) * size, PLOT_BASE + int(plot.get("plot_z", 0)) * size


def in_schematic(root):
    """Contraptions saved in a schematic's entity list, with anchors in schematic coordinates."""
    out = []
    for e in root.get("entities", []):
        tag = e.get("nbt") or {}
        c = parse(tag)
        if c is None:
            continue
        # the entity's absolute Pos sits on the anchor; its schematic-relative pos says where that is
        rel, absolute = e.get("blockPos") or e.get("pos"), tag.get("Pos")
        if rel is not None and absolute is not None:
            shift = [round(float(r)) - math.floor(float(a)) for r, a in zip(rel, absolute)]
            c.anchor = tuple(int(a) + s for a, s in zip(c.anchor, shift))
        out.append(c)
    return out


def _same(q):
    return tuple(int(v) for v in q)


def for_schematic(path, structure, root=None):
    """(contraptions, schematic -> grid mapping) for a structure from load_structure: a picked
    sub-level reads that sub-level's own entities; the main build reads the schematic's, plus those
    of every sub-level merged onto it (moved to where the sub-level was merged)."""
    root = root if root is not None else nbt.load(path)
    raw = root.get("sub_levels", [])
    for i, sl in enumerate(getattr(structure, "sub_levels", [])):
        if sl is structure and i < len(raw):
            return in_schematic(raw[i]), _same
    found = in_schematic(root)
    for i in getattr(structure, "merged", []):
        if i < len(raw):
            p = raw[i].get("position", {})
            off = [int(round(p.get(k, 0.0))) for k in "xyz"]
            for c in in_schematic(raw[i]):
                c.anchor = tuple(a + o for a, o in zip(c.anchor, off))
                found.append(c)
    shift = getattr(structure, "grid_shift", (0, 0, 0))
    return found, (lambda q: tuple(int(v + s) for v, s in zip(q, shift)))


def for_region(world, corner1, corner2, dimension="overworld"):
    """(contraptions, world -> grid mapping) for World.extract(corner1, corner2)."""
    lo = [min(u, v) for u, v in zip(corner1, corner2)]
    hi = [max(u, v) for u, v in zip(corner1, corner2)]
    return in_world(world, lo, hi, dimension), (lambda q: tuple(int(v - o) for v, o in zip(q, lo)))


def for_ship(world, ship, structure):
    """(contraptions, save -> ship grid mapping) for ship.structure()."""
    from .sable import SECTION_Y_OFFSET
    x0, z0 = plot_origin(ship)
    lx, ly, lz = structure.plot_lo
    return on_ship(world, ship), (lambda q: (int(q[0] - x0 - lx), int(q[1] + SECTION_Y_OFFSET - ly),
                                             int(q[2] - z0 - lz)))


# -- placing them ------------------------------------------------------------------------------------
def _rot_matrix(axis, degrees):
    """Create's rotation (see the module notes) as a 3x3 matrix acting on column vectors."""
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def angle_for(c, mode):
    if mode == "assembled":
        return 0.0
    if mode == "snap":
        return round(c.angle / 90.0) * 90.0
    return c.angle


def rotate_state(st, m):
    """Turn a block state by a quarter-turn matrix: facing, axis and side flags follow."""
    m = np.rint(m).astype(int)
    if (m == np.eye(3, dtype=int)).all():
        return st

    def turn(d):
        v = m @ np.array(DIRS[d])
        return next(k for k, dv in DIRS.items() if tuple(dv) == tuple(v))
    props = dict(st.props)
    out = {}
    for k, v in props.items():
        if k in ("facing", "horizontal_facing") and v in DIRS:
            nv = turn(v)
            # a horizontal-only facing can't point up or down: keep the original then
            out[k] = nv if not (k == "horizontal_facing" and nv in ("up", "down")) else v
        elif k == "axis" and v in ("x", "y", "z"):
            a = np.abs(m @ np.eye(3, dtype=int)["xyz".index(v)])
            out[k] = "xyz"[int(np.argmax(a))]
        elif k in DIRS and k not in out:
            out[turn(k)] = v  # fences, pipes, walls: the flag for each side moves with the side
        else:
            out.setdefault(k, v)
    return BlockState(st.name, tuple(sorted(out.items())))


def placed(c, angle):
    """The contraption's blocks turned by `angle` (a multiple of 90 here): [(x, y, z, state)] in
    the save's coordinates, and their block entity data by position."""
    m = _rot_matrix(c.axis, angle)
    blocks, data = [], {}
    for dx, dy, dz, st in c.blocks:
        r = m @ np.array([dx, dy, dz], float)  # about the anchor block's centre: integer offsets stay integer
        pos = tuple(int(round(v)) + a for v, a in zip(r, c.anchor))
        blocks.append((*pos, rotate_state(st, m)))
        if (dx, dy, dz) in c.block_data:
            data[pos] = c.block_data[(dx, dy, dz)]
    return blocks, data


@dataclass
class MovingPart:
    """A contraption to be drawn at an angle off the grid (GLB): its blocks in their own grid,
    where that grid's (0, 0, 0) sits in the export, and the turn about the anchor's centre."""
    structure: Structure
    offset: tuple  # export grid position of the part's (0, 0, 0)
    center: tuple  # export coordinates (blocks) of the turning point
    matrix: np.ndarray


def attach(structure, contraptions, to_local, mode="world", exact=False):
    """Add contraptions to an export. to_local maps save coordinates to the structure's grid.
    exact: True for GLB, where mode "world" keeps the precise angle as a MovingPart; otherwise
    every contraption is snapped onto the grid. Grows the structure to fit. Returns notes."""
    notes, extra, extra_data, moving = [], [], {}, []
    for c in contraptions:
        angle = angle_for(c, mode)
        off_grid = abs(angle - round(angle / 90) * 90) > 0.01
        if off_grid and exact:
            sub = Structure((1, 1, 1))
            lo = np.min([b[:3] for b in c.blocks], 0)
            sub.blocks = [(x - lo[0], y - lo[1], z - lo[2], st) for x, y, z, st in c.blocks]
            sub.size = tuple(int(v) for v in np.max([b[:3] for b in sub.blocks], 0) + 1)
            sub.block_data = {(x - lo[0], y - lo[1], z - lo[2]): d for (x, y, z), d in c.block_data.items()}
            a = to_local(c.anchor)
            moving.append(MovingPart(sub, tuple(int(v) for v in np.add(a, lo)),
                                     tuple(float(v) + 0.5 for v in a), _rot_matrix(c.axis, angle)))
            notes.append(f"Moving contraption ({c.kind}, {len(c.blocks)} blocks) at {angle:.1f} degrees")
            continue
        if off_grid:
            angle = round(angle / 90) * 90
        blocks, data = placed(c, angle)
        extra += [(*to_local(b[:3]), b[3]) for b in blocks]
        extra_data.update({to_local(p): d for p, d in data.items()})
        notes.append(f"Moving contraption ({c.kind}, {len(c.blocks)} blocks) placed at {angle:.0f} degrees"
                     + (f" (saved at {c.angle:.1f}, snapped to the grid)" if abs(angle - c.angle) > 0.01 else ""))
    if not (extra or moving):
        return notes
    # grow the structure so everything fits (contraptions can stick out past the build)
    pts = [b[:3] for b in structure.blocks] + [b[:3] for b in extra]
    for mp in moving:  # rough bounds of a turning part: its blocks' farthest reach from the centre
        r = np.linalg.norm(np.array(mp.structure.size, float)) + 1
        pts += [tuple(int(math.floor(v - r)) for v in mp.center), tuple(int(math.ceil(v + r)) for v in mp.center)]
    lo = np.min(pts, 0)
    shift = -np.minimum(lo, 0)
    if shift.any():
        sx, sy, sz = (int(v) for v in shift)
        structure.blocks = [(x + sx, y + sy, z + sz, st) for x, y, z, st in structure.blocks]
        structure.block_data = {(x + sx, y + sy, z + sz): d for (x, y, z), d in structure.block_data.items()}
        for mp in getattr(structure, "moving", []):
            mp.offset = tuple(o + s for o, s in zip(mp.offset, shift))
            mp.center = tuple(o + s for o, s in zip(mp.center, shift))
    taken = {b[:3] for b in structure.blocks}
    for x, y, z, st in extra:
        p = (x + int(shift[0]), y + int(shift[1]), z + int(shift[2]))
        if p not in taken:  # a contraption never overlaps the build in game; keep the build if it does here
            structure.blocks.append((*p, st))
            taken.add(p)
    for (x, y, z), d in extra_data.items():
        structure.block_data[(x + int(shift[0]), y + int(shift[1]), z + int(shift[2]))] = d
    for mp in moving:
        mp.offset = tuple(int(o + s) for o, s in zip(mp.offset, shift))
        mp.center = tuple(float(o + s) for o, s in zip(mp.center, shift))
    structure.moving = list(getattr(structure, "moving", [])) + moving
    hi = np.max([b[:3] for b in structure.blocks], 0) + 1
    structure.size = tuple(int(max(a, b)) for a, b in zip(hi, structure.size))
    return notes
