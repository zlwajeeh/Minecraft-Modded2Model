"""Read Sable sub-levels (the physics ships used by Create Aeronautics) from a world save.

Layout, found by inspecting saves (not documented by Sable, so it may change):
  <world>/sublevels/r.<x>.<z>.<n>.slvls   region-style file: 1024-slot header, then per slot
                                          [length u32][1 byte][gzip NBT]
  each slot's NBT: plot.chunks{<packed ChunkPos>: {sections: {<Y>: {block_states: ...}}}},
                   pose.position / orientation, world_bounds, uuid
Blocks are stored unrotated in the ship's own grid, which is what a print wants.
"""
import glob
import os
import struct

import numpy as np

from . import nbt
from .schematic import AIR, BlockState, Structure
from .world import _unpack_indices


SECTION_Y_OFFSET = 64  # Sable plot section keys are indices from y = -64


def _signed32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


class Ship:
    def __init__(self, ident, data):
        self.id = ident
        self.data = data
        pos = data.get("pose", {}).get("position", {})
        self.position = tuple(pos.get(k, 0.0) for k in "xyz")
        self.uuid = "-".join(f"{v & 0xFFFFFFFF:08x}" for v in data.get("uuid", []))

    @property
    def name(self):
        """The ship's name: Sable's display name if set, else the text on its nameplates ("" if none)."""
        shown = _plain(self.data.get("display_name"))
        if shown:
            return shown
        plates = (be for chunk in self.data["plot"]["chunks"].values() for be in chunk.get("block_entities", []))
        return " / ".join(nameplate_names(plates))

    def sections(self):
        for key, chunk in self.data["plot"]["chunks"].items():
            packed = int(key)
            cx, cz = _signed32(packed), _signed32(packed >> 32)
            for sy, sec in chunk.get("sections", {}).items():
                bs = sec.get("block_states")
                if not bs:
                    continue
                palette = [BlockState(e["Name"], tuple(sorted((k, str(v)) for k, v in e.get("Properties", {}).items())))
                           for e in bs["palette"]]
                yield cx, int(sy), cz, palette, _unpack_indices(bs.get("data"), len(palette))

    def structure(self):
        cells = []
        for cx, sy, cz, palette, idx in self.sections():
            grid = idx.reshape(16, 16, 16)  # [y][z][x]
            for i, st in enumerate(palette):
                if st.name in AIR:
                    continue
                ys, zs, xs = np.nonzero(grid == i)
                for x, y, z in zip(xs + cx * 16, ys + sy * 16, zs + cz * 16):
                    cells.append((int(x), int(y), int(z), st))
        if not cells:
            return Structure((0, 0, 0))
        arr = np.array([c[:3] for c in cells])
        lo, hi = arr.min(0), arr.max(0)
        s = Structure(tuple(int(v) for v in hi - lo + 1))
        s.blocks = [(x - lo[0], y - lo[1], z - lo[2], st) for x, y, z, st in cells]
        s.plot_lo = tuple(int(v) for v in lo)  # plot-local (x, section y, z) of the grid's corner
        for key, chunk in self.data["plot"]["chunks"].items():
            packed = int(key)
            cx, cz = _signed32(packed), _signed32(packed >> 32)
            for be in chunk.get("block_entities", []):
                if {"x", "y", "z"} <= be.keys():
                    # block entities use the plot's absolute coordinates; chunk keys are plot-local
                    bx = cx * 16 + (be["x"] & 15)
                    bz = cz * 16 + (be["z"] & 15)
                    # section keys count from the world's floor (y = -64), block entities use real y
                    by = be["y"] + SECTION_Y_OFFSET
                    s.block_data[(bx - int(lo[0]), by - int(lo[1]), bz - int(lo[2]))] = be
        return s


def _plain(text):
    """Nameplate / display text as plain text (it may be a JSON text component)."""
    import json
    s = str(text or "").strip()
    if s[:1] in "{[\"":
        try:
            v = json.loads(s)
        except ValueError:
            return s
        if isinstance(v, str):
            return v.strip()
        parts = v if isinstance(v, list) else [v]
        return "".join(str(p.get("text", "")) + "".join(str(e.get("text", "")) for e in p.get("extra", []) if isinstance(e, dict))
                       if isinstance(p, dict) else str(p) for p in parts).strip()
    return s


def nameplate_names(block_entities):
    """Text on Aeronautics nameplates (simulated:nameplate), read from each row's controller plate
    (the one without a ControllerPos). Distinct names, in order found."""
    out = []
    for be in block_entities:
        if isinstance(be, dict) and str(be.get("id", "")).endswith(":nameplate") and "ControllerPos" not in be:
            name = _plain(be.get("Name"))
            if name and name not in out:
                out.append(name)
    return out


def structure_name(structure):
    """A ship's name from the nameplates among its blocks ("" if it has none)."""
    return " / ".join(nameplate_names((structure.block_data or {}).values()))


def ships(world):
    out = []
    for path in sorted(glob.glob(os.path.join(world, "sublevels", "*.slvls"))):
        with open(path, "rb") as f:
            d = f.read()
        name = os.path.basename(path)[:-6]
        for slot in range(1024):
            if len(d) < 4096:
                break
            loc = struct.unpack(">I", d[4 * slot:4 * slot + 4])[0]
            off = (loc >> 8) * 4096
            if not off or off + 5 > len(d):
                continue
            length = struct.unpack(">I", d[off:off + 4])[0]
            out.append(Ship(f"{name}:{slot}", nbt.loads(d[off + 5:off + 4 + length])))
    return out
