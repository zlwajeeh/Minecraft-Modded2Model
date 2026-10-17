"""Read blocks from a Java Edition world save (Anvil .mca region files, 1.18+ chunk format)."""
import os
import struct
import zlib

import numpy as np

from . import nbt
from .schematic import AIR, BlockState, Structure

_STATES = {}  # palette key -> [BlockState] (palettes repeat across chunks)


def palette_states(key):
    """The BlockStates of a palette key from World.section_data (cached)."""
    states = _STATES.get(key)
    if states is None:
        states = [BlockState(e["Name"], tuple(sorted((k, str(v)) for k, v in e.get("Properties", {}).items())))
                  for e in nbt.palette(key)]
        if len(_STATES) > 20000:
            _STATES.clear()
        _STATES[key] = states
    return states

DIMENSIONS = {"overworld": "", "nether": "DIM-1", "end": "DIM1"}


def region_dir(world, dimension="overworld"):
    sub = DIMENSIONS.get(dimension)
    if sub is None:  # modded dimension, e.g. "mymod:sky" -> dimensions/mymod/sky
        ns, _, path = dimension.partition(":")
        return os.path.join(world, "dimensions", ns, path, "region")
    return os.path.join(world, sub, "region")


class Region:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()

    def raw(self, lx, lz):
        """Decompressed NBT bytes of the chunk at local coords (0..31), or None if not generated."""
        i = 4 * (lx + 32 * lz)
        if len(self.data) < 8192:
            return None
        loc = struct.unpack(">I", self.data[i:i + 4])[0]
        offset, count = (loc >> 8) * 4096, loc & 0xFF
        if not offset or not count:
            return None
        length, comp = struct.unpack(">IB", self.data[offset:offset + 5])
        raw = self.data[offset + 5:offset + 4 + length]
        if comp == 2:
            return zlib.decompress(raw)
        if comp == 1:
            import gzip
            return gzip.decompress(raw)
        if comp == 3:
            return raw
        raise ValueError(f"unsupported chunk compression {comp}")

    def chunk(self, lx, lz):
        """NBT of the chunk at local coords (0..31) or None if not generated."""
        raw = self.raw(lx, lz)
        return None if raw is None else nbt.loads(raw)


def _unpack_indices(longs, palette_len):
    """Decode a packed block-state index array (4096 entries, entries never span longs)."""
    if palette_len <= 1 or longs is None or len(longs) == 0:
        return np.zeros(4096, np.int64)
    bits = max(4, (palette_len - 1).bit_length())
    per_long = 64 // bits
    arr = np.array(longs, dtype=np.int64).view(np.uint64)
    shifts = (np.arange(per_long, dtype=np.uint64) * np.uint64(bits))
    mask = np.uint64((1 << bits) - 1)
    vals = (arr[:, None] >> shifts[None, :]) & mask
    return vals.ravel()[:4096].astype(np.int64)


class World:
    def __init__(self, path, dimension="overworld"):
        self.path = path
        self.dir = region_dir(path, dimension)
        self._regions = {}

    def _region(self, rx, rz):
        key = (rx, rz)
        if key not in self._regions:
            p = os.path.join(self.dir, f"r.{rx}.{rz}.mca")
            self._regions[key] = Region(p) if os.path.isfile(p) else None
        return self._regions[key]

    def section_data(self, cx, cz):
        """[(section y, palette key, packed indices)] of a chunk, nothing decoded yet (see
        palette_states and _unpack_indices); None if the chunk isn't generated."""
        reg = self._region(cx >> 5, cz >> 5)
        raw = reg.raw(cx & 31, cz & 31) if reg is not None else None
        return None if raw is None else nbt.chunk_sections(raw)

    def sections(self, cx, cz):
        """Yield (section_y, palette [BlockState], indices[4096] in y,z,x order)."""
        for y, key, packed in self.section_data(cx, cz) or ():
            states = palette_states(key)
            yield y, states, _unpack_indices(packed, len(states))

    def extract(self, lo, hi):
        """Blocks in the inclusive box lo..hi (world coords) as a Structure with origin at lo."""
        x0, y0, z0 = (min(a, b) for a, b in zip(lo, hi))
        x1, y1, z1 = (max(a, b) for a, b in zip(lo, hi))
        s = Structure((x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1))
        for cx in range(x0 >> 4, (x1 >> 4) + 1):
            for cz in range(z0 >> 4, (z1 >> 4) + 1):
                for sy, palette, idx in self.sections(cx, cz):
                    if sy * 16 > y1 or sy * 16 + 15 < y0:
                        continue
                    grid = idx.reshape(16, 16, 16)  # [y][z][x]
                    for i, st in enumerate(palette):
                        if st.name in AIR:
                            continue
                        ys, zs, xs = np.nonzero(grid == i)
                        wx, wy, wz = xs + cx * 16, ys + sy * 16, zs + cz * 16
                        keep = (wx >= x0) & (wx <= x1) & (wy >= y0) & (wy <= y1) & (wz >= z0) & (wz <= z1)
                        for x, y, z in zip(wx[keep], wy[keep], wz[keep]):
                            s.blocks.append((int(x - x0), int(y - y0), int(z - z0), st))
        return s

    def chunk_list(self):
        """All generated chunk coords (for overviews)."""
        out = []
        for f in os.listdir(self.dir):
            parts = f.split(".")
            if len(parts) == 4 and parts[3] == "mca":
                rx, rz = int(parts[1]), int(parts[2])
                reg = self._region(rx, rz)
                for i in range(1024):
                    if reg and len(reg.data) >= 8192 and struct.unpack(">I", reg.data[4 * i:4 * i + 4])[0]:
                        out.append((rx * 32 + i % 32, rz * 32 + i // 32))
        return out
