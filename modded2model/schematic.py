"""Load block layouts from vanilla/Create structure files (.nbt).

Create's Schematic and Quill saves the vanilla structure format:
  size: [x, y, z]
  palette: [{Name: "create:shaft", Properties: {axis: "y"}}, ...]
  blocks: [{pos: [x, y, z], state: <palette index>, nbt: {...}}, ...]

With Create Aeronautics installed, ships (Sable sub-levels) are saved in the same file under
"sub_levels": a list of complete structures (size/palette/blocks) with a position and an
orientation. They spawn as their own physics objects when the schematic is placed.
"""
from dataclasses import dataclass, field

from . import nbt

AIR = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air", "minecraft:structure_void"}


@dataclass(frozen=True)
class BlockState:
    name: str
    props: tuple = ()  # sorted (key, value) pairs

    @property
    def properties(self):
        return dict(self.props)


@dataclass
class Structure:
    size: tuple
    blocks: list = field(default_factory=list)  # [(x, y, z, BlockState)]
    block_data: dict = field(default_factory=dict)  # (x, y, z) -> block entity NBT (e.g. a wheel mount's tire)
    sub_levels: list = field(default_factory=list)  # Aeronautics ships saved in the schematic (Structures)
    merged: list = field(default_factory=list)  # indices of sub_levels merged into these blocks
    skipped: list = field(default_factory=list)  # rotated sub_levels left out


def _parse(root):
    palettes = root.get("palettes")
    palette = palettes[0] if palettes else root.get("palette", [])
    states = []
    for entry in palette:
        props = tuple(sorted((k, str(v)) for k, v in entry.get("Properties", {}).items()))
        states.append(BlockState(entry["Name"], props))
    s = Structure(tuple(root.get("size", (0, 0, 0))))
    for b in root.get("blocks", []):
        st = states[b["state"]]
        if st.name in AIR:
            continue
        x, y, z = b["pos"]
        s.blocks.append((x, y, z, st))
        if "nbt" in b:
            s.block_data[(x, y, z)] = b["nbt"]
    return s


def load_structure(path, sub_level=None, merge_sub_levels=True):
    """Load a schematic.

    sub_level=None: the normal blocks (plus any unrotated ships attached to them, merged onto
    the grid), or, if the file holds only Aeronautics ships, the largest ship.
    sub_level=N: ship N (0-based). The result's .sub_levels lists every ship.
    """
    root = nbt.load(path)
    main = _parse(root)
    main.sub_levels = [_parse(sl) for sl in root.get("sub_levels", [])]
    if sub_level is not None:
        if not 0 <= sub_level < len(main.sub_levels):
            raise ValueError(f"schematic has {len(main.sub_levels)} sub-level(s); no sub-level {sub_level}")
        chosen = main.sub_levels[sub_level]
        chosen.sub_levels = main.sub_levels
        return chosen
    if not main.blocks and main.sub_levels:
        chosen = max(main.sub_levels, key=lambda s: len(s.blocks))
        chosen.sub_levels = main.sub_levels
        return chosen
    if merge_sub_levels and main.sub_levels:
        _merge_aligned(main, root.get("sub_levels", []))
    return main


def _merge_aligned(main, raw_levels):
    """Place sub-levels that aren't rotated onto the main grid at their (rounded) position.

    A sub-level attached by a piston or sticker is saved as its own small ship; when its
    orientation is the identity its position is its grid corner in the main structure's
    coordinates (e.g. 8.00, 4.01, 2.00). Rotated sub-levels can't share the grid and stay out.
    """
    main.merged, main.skipped = [], []
    placed = []
    for i, (sl, raw) in enumerate(zip(main.sub_levels, raw_levels)):
        o, p = raw.get("orientation", {}), raw.get("position", {})
        if abs(abs(o.get("w", 1.0)) - 1.0) > 1e-3:
            main.skipped.append(i)
            continue
        off = tuple(int(round(p.get(k, 0.0))) for k in "xyz")
        placed.append((off, sl))
        main.merged.append(i)
    if not placed:
        return
    allpos = [(x, y, z) for x, y, z, _ in main.blocks] +              [(x + o[0], y + o[1], z + o[2]) for o, sl in placed for x, y, z, _ in sl.blocks]
    lo = [min(p[k] for p in allpos) for k in range(3)]
    hi = [max(p[k] for p in allpos) for k in range(3)]
    shift = [-min(0, v) for v in lo]

    def mv(x, y, z, o=(0, 0, 0)):
        return x + o[0] + shift[0], y + o[1] + shift[1], z + o[2] + shift[2]
    blocks = [(*mv(x, y, z), st) for x, y, z, st in main.blocks]
    data = {mv(*k): v for k, v in main.block_data.items()}
    main.grid_shift = tuple(shift)  # schematic coordinates -> this structure's grid
    taken = {b[:3] for b in blocks}
    for o, sl in placed:
        for x, y, z, st in sl.blocks:
            q = mv(x, y, z, o)
            if q not in taken:
                blocks.append((*q, st))
                taken.add(q)
        data.update({mv(*k, o): v for k, v in sl.block_data.items()})
    main.blocks, main.block_data = blocks, data
    main.size = tuple(max(main.size[k], hi[k] + shift[k] + 1) for k in range(3))
