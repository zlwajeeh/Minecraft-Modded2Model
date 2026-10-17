"""Block swaps and removals applied to a structure before exporting.

- Drive-by-Wire backup blocks (control backups hidden inside vehicles) are removed.
- Valkyrien Skies blocks from older builds become their Aeronautics or vanilla look-alikes.
- Blocks the export can't draw (no blockstate, or drawn only by code) are kept as full cubes
  or left out.
"""
from .schematic import BlockState, Structure

BACKUP_BLOCKS = ("drivebywire:backup_block",)
COLORS = ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "light_gray",
          "cyan", "purple", "blue", "brown", "green", "red", "black")


def vs_replacement(st):
    """Aeronautics/vanilla stand-in for a Valkyrien Skies (Eureka) block, or None."""
    ns, _, path = st.name.partition(":")
    if ns != "vs_eureka":
        return None
    if path.endswith("_ship_helm"):  # helm -> stairs of the same wood, facing the same way
        wood = path[:-len("_ship_helm")]
        facing = st.properties.get("facing", "north")
        return BlockState(f"minecraft:{wood}_stairs", (("facing", facing), ("half", "bottom"),
                                                        ("shape", "straight"), ("waterlogged", "false")))
    if path.endswith("_balloon") or path == "balloon":  # balloon -> Aeronautics envelope
        color = path[:-len("_balloon")] if path.endswith("_balloon") else "white"
        return BlockState(f"aeronautics:{color if color in COLORS else 'white'}_envelope", ())
    if path == "floater":
        return BlockState("minecraft:brown_wool", ())
    return None


def plain_full(vox, state):
    """A plain solid block: one whole-cell cube with all six faces and opaque textures, and
    nothing else drawn (no shafts, cogs or other parts)."""
    from .visibility import _opaque
    from . import codeshapes
    if not vox.block_shape(state).all():
        return False
    if (codeshapes.KINETIC_SHAFTS.get(state.name) or codeshapes.NORTH_FRAME_MODELS.get(state.name)
            or state.name in codeshapes.JAR_MODELS):
        return False
    resolved = vox.assets.resolve(state) or []
    if state.name in codeshapes.RULES:  # copycats: a full copycat block is plain
        return True
    elements = [el for m, _, _ in resolved for el in m.get("raw_elements", m["elements"])]
    return len(elements) == 1 and _opaque(vox.assets, state)


def interior_blocks(structure, vox, seal_width=1):
    """Positions of blocks inside the model that aren't plain solid blocks (cogwheels,
    gearboxes, shafts, machines, furniture...). Done like the interior fill: every block with
    geometry is a wall, doorways and windows up to seal_width blocks wide are sealed, and outside
    air floods in; a block that doesn't touch that outside air is inside."""
    import numpy as np
    from scipy import ndimage
    from .printprep import block_fill
    solid = np.zeros(structure.size, bool)
    for x, y, z, st in structure.blocks:
        solid[x, y, z] = bool(vox.block_shape(st).any())
    _, outside = block_fill(solid, max(1, int(round(seal_width))))
    touching = ndimage.binary_dilation(np.pad(outside, 1, constant_values=True),
                                       structure=ndimage.generate_binary_structure(3, 1))[1:-1, 1:-1, 1:-1]
    cache, inner = {}, set()
    for x, y, z, st in structure.blocks:
        if touching[x, y, z]:
            continue
        if st not in cache:
            cache[st] = plain_full(vox, st)
        if not cache[st]:
            inner.add((x, y, z))
    return inner


SOLID_STANDIN = "minecraft:gray_concrete"  # what "make solid" turns inside machinery into


def clean(structure, vox, remove_backups=True, replace_vs=True, missing="cube", hollow=False, interior=None):
    """Return (structure, report lines). missing: "cube" (draw unknown blocks as full cubes)
    or "skip" (leave them out). interior: what to do with blocks inside the hull that aren't
    plain solid blocks (see interior_blocks): "keep", "hollow" (remove them, leaving the inside
    hollow) or "solid" (replace each with a plain solid block). hollow=True means "hollow"."""
    interior = interior or ("hollow" if hollow else "keep")
    blocks, notes = [], []
    removed, swapped, skipped = 0, {}, {}
    for x, y, z, st in structure.blocks:
        if remove_backups and st.name in BACKUP_BLOCKS:
            removed += 1
            continue
        if replace_vs:
            new = vs_replacement(st)
            if new is not None:
                swapped[st.name] = new.name
                st = new
        if missing == "skip":
            vox.block_shape(st)
            if st.name in vox.unknown or st.name in vox.no_geometry:
                skipped[st.name] = skipped.get(st.name, 0) + 1
                continue
        blocks.append((x, y, z, st))
    if removed:
        notes.append(f"Removed {removed} backup block(s)")
    for a, b in sorted(swapped.items()):
        notes.append(f"Replaced {a} with {b}")
    if skipped:
        notes.append("Left out blocks with no model: " + ", ".join(f"{k} ({n})" for k, n in sorted(skipped.items())))
        # these were only noted to decide; don't report them again as drawn as cubes
        vox.unknown -= set(skipped)
        vox.no_geometry -= set(skipped)
    if interior in ("hollow", "solid"):
        inner = interior_blocks(Structure(structure.size, blocks), vox)
        if inner and interior == "hollow":
            blocks = [b for b in blocks if b[:3] not in inner]
            notes.append(f"Hollowed the interior: removed {len(inner)} non-solid block(s) inside the hull")
        elif inner:
            solid = BlockState(SOLID_STANDIN, ())
            blocks = [(x, y, z, solid if (x, y, z) in inner else st) for x, y, z, st in blocks]
            notes.append(f"Made the interior solid: replaced {len(inner)} non-solid block(s) inside the hull "
                         f"with solid blocks")
    keep = {(x, y, z) for x, y, z, _ in blocks}
    out = Structure(structure.size, blocks, {k: v for k, v in structure.block_data.items() if k in keep},
                    structure.sub_levels, structure.merged, structure.skipped)
    return out, notes
