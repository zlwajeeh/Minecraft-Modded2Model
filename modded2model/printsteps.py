"""3D-print block operations (fill, seal, connect, delete floating, hollow, melt).

Credit: the algorithms are ported from ObjFileManip.cpp of Eric Haines' open-source Mineways.

They work on whole blocks: every cell of the build is AIR, a BLOCK (has geometry) or FILLER
(material the steps add). Run in this order (process):

1. Fill air bubbles (checkAndRemoveBubbles): air that outside air can't reach is filled.
   - Seal off entrances (coatSurfaces, rule 2): one-block doorways and windows are plugged
     while outside air floods in, then unplugged.
   - Fill in isolated tunnels in the model's base (EXPT_SEAL_SIDE_TUNNELS): outside air only
     enters from the top of the box, so tunnels cut open by the sides or bottom get filled.
2. Connect parts sharing an edge (fixTouchingEdges) and connect corner tips
   (connectCornerTips), repeated until nothing changes. "Weld all shared edges" welds edges
   inside one part too.
3. Delete floating objects (processFloatingGroups): parts smaller than N blocks, and parts that
   are only tree (logs and leaves), except the largest part.
4. Hollow out the bottom (hollowBottomOfModel), with walls N blocks thick, and superhollow
   (hollowSeed) to carve out more.
5. Melt snow blocks (meltSnow).
"""
import math

import numpy as np
from scipy import ndimage

AIR, BLOCK, FILLER = 0, 1, 2
FACE6 = ndimage.generate_binary_structure(3, 1)
TREE_PARTS = ("_log", "_leaves", "_wood", "vine")


class Options:
    def __init__(self, fill_bubbles=False, seal_entrances=False, seal_tunnels=False,
                 connect_parts=False, connect_corners=False, weld_all_edges=False,
                 delete_floating=0, hollow_blocks=0, superhollow=False, melt_snow=False):
        self.fill_bubbles, self.seal_entrances, self.seal_tunnels = fill_bubbles, seal_entrances, seal_tunnels
        self.connect_parts, self.connect_corners, self.weld_all_edges = connect_parts, connect_corners, weld_all_edges
        self.delete_floating, self.hollow_blocks, self.superhollow = delete_floating, hollow_blocks, superhollow
        self.melt_snow = melt_snow


def process(structure, vox, opts, report=print):
    """Return (cells, stats): cells is a padded (one AIR cell all round) grid of AIR/BLOCK/FILLER
    after the operations; index it with block coordinates + 1."""
    sx, sy, sz = structure.size
    cells = np.zeros((sx + 2, sy + 2, sz + 2), np.uint8)
    names = {}
    for x, y, z, st in structure.blocks:
        if vox.block_shape(st).any():
            cells[x + 1, y + 1, z + 1] = BLOCK
            names[(x + 1, y + 1, z + 1)] = st.name
    stats = {}
    if opts.fill_bubbles:
        stats["bubble blocks filled"] = fill_air_bubbles(cells, opts.seal_entrances, opts.seal_tunnels)
    if opts.connect_parts:
        welded = tips = 0
        while True:  # do { fixTouchingEdges; connectCornerTips } while (found)
            w = fix_touching_edges(cells, opts.weld_all_edges)
            t = connect_corner_tips(cells) if opts.connect_corners else 0
            welded, tips = welded + w, tips + t
            if not (w or t):
                break
        stats["edge weld blocks"], stats["corner tip blocks"] = welded, tips
    if opts.delete_floating:
        stats["floating parts deleted"] = delete_floating(cells, names, opts.delete_floating)
    if opts.hollow_blocks:
        stats["blocks hollowed"] = hollow_bottom(cells, opts.hollow_blocks, opts.superhollow)
    if opts.melt_snow:
        snow = [p for p, n in names.items() if n == "minecraft:snow_block" and cells[p] != AIR]
        for p in snow:
            cells[p] = AIR
        stats["snow blocks melted"] = len(snow)
    for k, v in stats.items():
        if v:
            report(f"Print steps: {v} {k}")
    return cells, stats


# --- 1. fill air bubbles ----------------------------------------------------------------------
def _coat(cells):
    """coatSurfaces rule 2: an air cell with solid on both sides along x and air on both sides
    along z (or the other way round) may be a door or window: plug it for now."""
    s = cells != AIR
    plug = np.zeros_like(s)
    c = (slice(1, -1),) * 3

    def sh(dx, dz):
        return s[1 + dx:s.shape[0] - 1 + dx, 1:-1, 1 + dz:s.shape[2] - 1 + dz]
    xs, zs = sh(-1, 0) & sh(1, 0), sh(0, -1) & sh(0, 1)
    xa, za = ~sh(-1, 0) & ~sh(1, 0), ~sh(0, -1) & ~sh(0, 1)
    plug[c] = ~s[c] & ((xs & za) | (xa & zs))
    # Only plug candidates that face outside air are entrances. In a room one block deep (a car
    # cab) the room's own cells pass the same test; plugging them too would leave no room to
    # fill, so they stay air and become the bubble.
    outside = _outside_air(~s & ~plug)
    facing_out = ndimage.binary_dilation(outside, structure=FACE6)
    return plug & facing_out


def _outside_air(passable, top_only=False):
    """findGroups' surrounding air group: air connected (face neighbours) to the box border, or
    with top_only to the top of the box only."""
    lab, _ = ndimage.label(passable, structure=FACE6)
    if top_only:
        seeds = np.unique(lab[:, -1, :])
    else:
        seeds = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                          lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    return np.isin(lab, seeds[seeds > 0])


def _outside_air_tunnels(air):
    """Seal side tunnels: the box's sides and bottom are put in the surrounding air group up
    front (addVolumeToGroup), so they count as outside but carry no air; air gets in only from
    the top, and tunnels opening on the sides or bottom become bubbles."""
    inner = air.copy()
    inner[0], inner[-1], inner[:, 0], inner[:, :, 0], inner[:, :, -1] = False, False, False, False, False
    return _outside_air(inner, True) | (air & ~inner)


def fill_air_bubbles(cells, seal_entrances=False, seal_tunnels=False):
    """checkAndRemoveBubbles: every air group except the surrounding one is filled. With seal
    entrances, doorways are plugged (coatSurfaces) for a first pass, then unplugged and the
    bubbles checked again (findGroups runs twice). Returns cells filled."""
    outside = _outside_air_tunnels if seal_tunnels else _outside_air
    filled = 0
    if seal_entrances:
        plug = _coat(cells)
        air = cells == AIR
        bubbles = air & ~plug & ~outside(air & ~plug)
        cells[bubbles] = FILLER
        filled += int(bubbles.sum())
    air = cells == AIR
    bubbles = air & ~outside(air)
    cells[bubbles] = FILLER
    return filled + int(bubbles.sum())


# --- 2. connect parts --------------------------------------------------------------------------
def _groups(cells):
    lab, _ = ndimage.label(cells != AIR, structure=FACE6)
    return lab


def _obscurity(solid):
    """TouchCell obscurity: in how many of the 6 directions something blocks the view."""
    ob = np.zeros(solid.shape, np.int8)
    for ax in range(3):
        cum = np.cumsum(solid, axis=ax)
        total = np.take(cum, [-1], axis=ax)
        ob += (cum - solid) > 0           # something on the low side
        ob += (total - cum) > 0           # something on the high side
    return ob


def fix_touching_edges(cells, weld_all=False):
    """fixTouchingEdges: two solid cells meeting only along an edge (diagonal in a 2x2 square
    whose other two cells are air) are welded by filling one of those air cells. Candidates are
    filled most hidden first, then most shared, then nearest the middle; filling one settles
    every edge it touches. Returns how many cells were filled."""
    solid = cells != AIR
    lab = _groups(cells)
    count = {}
    edges = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        n = [s - 1 for s in cells.shape]

        def sl(da, db):
            s = [slice(0, n[i]) for i in range(3)]
            s[a] = slice(da, n[a] + da)
            s[b] = slice(db, n[b] + db)
            return tuple(s)
        for (p, q, e1, e2) in (((0, 0), (1, 1), (1, 0), (0, 1)), ((1, 0), (0, 1), (0, 0), (1, 1))):
            m = solid[sl(*p)] & solid[sl(*q)] & ~solid[sl(*e1)] & ~solid[sl(*e2)]
            if not weld_all:
                m &= lab[sl(*p)] != lab[sl(*q)]
            for base in np.argwhere(m):
                def at(off):
                    c = list(base)
                    c[a] += off[0]
                    c[b] += off[1]
                    return tuple(c)
                c1, c2 = at(e1), at(e2)
                edges.append((c1, c2))
                count[c1] = count.get(c1, 0) + 1
                count[c2] = count.get(c2, 0) + 1
    if not edges:
        return 0
    ob = _obscurity(solid)
    centre = np.argwhere(solid).mean(0)
    order = sorted(count, key=lambda c: (-ob[c], -count[c], float(np.linalg.norm(np.array(c) - centre))))
    by_cell = {}
    for i, (c1, c2) in enumerate(edges):
        by_cell.setdefault(c1, []).append(i)
        by_cell.setdefault(c2, []).append(i)
    done = [False] * len(edges)
    filled = 0
    for c in order:
        if count[c] <= 0 or cells[c] != AIR:
            continue
        for i in by_cell[c]:  # decrementNeighbors
            if not done[i]:
                done[i] = True
                for cc in edges[i]:
                    count[cc] -= 1
        cells[c] = FILLER
        filled += 1
    return filled


def connect_corner_tips(cells):
    """connectCornerTips: a solid cell with air below whose diagonal corner below belongs to
    another part, with the other six cells of that 2x2x2 empty, gets the cell below filled."""
    filled = 0
    lab = _groups(cells)
    nx, ny, nz = cells.shape
    for y in range(2, ny - 1):
        for x, z in np.argwhere((cells[:, y, :] != AIR) & (cells[:, y - 1, :] == AIR)):
            for ox, oz in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                tx, tz = x + ox, z + oz
                if not (0 <= tx < nx and 0 <= tz < nz):
                    continue
                if cells[tx, y - 1, tz] == AIR or lab[tx, y - 1, tz] == lab[x, y, z]:
                    continue
                others = [(x + ox * i, y - j, z + oz * k) for i in (0, 1) for j in (0, 1) for k in (0, 1)
                          if (i, j, k) not in ((0, 0, 0), (1, 1, 1))]
                if all(cells[o] == AIR for o in others):
                    cells[x, y - 1, z] = FILLER
                    filled += 1
                    break
    return filled


# --- 3. delete floating objects ----------------------------------------------------------------
def delete_floating(cells, names, smaller_than):
    """processFloatingGroups: delete parts with fewer than smaller_than blocks, and parts that
    are only tree (logs/leaves, with some leaves); the largest part (lowest on a tie) stays."""
    lab, n = ndimage.label(cells != AIR, structure=FACE6)
    if n <= 1:
        return 0
    pop = np.bincount(lab.ravel())
    pop[0] = 0
    lows = ndimage.minimum(np.indices(lab.shape)[1], lab, index=np.arange(1, n + 1))
    survivor = max(range(1, n + 1), key=lambda i: (pop[i], -lows[i - 1]))
    deleted = 0
    tree = {}
    for p, name in names.items():
        g = lab[p]
        if g:
            t = tree.setdefault(g, [True, False])
            is_tree = any(k in name for k in TREE_PARTS)
            t[0] &= is_tree
            t[1] |= "leaves" in name
    for g in range(1, n + 1):
        if g == survivor:
            continue
        only_tree, leaves = tree.get(g, [False, False])
        if pop[g] < smaller_than or (only_tree and leaves):
            cells[lab == g] = AIR
            deleted += 1
    return deleted


# --- 4. hollow out the bottom ------------------------------------------------------------------
def hollow_bottom(cells, thick, superhollow=False):
    """hollowBottomOfModel: working up from the bottom, remove each cell (inside a wall of
    `thick` cells) that has air below and solid all round it and above; posts are left at the
    corners of wide models. Superhollow (hollowSeed) then carves on from the hollowed columns.
    Returns how many cells were removed."""
    T = max(1, int(thick))
    solid_idx = np.argwhere(cells != AIR)
    if not len(solid_idx):
        return 0
    lo, hi = solid_idx.min(0), solid_idx.max(0)
    hollow = np.zeros(cells.shape, bool)  # HOLLOW_AIR_GROUP
    done = np.zeros((cells.shape[0], cells.shape[2]), int)
    add_post = (hi[0] > lo[0] + 8 + 2 * T) or (hi[2] > lo[2] + 8 + 2 * T)
    removed = 0
    nx, ny, nz = cells.shape

    def solid(x, y, z):
        return 0 <= x < nx and 0 <= y < ny and 0 <= z < nz and cells[x, y, z] != AIR

    dirs = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1))  # every face but the bottom
    for y in range(lo[1], hi[1] + 1):
        change = []
        for x in range(lo[0] + T, hi[0] - T + 1):
            for z in range(lo[2] + T, hi[2] - T + 1):
                if done[x, z]:
                    continue
                inner = (lo[0] + T < x < hi[0] - T) or (lo[2] + T < z < hi[2] - T)
                if not add_post or inner:
                    ok = (not solid(x, y - 1, z) and solid(x, y, z)
                          and all(solid(x + i, y + j, z + k) for i in (-1, 0, 1) for k in (-1, 0, 1) for j in (0, 1)
                                  if (i, j, k) != (0, 0, 0)))
                    if ok and T > 1:
                        for d in dirs:
                            for s in range(2, T + 1):
                                if not solid(x + d[0] * s, y + d[1] * s, z + d[2] * s):
                                    ok = False
                                    break
                            if not ok:
                                break
                    if ok:
                        change.append((x, y, z))
                    else:
                        done[x, z] = y
                elif not solid(x, y, z):
                    done[x, z] = y
        if not change:
            break
        for c in change:
            cells[c] = AIR
            hollow[c] = True
            removed += 1

    if superhollow:
        def open_air(x, y, z):
            return 0 <= x < nx and 0 <= y < ny and 0 <= z < nz and cells[x, y, z] == AIR and not hollow[x, y, z]

        for x in range(lo[0] + T, hi[0] - T + 1):
            for z in range(lo[2] + T, hi[2] - T + 1):
                top = max(done[x - 1, z], done[x + 1, z])  # neighbours along x, as in the original
                for y0 in range(done[x, z] + T, top):
                    stack = [(x, y0, z)]
                    while stack:
                        cx, cy, cz = stack.pop()
                        if not (0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz):
                            continue
                        if cells[cx, cy, cz] == AIR or hollow[cx, cy, cz]:
                            continue
                        ok = not any(open_air(cx + i, cy + j, cz + k)
                                     for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1))
                        if ok and T > 1:
                            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                                if any(open_air(cx + d[0] * s, cy + d[1] * s, cz + d[2] * s) for s in range(1, T)):
                                    ok = False
                                    break
                        if ok:
                            cells[cx, cy, cz] = AIR
                            hollow[cx, cy, cz] = True
                            removed += 1
                            stack += [(cx + 1, cy, cz), (cx - 1, cy, cz), (cx, cy + 1, cz), (cx, cy - 1, cz),
                                      (cx, cy, cz + 1), (cx, cy, cz - 1)]
    return removed


def hollow_blocks_for(mm, block_mm):
    """gHollowBlockThickness = ceil(wall thickness / block size)."""
    return int(math.ceil(mm / block_mm)) if mm > 0 else 0


# --- lesser blocks -----------------------------------------------------------------------------
def is_lesser(vox, state):
    """Lesser blocks are the small detailed ones (billboards and thin geometry:
    torches, flowers, rails, levers, buttons, ladders, signs...). Here: blocks whose shape fills
    less than a tenth of their cell; copycats are building material, never lesser."""
    if "copycat" in state.name:
        return False
    shape = vox.block_shape(state)
    return 0 < shape.mean() < 0.1
