"""Voxel-level fixes that make a model printable (the 3D print options)."""
import numpy as np
from scipy import ndimage

FACE_CONNECTED = ndimage.generate_binary_structure(3, 1)


def fill_enclosed(grid):
    """Fill sealed internal cavities (rooms closed off on all sides) so they print solid."""
    return ndimage.binary_fill_holes(grid)


def solidify(grid, gap_cells):
    """Fill the whole inside of a model, even when doors, hatches or windows open it up.

    Openings up to about 2 * gap_cells wide are plugged: growing the model by gap_cells
    closes them, which tells us where "outside" is. Outside then spreads back toward the
    model through empty cells only (never through walls), and every empty cell it can't
    reach is filled. Only hidden space is filled, so the outer surface doesn't move;
    openings show as recesses up to gap_cells deep.
    """
    if gap_cells <= 0:
        return fill_enclosed(grid)
    cube = np.ones((3, 3, 3), bool)  # square growth keeps flat faces and edges crisp
    pad = gap_cells + 1
    g = np.pad(grid, pad)
    grown = ndimage.binary_dilation(g, structure=cube, iterations=gap_cells)
    outside = ~ndimage.binary_fill_holes(grown)
    outside = ndimage.binary_dilation(outside, structure=cube, iterations=gap_cells, mask=~g)
    s = slice(pad, -pad)
    return grid | ~outside[s, s, s]


def _outside(air):
    """Air cells connected (6-way) to the border of the array."""
    labels, _ = ndimage.label(air, structure=FACE_CONNECTED)
    border = np.unique(np.concatenate([labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(),
                                       labels[:, -1].ravel(), labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]))
    return np.isin(labels, border[border > 0])


def seal_openings(solid, width):
    """The "seal off entrances" coating: air blocks in a doorway or window up to `width`
    blocks wide (solid on both sides along x with air in front and behind along z, or the
    same with x and z swapped) become temporary plugs. Roof openings are left open."""
    plug = np.zeros_like(solid)
    if width < 1:
        return plug
    air = ~solid
    pad = width + 1
    s = np.pad(solid, ((pad, pad), (0, 0), (pad, pad)))
    a = np.pad(air, ((pad, pad), (0, 0), (pad, pad)), constant_values=True)
    nx, _, nz = solid.shape

    def shifted(arr, axis, d):
        sl = [slice(None)] * 3
        sl[axis] = slice(pad + d, pad + d + (nx if axis == 0 else nz))
        other = 2 if axis == 0 else 0
        sl[other] = slice(pad, pad + (nz if other == 2 else nx))
        return arr[tuple(sl)]

    for axis, other in ((0, 2), (2, 0)):
        across = np.zeros_like(solid)
        for lo in range(1, width + 1):  # solid `lo` blocks one way and `hi` the other, gap <= width
            for hi in range(1, width + 2 - lo):
                across |= shifted(s, axis, -lo) & shifted(s, axis, hi)
        if width == 1:  # default rule: and open in front and behind
            across &= shifted(a, other, -1) & shifted(a, other, 1)
        plug |= air & across
    return plug


def block_fill(block_solid, seal_width=1):
    """Block-level fill. Returns (filled, outside) block grids (same shape as
    block_solid): filled = solid blocks plus every air block outside air can't reach once
    doors and windows up to seal_width wide are temporarily plugged; outside = the rest."""
    solid = np.pad(block_solid, 1)
    plug = seal_openings(solid, seal_width)
    bubbles = ~solid & ~plug & ~_outside(~(solid | plug))
    filled = solid | bubbles
    # plugs come back out; any plug no longer reachable from outside stays filled
    filled |= ~filled & ~_outside(~filled)
    s = (slice(1, -1),) * 3
    return filled[s], ~filled[s]


def thicken(grid, min_cells):
    """Grow parts thinner than min_cells until they reach that thickness, and no further.

    Works one axis at a time: a run of solid cells shorter than min_cells along an axis
    grows by one cell on each side along that axis only, repeated until long enough. So a
    thin roof only gets thicker vertically and its outline stays where it was.
    """
    if min_cells <= 1:
        return grid
    grid = grid.copy()
    for axis in range(3):
        line = np.zeros((3, 3, 3), bool)
        idx = [1, 1, 1]
        for i in range(3):
            idx[axis] = i
            line[tuple(idx)] = True
        shape = [1, 1, 1]
        shape[axis] = min_cells
        run = np.ones(shape, bool)
        for _ in range((min_cells + 1) // 2):
            thin = grid & ~ndimage.binary_opening(grid, structure=run)
            if not thin.any():
                break
            grid |= ndimage.binary_dilation(thin, structure=line)
    return grid


def hollow(grid, wall_cells):
    """Empty the inside of solid regions, leaving walls wall_cells thick."""
    inner = ndimage.binary_erosion(grid, iterations=wall_cells, border_value=0)
    return grid & ~inner


def fix_nonmanifold(grid, max_passes=10):
    """Fill cells where two solids touch only along an edge or at a corner.

    Those contacts make edges shared by four faces, which slicers treat as broken
    geometry. Filling one neighbor cell turns them into a proper solid connection.
    """
    g = np.pad(grid, 1)
    for _ in range(max_passes):
        changed = False
        # Edge contacts: in each axis-aligned 2x2 square, diagonal cells solid, the other two empty.
        for a, b in ((0, 1), (0, 2), (1, 2)):
            def sl(da, db):
                s = [slice(None)] * 3
                s[a] = slice(da, g.shape[a] - 1 + da)
                s[b] = slice(db, g.shape[b] - 1 + db)
                return tuple(s)
            p00, p11, p10, p01 = g[sl(0, 0)], g[sl(1, 1)], g[sl(1, 0)], g[sl(0, 1)]
            m = p00 & p11 & ~p10 & ~p01
            if m.any():
                g[sl(1, 0)] |= m
                changed = True
            m = p10 & p01 & ~p00 & ~p11
            if m.any():
                g[sl(0, 0)] |= m
                changed = True
        # Corner contacts: opposite corners of a 2x2x2 cube solid, all six others empty.
        def c(dx, dy, dz):
            return g[dx:g.shape[0] - 1 + dx, dy:g.shape[1] - 1 + dy, dz:g.shape[2] - 1 + dz]
        cube = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
        for d in cube[:4]:
            opp = tuple(1 - v for v in d)
            others = [q for q in cube if q not in (d, opp)]
            m = c(*d) & c(*opp)
            for q in others:
                m &= ~c(*q)
            if m.any():
                c(*others[0])[...] |= m
                changed = True
        if not changed:
            break
    return g[1:-1, 1:-1, 1:-1]
