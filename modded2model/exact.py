"""Exact export: each block's own model geometry (Blockbench elements, OBJ meshes) as triangles.

Every element becomes its own closed box, at its exact size and rotation, so nothing is
rounded to a grid. The boxes overlap where parts touch; slicers merge overlapping closed
shells of one object, which is how they're meant to be printed.

The voxel grid is still built alongside, only to work out the interior fill (solidify)
and which pieces are loose; that fill sits behind the exact surface.
"""
import numpy as np

from . import codeshapes
from .attachments import HANDLERS, element_tris
from .schematic import BlockState
from .voxelize import _blockstate_matrix, rasterize_mesh

CENTER = np.array([8.0, 8.0, 8.0])
FULL_BOX = {"from": [0, 0, 0], "to": [16, 16, 16]}


def _grow(el, min_units):
    """Make every side of an element at least min_units long (flat planes become printable)."""
    if min_units <= 0:
        return el
    lo, hi = np.array(el["from"], float), np.array(el["to"], float)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    short = hi - lo < min_units
    if not short.any():
        return el
    mid = (lo + hi) / 2
    lo = np.where(short, mid - min_units / 2, lo)
    hi = np.where(short, mid + min_units / 2, hi)
    return dict(el, **{"from": lo.tolist(), "to": hi.tolist()})


def _rotated(tris, m):
    return (tris - CENTER) @ m.T + CENTER


class ExactModels:
    def __init__(self, voxelizer, min_units=0.0):
        self.vox = voxelizer  # shares assets, overrides and the unknown/no_geometry reports
        self.min_units = min_units
        self._cache = {}

    def _boxes(self, boxes):
        return [element_tris(_grow({"from": b[:3], "to": b[3:6]}, self.min_units)) for b in boxes]

    def _model(self, model, m):
        parts = [_rotated(element_tris(_grow(el, self.min_units)), m) for el in model["elements"]]
        if model.get("mesh") is not None:
            parts.append(_rotated(close_holes(model["mesh"]), m))
        return parts

    def block_tris(self, state):
        """Triangles (N, 3, 3) of one block in 0..16 units."""
        if state in self._cache:
            return self._cache[state]
        assets, ov = self.vox.assets, self.vox.overrides.get(state.name)
        parts = []
        if ov == "skip":
            pass
        elif ov == "full":
            parts = [element_tris(FULL_BOX)]
        elif isinstance(ov, dict) and "boxes" in ov:
            parts = self._boxes(ov["boxes"])
        elif state.name in codeshapes.ALIASES:
            parts = [self.block_tris(BlockState(codeshapes.ALIASES[state.name], state.props))]
        elif state.name in codeshapes.JAR_MODELS:
            model_ids, _ = codeshapes.JAR_MODELS[state.name]
            m = _blockstate_matrix(*codeshapes.FACING_ROTATION[state.properties.get("facing", "up")])
            for mid in model_ids:
                model = assets.model(mid)
                if model is not None:
                    parts += self._model(model, m)
        elif state.name in codeshapes.MESH_RULES:
            parts = [codeshapes.MESH_RULES[state.name](state.properties)]
        elif state.name in codeshapes.RULES:
            parts = self._boxes(codeshapes.RULES[state.name](state.properties))
        else:
            resolved = assets.resolve(state)
            if resolved is not None:
                for model, xr, yr in resolved:
                    parts += self._model(model, _blockstate_matrix(xr, yr))
                    for part in codeshapes.extra_by_model(model.get("id"), assets):
                        parts += self._model(part, _blockstate_matrix(xr, yr))
                if resolved:  # parts the block entity draws on top, with the same rotation
                    m = _blockstate_matrix(resolved[0][1], resolved[0][2])
                    fy = codeshapes.extras_frame_yaw(state.name, state.properties.get("facing"))
                    me = m if fy is None else _blockstate_matrix(0, fy)
                    extra_ids = codeshapes.extra_models(state.name) + codeshapes.state_extra_models(state)
                    for mid in extra_ids:
                        extra = assets.model(mid)
                        if extra is not None:
                            parts += self._model(extra, me)
                    for mid in codeshapes.property_parts(state):
                        part = assets.model(mid)
                        if part is not None:
                            parts += self._model(part, np.eye(3))
                    for mid, rot in codeshapes.kinetic_shafts(state):
                        part = assets.model(mid)
                        if part is not None:
                            parts += self._model(part, _blockstate_matrix(*rot))
                    for mid, mo, sh in codeshapes.offset_parts(state, resolved[0][2], resolved[0][1]):
                        part = assets.model(mid)
                        if part is not None:
                            parts += [t + sh for t in self._model(part, mo)]
                    for mid, rot in codeshapes.north_frame_parts(state):
                        part = assets.model(mid)
                        if part is not None:
                            parts += self._model(part, _blockstate_matrix(*rot))
                    if not extra_ids and state.name not in HANDLERS:  # parts only the item model shows (blades, shafts, handles)
                        parts += self._model({"elements": assets.item_extra_elements(state.name, codeshapes.item_extras_allowed(state.name))}, m)
            if not parts and state.name not in codeshapes.JAR_MODELS:  # unknown block or drawn by code: same stand-in as the voxel path
                parts = [element_tris(FULL_BOX)]
        parts = [p for p in parts if len(p)]
        tris = np.concatenate(parts) if parts else np.zeros((0, 3, 3))
        self._cache[state] = tris
        return tris


def _filler_boxes(cells):
    """Merge filled cells into boxes: runs along z, identical runs stacked along y, then
    identical rectangles stacked along x."""
    boxes = []
    if not cells.any():
        return boxes
    nx, ny, nz = cells.shape
    open_ = {}  # (y0, y1, z0, z1) -> x where the box started
    for x in range(nx + 1):
        rects = set()
        if x < nx:
            runs = {}  # (z0, z1) -> list of y
            for y in range(ny):
                row = cells[x, y]
                if not row.any():
                    continue
                d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
                for z0, z1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                    runs.setdefault((z0, z1), []).append(y)
            for (z0, z1), ys in runs.items():
                start = prev = ys[0]
                for y in ys[1:] + [None]:
                    if y is not None and y == prev + 1:
                        prev = y
                        continue
                    rects.add((start, prev + 1, z0, z1))
                    if y is not None:
                        start = prev = y
        for r in list(open_):
            if r not in rects:
                y0, y1, z0, z1 = r
                boxes.append((open_.pop(r), y0, z0, x, y1, z1))
        for r in rects:
            open_.setdefault(r, x)
    return boxes
    nx, ny, nz = cells.shape
    for x in range(nx):
        runs = {}  # (z0, z1) -> list of y
        for y in range(ny):
            row = cells[x, y]
            if not row.any():
                continue
            d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
            for z0, z1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                runs.setdefault((z0, z1), []).append(y)
        for (z0, z1), ys in runs.items():
            start = prev = ys[0]
            for y in ys[1:] + [None]:
                if y is not None and y == prev + 1:
                    prev = y
                    continue
                boxes.append((x, start, z0, x + 1, prev + 1, z1))
                if y is not None:
                    start = prev = y
    return boxes


def close_holes(tris):
    """Cap the open holes of a triangle mesh (OBJ parts such as valve handles leave the ends of
    their spokes open) so it prints as a closed solid: each loop of boundary edges gets a fan
    of triangles facing the same way as the mesh around it."""
    tris = np.asarray(tris, float)
    if not len(tris):
        return tris
    pts, vid = np.unique(np.round(tris.reshape(-1, 3), 5), axis=0, return_inverse=True)
    vid = vid.reshape(-1, 3)
    directed = {}
    for a, b, c in vid:
        for e in ((a, b), (b, c), (c, a)):
            directed[e] = directed.get(e, 0) + 1
    nxt = {}
    for (a, b), n in directed.items():
        if (b, a) not in directed:  # boundary edge
            nxt.setdefault(a, []).append(b)
    caps = []
    while nxt:
        start = next(iter(nxt))
        loop, cur = [start], start
        while True:
            outs = nxt.get(cur)
            if not outs:
                break
            b = outs.pop()
            if not outs:
                del nxt[cur]
            if b == start:
                break
            loop.append(b)
            cur = b
        if len(loop) >= 3:
            p = pts[loop]
            c = p.mean(0)
            # the boundary runs one way around the hole; the cap runs the other way
            for i in range(len(p)):
                caps.append([p[(i + 1) % len(p)], p[i], c])
    return np.concatenate([tris, np.array(caps)]) if caps else tris


def _shells(tris):
    """Split triangles into groups that share vertices (each element box is one closed shell)."""
    if len(tris) <= 12:
        return [tris]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    _, vid = np.unique(np.round(tris.reshape(-1, 3), 4), axis=0, return_inverse=True)
    vid = vid.reshape(-1, 3)
    n = len(tris)
    rows = np.repeat(np.arange(n), 3)
    g = coo_matrix((np.ones(3 * n), (rows, vid.ravel() + n)), shape=(n + vid.max() + 1,) * 2)
    _, lab = connected_components(g, directed=False)
    lab = lab[:n]
    return [tris[lab == i] for i in np.unique(lab)]


def assemble(structure, exact, grid, filled, keep_cells, cell_units, expo=None):
    """All output triangles in 1/16-block units.

    grid: the voxel model; filled: grid after fill/solidify (extra cells become filler boxes);
    keep_cells: cells of pieces to keep (others' blocks are dropped); cell_units: 16 / res.
    """
    vox = exact.vox
    res = vox.res
    off = vox.grid_offset
    full = {}
    states = {(int(x), int(y), int(z)): st for x, y, z, st in structure.blocks}
    for x, y, z, st in structure.blocks:
        full[(x, y, z)] = bool(vox.block_shape(st).all())
    out = []
    from .progress import report
    n = max(1, len(structure.blocks))
    hidden = codeshapes.hidden_blocks(structure)
    for k, (x, y, z, st) in enumerate(structure.blocks):
        report(k / n, f"Building print model ({k}/{n} blocks)")
        if (x, y, z) in hidden:  # drawn as part of a multiblock (structure_parts)
            continue
        # Skip solid cubes buried on all six sides: they can't be seen and only add size.
        if full[(x, y, z)] and all(full.get((x + dx, y + dy, z + dz)) for dx, dy, dz in
                                   ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))):
            continue
        sl = tuple(slice(off[i] + p * res, off[i] + (p + 1) * res) for i, p in enumerate((x, y, z)))
        if keep_cells is not None and not keep_cells[sl].any() and grid[sl].any():
            continue  # this block belongs to a dropped loose piece
        tris = exact.block_tris(st)
        if len(tris):
            out.append(tris + np.array([x, y, z], float) * 16)
        if st.name in codeshapes.ATTACHMENT_BLOCKS:
            def nb(side, p=(x, y, z)):
                d = codeshapes.PIPE_DIRS[side]
                return states.get((p[0] + d[0], p[1] + d[1], p[2] + d[2]))
            for mid in codeshapes.pipe_attachments(st, nb):
                part = vox.assets.model(mid)
                if part is not None:
                    pt = exact._model(part, np.eye(3))
                    if pt:
                        out.append(np.concatenate(pt) + np.array([x, y, z], float) * 16)
        for mid, mo, sh in codeshapes.data_parts(st, structure.block_data.get((x, y, z))):
            part = vox.assets.model(mid)
            if part is not None:
                pt = exact._model(part, mo)
                if pt:
                    out.append(np.concatenate(pt) + sh + np.array([x, y, z], float) * 16)
        handler = HANDLERS.get(st.name)
        if handler:
            vox_tris, smooth_tris = handler(st, structure.block_data.get((x, y, z)), vox.assets)
            for t in (vox_tris, smooth_tris):
                if len(t):
                    out.append(t + np.array([x, y, z], float) * 16)
    for mid, m, t in codeshapes.structure_parts(structure):
        part = vox.assets.model(mid)
        if part is not None:
            c = np.array([8.0, 8, 8])
            pt = exact._model(part, m)  # turned about the block centre
            if pt:
                out.append(np.concatenate(pt) - c + m @ c + t)
    removed_cells = None
    if expo is not None and out:
        # Print culling works per closed shell (one model element box, or one OBJ part): a shell
        # is dropped only when none of its faces can be seen from outside, so whatever remains is
        # still made of closed shells (watertight).
        from .visibility import visible
        shells = [c for t in out for c in _shells(t)]
        vis = visible(np.concatenate(shells), expo)
        kept, gone, k = [], [], 0
        for t in shells:
            (kept if vis[k:k + len(t)].any() else gone).append(t)
            k += len(t)
        out = kept
        if gone:  # where the model is filled, fill the space the dropped parts took
            removed_cells = np.zeros_like(grid)
            rasterize_mesh(removed_cells, np.concatenate(gone) / cell_units + np.asarray(off))
    extra = filled & ~grid
    if removed_cells is not None:
        extra |= filled & removed_cells
    for x0, y0, z0, x1, y1, z1 in _filler_boxes(extra):
        lo = (np.array([x0, y0, z0]) - off) * cell_units
        hi = (np.array([x1, y1, z1]) - off) * cell_units
        out.append(element_tris({"from": lo.tolist(), "to": hi.tolist()}))
    return np.concatenate(out) if out else np.zeros((0, 3, 3))
