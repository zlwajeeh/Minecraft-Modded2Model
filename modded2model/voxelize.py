"""Turn a structure into a solid voxel grid (res cells per block, Y up)."""
import math

import numpy as np

from . import codeshapes
from .schematic import BlockState


def _rot_matrix(axis, degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _blockstate_matrix(x_rot, y_rot):
    # Minecraft rotates the model around X first, then Y, both clockwise
    # when looking down the axis toward the origin (so y=90 turns north into east).
    rx = _rot_matrix("x", -x_rot)
    ry = _rot_matrix("y", -y_rot)
    return ry @ rx


def _element_cells(el, model_rot, res):
    """Return an (N, 3) int array of cells (in one block, 0..res-1) filled by one model element."""
    lo = np.array(el["from"], float)
    hi = np.array(el["to"], float)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    # Flat planes (e.g. cross-shaped plants, panes of zero thickness) get one cell of thickness.
    thin = hi - lo < 16 / res
    mid = (lo + hi) / 2
    lo = np.where(thin, mid - 8 / res, lo)
    hi = np.where(thin, mid + 8 / res, hi)

    el_rot = np.eye(3)
    origin = np.array([8.0, 8.0, 8.0])
    if "rotation" in el and el["rotation"].get("angle", 0):
        r = el["rotation"]
        el_rot = _rot_matrix(r["axis"], r["angle"])
        origin = np.array(r.get("origin", [8, 8, 8]), float)
    center = np.array([8.0, 8.0, 8.0])

    def to_block(p):  # element space -> block space
        p = (p - origin) @ el_rot.T + origin
        return (p - center) @ model_rot.T + center

    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    wc = to_block(corners)
    cell = 16 / res
    cmin = np.clip(np.floor(wc.min(0) / cell).astype(int), 0, res - 1)
    cmax = np.clip(np.ceil(wc.max(0) / cell).astype(int), 1, res)
    gx, gy, gz = np.meshgrid(*(np.arange(cmin[i], cmax[i]) for i in range(3)), indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    if not len(cells):
        return cells
    # Test each cell center by mapping it back into element space.
    pts = (cells + 0.5) * cell
    inv_model = model_rot  # orthonormal: inverse is transpose, applied as p @ M
    p = (pts - center) @ inv_model + center
    p = (p - origin) @ el_rot + origin
    eps = 1e-6
    inside = np.all((p >= lo - eps) & (p <= hi + eps), axis=1)
    return cells[inside]


def _mesh_shape(tris, model_rot, res):
    """Voxelize a triangle mesh (0..16 units) into a res^3 block grid, filling closed interiors."""
    from scipy import ndimage
    center = np.array([8.0, 8.0, 8.0])
    t = (tris - center) @ model_rot.T + center
    cell = 16 / res
    pts = []
    for tri in t:
        a, b, c = tri
        n = int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / (cell / 2))) + 1
        u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        keep = u + v <= 1
        u, v = u[keep], v[keep]
        pts.append(a + np.outer(u, b - a) + np.outer(v, c - a))
    p = np.floor(np.concatenate(pts) / cell).astype(int)
    p = p[np.all((p >= 0) & (p < res), axis=1)]
    shape = np.zeros((res, res, res), bool)
    shape[p[:, 0], p[:, 1], p[:, 2]] = True
    return ndimage.binary_fill_holes(shape)


def rasterize_mesh(grid, tris):
    """OR a closed triangle mesh (N, 3, 3), given in grid cell units, into grid as a solid."""
    from scipy import ndimage
    pts = []
    for a, b, c in tris:
        n = int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) * 2)) + 1
        u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        keep = u + v <= 1
        pts.append(a + np.outer(u[keep], b - a) + np.outer(v[keep], c - a))
    p = np.floor(np.concatenate(pts)).astype(int)
    p = p[np.all((p >= 0) & (p < np.array(grid.shape)), axis=1)]
    if not len(p):
        return
    p0 = p.min(0)
    local = np.zeros(tuple(p.max(0) - p0 + 1), bool)
    local[tuple((p - p0).T)] = True
    local = ndimage.binary_fill_holes(local)  # solid, not just a shell
    grid[tuple(slice(p0[i], p0[i] + local.shape[i]) for i in range(3))] |= local


class Voxelizer:
    def __init__(self, assets, res=16, overrides=None):
        self.assets = assets
        self.res = res
        self.overrides = overrides or {}
        self._shape_cache = {}
        self.unknown = set()   # blocks with no blockstate file: drawn as full cubes
        self.no_geometry = set()  # blocks whose models have no elements (rendered by code)
        self.smooth_meshes = []  # closed meshes kept as triangles, in grid cell units
        self.grid_offset = np.zeros(3, int)  # cells the grid was padded by at the low end

    def block_shape(self, state):
        from .attachments import HANDLERS
        if state in self._shape_cache:
            return self._shape_cache[state]
        res = self.res
        shape = np.zeros((res, res, res), bool)
        ov = self.overrides.get(state.name)
        if ov == "skip":
            pass
        elif ov == "full":
            shape[:] = True
        elif isinstance(ov, dict) and "boxes" in ov:
            self._fill_boxes(shape, ov["boxes"])
        elif state.name in codeshapes.ALIASES:
            shape = self.block_shape(BlockState(codeshapes.ALIASES[state.name], state.props))
        elif state.name in codeshapes.JAR_MODELS:
            model_ids, _ = codeshapes.JAR_MODELS[state.name]
            xr, yr = codeshapes.FACING_ROTATION[state.properties.get("facing", "up")]
            m = _blockstate_matrix(xr, yr)
            for mid in model_ids:
                model = self.assets.model(mid)
                if model is None:
                    self.unknown.add(mid)
                    continue
                for el in model["elements"]:
                    c = _element_cells(el, m, res)
                    if len(c):
                        shape[c[:, 0], c[:, 1], c[:, 2]] = True
        elif state.name in codeshapes.MESH_RULES:
            shape = _mesh_shape(codeshapes.MESH_RULES[state.name](state.properties), np.eye(3), res)
        elif state.name in codeshapes.RULES:
            self._fill_boxes(shape, codeshapes.RULES[state.name](state.properties))
        else:
            parts = self.assets.resolve(state)
            if parts is None:
                self.unknown.add(state.name)
                shape[:] = True
            else:
                any_el = False
                for model, xr, yr in parts:
                    m = _blockstate_matrix(xr, yr)
                    for el in model["elements"]:
                        any_el = True
                        c = _element_cells(el, m, res)
                        if len(c):
                            shape[c[:, 0], c[:, 1], c[:, 2]] = True
                    if model.get("mesh") is not None:
                        any_el = True
                        shape |= _mesh_shape(model["mesh"], m, res)
                    for part in codeshapes.extra_by_model(model.get("id"), self.assets):
                        for el in part["elements"]:
                            c = _element_cells(el, m, res)
                            if len(c):
                                shape[c[:, 0], c[:, 1], c[:, 2]] = True
                if parts:  # parts the block entity draws on top, with the same rotation
                    m = _blockstate_matrix(parts[0][1], parts[0][2])
                    extra_ids = codeshapes.extra_models(state.name) + codeshapes.state_extra_models(state)
                    extra_els = [el for mid in extra_ids for el in (self.assets.model(mid) or {}).get("elements", [])]
                    fy = codeshapes.extras_frame_yaw(state.name, state.properties.get("facing"))
                    me = m if fy is None or not extra_ids else _blockstate_matrix(0, fy)
                    if not extra_ids and state.name not in HANDLERS:  # parts only the item model shows (blades, shafts, handles)
                        extra_els = self.assets.item_extra_elements(state.name, codeshapes.item_extras_allowed(state.name))
                    for mid in codeshapes.property_parts(state):
                        for el in (self.assets.model(mid) or {}).get("elements", []):
                            c = _element_cells(el, np.eye(3), res)
                            if len(c):
                                shape[c[:, 0], c[:, 1], c[:, 2]] = True
                    for mid, rot in codeshapes.kinetic_shafts(state):
                        mk = _blockstate_matrix(*rot)
                        for el in (self.assets.model(mid) or {}).get("elements", []):
                            c = _element_cells(el, mk, res)
                            if len(c):
                                shape[c[:, 0], c[:, 1], c[:, 2]] = True
                    for mid, rot in codeshapes.north_frame_parts(state):
                        mf = _blockstate_matrix(*rot)
                        for el in (self.assets.model(mid) or {}).get("elements", []):
                            c = _element_cells(el, mf, res)
                            if len(c):
                                shape[c[:, 0], c[:, 1], c[:, 2]] = True
                    for el in extra_els:
                            c = _element_cells(el, me, res)
                            if len(c):
                                shape[c[:, 0], c[:, 1], c[:, 2]] = True
                if not any_el and not codeshapes.empty_by_design(state):
                    self.no_geometry.add(state.name)
                    shape[:] = True
        self._shape_cache[state] = shape
        return shape

    def _fill_boxes(self, shape, boxes):
        for box in boxes:
            c = _element_cells({"from": box[:3], "to": box[3:6]}, np.eye(3), self.res)
            if len(c):
                shape[c[:, 0], c[:, 1], c[:, 2]] = True

    def voxelize(self, structure):
        res = self.res
        sx, sy, sz = structure.size
        grid = np.zeros((sx * res, sy * res, sz * res), bool)
        from .progress import report
        n = max(1, len(structure.blocks))
        hidden = codeshapes.hidden_blocks(structure)
        for k, (x, y, z, state) in enumerate(structure.blocks):
            report(k / n, f"Reading block shapes ({k}/{n})")
            if (x, y, z) in hidden:  # drawn as part of a multiblock (structure_parts)
                continue
            grid[x * res:(x + 1) * res, y * res:(y + 1) * res, z * res:(z + 1) * res] |= self.block_shape(state)
        return self._add_attachments(grid, structure)

    def _add_attachments(self, grid, structure):
        """Draw geometry blocks render outside their cell (e.g. wheels), growing the grid if needed."""
        from .attachments import HANDLERS
        res = self.res
        cell = 16 / res
        meshes, smooth = [], []
        for x, y, z, state in structure.blocks:
            handler = HANDLERS.get(state.name)
            if handler:
                vox_tris, smooth_tris = handler(state, structure.block_data.get((x, y, z)), self.assets)
                shift = np.array([x, y, z], float) * 16
                if len(vox_tris):
                    meshes.append(vox_tris + shift)
                if len(smooth_tris):
                    smooth.append(smooth_tris + shift)
        from .attachments import element_tris
        for mid, m, t in codeshapes.structure_parts(structure):  # multiblocks, hydraulic rods
            part = self.assets.model(mid)
            if part is None:
                continue
            for el in part["elements"]:  # each box its own solid
                meshes.append(element_tris(el) @ m.T + t)
            if part.get("mesh") is not None and len(part["mesh"]):
                meshes.append(part["mesh"] @ m.T + t)
        self.smooth_meshes = []
        self.grid_offset = np.zeros(3, int)
        if not meshes and not smooth:
            return grid
        allpts = np.concatenate([m.reshape(-1, 3) for m in meshes + smooth])
        lo = np.floor(allpts.min(0) / cell).astype(int)
        hi = np.ceil(allpts.max(0) / cell).astype(int)
        pad_lo = np.maximum(0, -lo)
        pad_hi = np.maximum(0, hi - np.array(grid.shape))
        grid = np.pad(grid, [(int(pad_lo[i]), int(pad_hi[i])) for i in range(3)])
        offset = pad_lo
        self.grid_offset = pad_lo
        # Smooth meshes go to the output as triangles, in grid cell units.
        self.smooth_meshes = [t / cell + offset for t in smooth]
        for tris in meshes:
            rasterize_mesh(grid, tris / cell + offset)
        return grid
