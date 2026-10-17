"""Textured export for Blender: a binary glTF (.glb) with every face, UV and texture.

Faces follow Minecraft's model rules: each Blockbench element face has its texture, UV
rectangle and UV rotation; faces marked with a cullface are dropped when a full block
covers them. OBJ models keep their own UVs. Copycats show the block they copy.
Units: 1 block = 1 metre, Y up, same axes as Minecraft. Textures are pixel art, so the
samplers use nearest filtering; cut-out textures (glass, grates) use alpha masking.
"""
import json
import math
import struct

import numpy as np

from . import codeshapes
from .connected import ConnectedTextures
from .attachments import HANDLERS
from .schematic import BlockState
from .voxelize import _blockstate_matrix, _rot_matrix

CENTER = np.array([8.0, 8.0, 8.0])
DIRS = {"north": (0, 0, -1), "south": (0, 0, 1), "west": (-1, 0, 0), "east": (1, 0, 0),
        "up": (0, 1, 0), "down": (0, -1, 0)}
FOLIAGE = (0.28, 0.71, 0.09)
GRASS = (0.47, 0.75, 0.35)


def _corners(face, lo, hi):
    """Face corners in texture order: top-left, top-right, bottom-right, bottom-left (seen from outside)."""
    (x1, y1, z1), (x2, y2, z2) = lo, hi
    return {
        "north": [(x2, y2, z1), (x1, y2, z1), (x1, y1, z1), (x2, y1, z1)],
        "south": [(x1, y2, z2), (x2, y2, z2), (x2, y1, z2), (x1, y1, z2)],
        "west": [(x1, y2, z1), (x1, y2, z2), (x1, y1, z2), (x1, y1, z1)],
        "east": [(x2, y2, z2), (x2, y2, z1), (x2, y1, z1), (x2, y1, z2)],
        "up": [(x1, y2, z1), (x2, y2, z1), (x2, y2, z2), (x1, y2, z2)],
        "down": [(x1, y1, z2), (x2, y1, z2), (x2, y1, z1), (x1, y1, z1)],
    }[face]


def _default_uv(face, lo, hi):
    (x1, y1, z1), (x2, y2, z2) = lo, hi
    return {
        "north": [16 - x2, 16 - y2, 16 - x1, 16 - y1], "south": [x1, 16 - y2, x2, 16 - y1],
        "west": [z1, 16 - y2, z2, 16 - y1], "east": [16 - z2, 16 - y2, 16 - z1, 16 - y1],
        "up": [x1, z1, x2, z2], "down": [x1, 16 - z2, x2, 16 - z1],
    }[face]


def _uv_corners(uv, rotation):
    u1, v1, u2, v2 = uv
    c = [(u1, v1), (u2, v1), (u2, v2), (u1, v2)]
    k = (int(rotation) // 90) % 4
    return c[k:] + c[:k]  # rotating the texture clockwise shifts which corner each vertex gets


def resolve_texture(ref, textures, fallback=None):
    seen = 0
    while ref and ref.startswith("#") and seen < 10:
        ref = textures.get(ref[1:]) or (fallback or {}).get(ref[1:])
        seen += 1
    if not ref or ref.startswith("#"):
        return None
    return ref if ":" in ref else "minecraft:" + ref


_QUAD_TRIS = (0, 3, 2, 0, 2, 1)  # a quad's two triangles, counter-clockwise from outside


class Faces:
    """Collects textured quads/triangles in 1/16-block units."""

    def __init__(self):
        self.groups = {}  # (texture id, tint) -> [positions (N,3), uvs (N,2)] as triangle lists
        self.quads = []  # axis-aligned quads: (group key, index in group, axis, side, plane, rect)
        self.keep = set()  # (group key, index) chunks never removed by cull_invisible
        self.block = None  # block being added; exact overlaps within one block are resolved
        self._covering = {}  # (block, axis, side, plane, rect) -> (group key, index)

    def add_tris(self, texture, tint, pos, uv):
        g = self.groups.setdefault((texture, tint), [[], []])
        g[0].append(np.asarray(pos, float).reshape(-1, 3))
        g[1].append(np.asarray(uv, float).reshape(-1, 2))
        return (texture, tint), len(g[0]) - 1

    def add_quad(self, texture, tint, corners, uvs, cullable=True, overlay=False):
        """overlay: the texture has see-through pixels, so a face exactly under it stays."""
        # plain floats: numpy is slow for one 4x3 quad, and this runs for every face
        c = [[float(v) for v in p] for p in (corners.tolist() if hasattr(corners, "tolist") else corners)]
        t = [(float(u) / 16.0, float(v) / 16.0) for u, v in (uvs.tolist() if hasattr(uvs, "tolist") else uvs)]
        key, i = self.add_tris(texture, tint, [c[k] for k in _QUAD_TRIS], [t[k] for k in _QUAD_TRIS])
        (x0, y0, z0), (x1, y1, z1), (x3, y3, z3) = c[0], c[1], c[3]
        ux, uy, uz, vx, vy, vz = x3 - x0, y3 - y0, z3 - z0, x1 - x0, y1 - y0, z1 - z0
        n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
        ln = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        if ln <= 1e-9:
            return
        an = (abs(n[0]), abs(n[1]), abs(n[2]))
        axis = 0 if an[0] >= an[1] and an[0] >= an[2] else (1 if an[1] >= an[2] else 2)
        if an[axis] / ln <= 0.9999:
            return  # not axis-aligned
        rect = tuple(round(f(p[a_] for p in c), 3) for a_ in range(3) if a_ != axis for f in (min, max))
        plane = round(c[0][axis], 3)
        side = 1 if n[axis] > 0 else -1
        if self.block is not None:
            k = (self.block, axis, side > 0, plane, rect)
            prev = self._covering.get(k)
            if prev is not None and not overlay:
                # Two faces of one block exactly on top of each other: Minecraft draws the
                # later one over the earlier (e.g. a thruster's mirrored nozzle end), so
                # keep only that instead of letting them flicker
                pk, pi = prev
                self.groups[pk][0][pi] = np.zeros((0, 3))
                self.groups[pk][1][pi] = np.zeros((0, 2))
            self._covering[k] = (key, i)
        if cullable:  # axis-aligned: remember it for hidden-face culling
            self.quads.append((key, i, axis, side, plane, rect, self.block))

    def cull_invisible(self, expo):
        """Drop triangles that can't be seen from outside (see visibility.visible)."""
        from .visibility import visible
        keys, chunks = [], []
        for key, (pos, uv) in self.groups.items():
            for i, p in enumerate(pos):
                if len(p) and (key, i) not in self.keep:
                    keys.append((key, i))
                    chunks.append(p.reshape(-1, 3, 3))
        if not chunks:
            return 0
        allt = np.concatenate(chunks)
        vis = visible(allt, expo)
        removed, k = 0, 0
        for (key, i), t in zip(keys, chunks):
            m = vis[k:k + len(t)]
            k += len(t)
            if not m.all():
                pos, uv = self.groups[key]
                removed += int((~m).sum())
                pos[i] = t[m].reshape(-1, 3)
                uv[i] = uv[i].reshape(-1, 3, 2)[m].reshape(-1, 2)
        return removed

    def separate_back_to_back(self, gap=0.01):
        """Back-to-back faces of one block on the same rectangle (an inside-out box's wall
        against the box it lines, like a thruster nozzle or scaffold frame) are both meant to be
        seen, each from its own side. Push each a hair along its normal so renderers that draw
        both sides (Blender, Cycles) don't flicker between them."""
        by_rect = {}
        for q in self.quads:
            by_rect.setdefault((q[6], q[2], q[4], q[5]), []).append(q)
        moved = 0
        for qs in by_rect.values():
            if len({q[3] for q in qs}) < 2 or qs[0][6] is None:
                continue
            for key, i, axis, side, *_ in qs:
                p = self.groups[key][0][i]
                if len(p):
                    p = p.copy()
                    p[:, axis] += side * gap
                    self.groups[key][0][i] = p
                    moved += 1
        return moved

    def cull_hidden(self, full):
        """Drop faces nobody can see: faces pressed against a full neighbouring block, and pairs
        of back-to-back faces covering the same rectangle (where two blocks or parts touch).
        full: (x, y, z) block -> is a full cube. Returns the number of quads removed."""
        drop = set()
        by_rect = {}
        for q in self.quads:
            key, i, axis, side, plane, rect, _ = q
            by_rect.setdefault((axis, plane, rect), []).append(q)
            if abs(plane / 16 - round(plane / 16)) < 1e-6:  # on a block boundary
                (a0, a1), (b0, b1) = rect[0:2], rect[2:4]
                others = [a for a in range(3) if a != axis]
                block = [0, 0, 0]
                block[axis] = int(round(plane / 16)) + (0 if side > 0 else -1)
                # the face must lie within one neighbouring block's face to be fully covered by it
                cells = [int(np.floor((a0 + 1e-6) / 16)), int(np.floor((b0 + 1e-6) / 16))]
                if np.floor((a1 - 1e-6) / 16) == cells[0] and np.floor((b1 - 1e-6) / 16) == cells[1]:
                    block[others[0]], block[others[1]] = cells
                    if full.get(tuple(block)):
                        drop.add((key, i))
        for qs in by_rect.values():
            pos = [q for q in qs if q[3] > 0]
            neg = [q for q in qs if q[3] < 0]
            for a in pos:  # each facing pair from two different blocks hides both
                b = next((b for b in neg if b[6] != a[6] and (b[0], b[1]) not in drop), None)
                if b is not None:
                    drop.add((a[0], a[1]))
                    drop.add((b[0], b[1]))
        for key, i in drop:
            g = self.groups[key]
            g[0][i] = np.zeros((0, 3))
            g[1][i] = np.zeros((0, 2))
        self.quads = [q for q in self.quads if (q[0], q[1]) not in drop]
        return len(drop)


def _transform(points, el_rot, model_rot, shift):
    p = np.asarray(points, float)
    if el_rot is not None:
        origin, m = el_rot
        p = (p - origin) @ m.T + origin
    p = (p - CENTER) @ model_rot.T + CENTER
    return p + shift


_ROTATED = {}


def _rotate_dir(d, m):
    """The face direction d ends up facing after rotation matrix m (cached: few distinct rotations)."""
    key = (d, m.tobytes())
    out = _ROTATED.get(key)
    if out is None:
        v = np.array(DIRS[d], float) @ m.T
        out = _ROTATED[key] = max(DIRS, key=lambda k: np.dot(DIRS[k], v))
    return out


def _edge_pieces(corners, uv):
    """Copycat texture assembly (Copycats+/Create templates): split a face at its middle along
    any side shorter than a block; the low half shows the low edge of the texture and the high
    half the high edge, so the copied block's borders frame every copycat edge.
    corners: TL, TR, BR, BL; uv: (u0, v0, u1, v1) in 0..16. Yields (corners, uv rect)."""
    c = [[float(v) for v in p] for p in (corners.tolist() if hasattr(corners, "tolist") else corners)]
    u0, v0, u1, v1 = uv

    def segs(a, b):
        w = b - a
        if abs(w) >= 16 - 1e-6 or w == 0:
            return [((0.0, 1.0), (a, b))]
        h = w / 2
        lo, hi = (0.0, 16.0) if w > 0 else (16.0, 0.0)  # texture edges in the direction of travel
        return [((0.0, 0.5), (lo, lo + h)), ((0.5, 1.0), (hi - h, hi))]

    def at(fu, fv):  # bilinear point on the face
        out = []
        for a, b, cc, d in zip(c[0], c[1], c[2], c[3]):
            top = a + (b - a) * fu
            bottom = d + (cc - d) * fu
            out.append(top + (bottom - top) * fv)
        return out

    for (fu0, fu1), (tu0, tu1) in segs(u0, u1):
        for (fv0, fv1), (tv0, tv1) in segs(v0, v1):
            yield [at(fu0, fv0), at(fu1, fv0), at(fu1, fv1), at(fu0, fv1)], (tu0, tv0, tu1, tv1)


def _ct_uvs(uvs, col, row, n):
    """Move 0..16 face UVs into tile (col, row) of an n x n connected sheet."""
    return [((col * 16 + u) / n, (row * 16 + v) / n) for u, v in uvs]


class TexturedBuilder:
    def __init__(self, vox, structure):
        self.vox = vox
        self.assets = vox.assets
        self.structure = structure
        self.faces = Faces()
        self.full = {(x, y, z): bool(vox.block_shape(st).all()) for x, y, z, st in structure.blocks}
        # What each block connects as for connected textures: copycats connect as the block
        # they copy (Copycats+ passes CT through), everything else as itself.
        identity, copycats, states = {}, set(), {}
        for x, y, z, st in structure.blocks:
            pos = (x, y, z)
            states[pos] = st
            if "copycat" not in st.name:
                identity[pos] = st.name
                continue
            copycats.add(pos)
            data = structure.block_data.get(pos) or {}
            # material and CT switch: one Material/EnableCT, or one per part (multi-state copycats)
            parts = [(v.get("material"), v.get("enableCT", 1)) for v in (data.get("material_data") or {}).values()]
            parts = [(m, ct) for m, ct in parts if m and m.get("Name") not in (None, "create:copycat_base")]
            if not parts and data.get("Material"):
                parts = [(data["Material"], data.get("EnableCT", 1))]
            names = {m["Name"] for m, _ in parts}
            if "material_data" in data:
                # Multi-part copycats (half layers, bytes, boards...): Copycats+ renders each part
                # as its own scaled-down block, so every part gets a full bordered tile.
                identity[pos] = ("unique", pos)
            elif len(names) == 1 and all(ct for _, ct in parts):
                identity[pos] = names.pop()
            else:  # mixed materials or CT switched off: connects to nothing
                identity[pos] = ("unique", pos)

        def face_mask(pos, side):
            st = states.get(pos)
            if st is None:
                return b""
            shape = vox.block_shape(st)
            axis, last = {"west": (0, 0), "east": (0, -1), "down": (1, 0), "up": (1, -1),
                          "north": (2, 0), "south": (2, -1)}[side]
            return np.take(shape, last, axis=axis).astype(np.uint8).tobytes()  # outline touching that side

        self.ct = ConnectedTextures(self.assets, identity, self.full, copycats, face_mask)
        self.states = {(int(k[0]), int(k[1]), int(k[2])): v for k, v in states.items()}

    # --- model elements -------------------------------------------------------------
    def add_elements(self, elements, textures, model_rot, shift, pos, tint, fallback_textures=None,
                     material=None):
        for el in elements:
            tex_map = el.get("__textures", textures)
            # Faces are built from `from` and `to` as given, like Minecraft's FaceBakery: an
            # element with from > to (a Blockbench trick) gets its faces at the swapped ends,
            # turned inward, so sorting them would move those faces.
            lo = np.asarray(el["from"], float)
            hi = np.asarray(el["to"], float)
            el_rot = None
            r = el.get("rotation")
            if r and r.get("angle"):
                el_rot = (np.array(r.get("origin", [8, 8, 8]), float), _rot_matrix(r["axis"], r["angle"]))
            for face, f in (el.get("faces") or {}).items():
                if face not in DIRS:
                    continue
                cull = f.get("cullface")
                if cull in DIRS and pos is not None:
                    d = DIRS[_rotate_dir(cull, model_rot)]
                    if self.full.get((pos[0] + d[0], pos[1] + d[1], pos[2] + d[2])):
                        continue
                tex = resolve_texture(f.get("texture"), tex_map, fallback_textures)
                if material:  # copycat: the copied block's texture on the side this face points to
                    tex = material.get(_rotate_dir(face, model_rot)) or tex
                # copycats map the copied block 1:1 by position (Copycats+), not with the borrowed
                # shape's own UVs (a vanilla door's UVs mirror and squeeze its door artwork)
                uv = _default_uv(face, np.minimum(lo, hi), np.maximum(lo, hi)) if material else (
                    f.get("uv") or _default_uv(face, lo, hi))
                corners = _transform(_corners(face, lo, hi), el_rot, model_rot, shift)
                face_tint = tint if f.get("tintindex", -1) >= 0 else None
                uvc = _uv_corners(uv, 0 if material else f.get("rotation", 0))
                if pos is not None:
                    sheet = self.ct.sheet(tex)
                    if sheet:
                        col, row = self.ct.tile(pos, _rotate_dir(face, model_rot), sheet[1])
                        tex, uvc = sheet[0], _ct_uvs(uvc, col, row, sheet[1])
                # the two faces of a flat card sit back to back: they must not cull each other
                flat = abs(hi - lo)[{"west": 0, "east": 0, "down": 1, "up": 1, "north": 2, "south": 2}[face]] < 1e-6
                png = self.assets.texture_png(tex) if tex else None
                self.faces.add_quad(tex, face_tint, corners, uvc, cullable=not flat,
                                    overlay=bool(png and png[1]))

    def add_mesh(self, obj_id, textures, transform, tint=None):
        res = self.assets.obj_textured(obj_id)
        if res is None:
            return
        tris, uvs, mats = res
        pos = transform(tris)
        for mat in sorted(set(m for m in mats if m)) or [None]:
            sel = [i for i, m in enumerate(mats) if m == mat] if mat else list(range(len(mats)))
            tex = resolve_texture(mat, textures) if mat else None
            uv = uvs[sel].copy()
            uv[..., 1] = 1.0 - uv[..., 1]  # OBJ v goes up; glTF and NeoForge (flip_v) go down
            self.faces.add_tris(tex, tint, pos[sel], uv)

    # --- copycats -------------------------------------------------------------------
    def _material_textures(self, mat):
        """Per-direction textures of the block a copycat copies."""
        if not mat:
            return {}
        st = BlockState(mat.get("Name", "minecraft:air"),
                        tuple(sorted((k, str(v)) for k, v in mat.get("Properties", {}).items())))
        out = {}
        for model, xr, yr in (self.assets.resolve(st) or []):
            m = _blockstate_matrix(xr, yr)
            for el in model["elements"]:
                for face, f in (el.get("faces") or {}).items():
                    if face in DIRS:
                        out.setdefault(_rotate_dir(face, m), resolve_texture(f.get("texture"), model["textures"]))
            part = resolve_texture("#particle", model["textures"])
            for d in DIRS:
                out.setdefault(d, part)
        return out

    def add_boxes(self, boxes, data, shift, pos, block_name):
        data = data or {}
        for b in boxes:
            key = b[6] if len(b) > 6 else None
            if key and "material_data" in data:
                mat = (data["material_data"].get(key) or {}).get("material")
            else:
                mat = data.get("Material")
            if not mat or mat.get("Name") in (None, "minecraft:air"):
                mat = {"Name": "create:copycat_base"}
            texs = self._material_textures(mat)
            if block_name in codeshapes.RULE_TEXTURES:  # not a copycat (e.g. fluid pipe): its own texture
                texs = {d: codeshapes.RULE_TEXTURES[block_name] for d in DIRS}
            lo, hi = np.array(b[:3], float), np.array(b[3:6], float)
            for face in DIRS:
                corners = np.asarray(_corners(face, lo, hi), float) + shift
                # Copycats+ assembles every copycat face from the copied block's edges at 1:1
                # pixel scale; each part of a multi-part copycat is assembled on its own, so it
                # keeps its own border without any texture scaling.
                pieces = _edge_pieces(corners, _default_uv(face, lo, hi))
                sheet = self.ct.sheet(texs.get(face))
                tile = self.ct.tile(pos, face, sheet[1]) if sheet else None
                for pc, uv in pieces:
                    tex, uvc = texs.get(face), _uv_corners(uv, 0)
                    if sheet:
                        tex, uvc = sheet[0], _ct_uvs(uvc, tile[0], tile[1], sheet[1])
                    self.faces.add_quad(tex, None, pc, uvc)

    def add_mesh_rule(self, tris, data, shift):
        """Smooth copycat shapes (slopes): copied block's texture, UVs projected like block faces."""
        mat = (data or {}).get("Material")
        if not mat or mat.get("Name") in (None, "minecraft:air"):
            mat = {"Name": "create:copycat_base"}
        texs = self._material_textures(mat)
        for tri in tris:
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            face = max(DIRS, key=lambda d: np.dot(DIRS[d], n))
            x, y, z = tri[:, 0], tri[:, 1], tri[:, 2]
            uv = {"up": (x, z), "down": (x, 16 - z), "north": (16 - x, 16 - y), "south": (x, 16 - y),
                  "west": (z, 16 - y), "east": (16 - z, 16 - y)}[face]
            self.faces.add_tris(texs.get(face), None, tri + shift, np.column_stack(uv) / 16.0)

    # --- blocks ---------------------------------------------------------------------
    def add_block(self, x, y, z, st):
        pos = (x, y, z)
        self.faces.block = pos
        shift = np.array(pos, float) * 16
        assets, name = self.assets, st.name
        tint = FOLIAGE if "leaves" in name else GRASS if any(k in name for k in ("grass", "fern", "vine")) else None
        data = self.structure.block_data.get(pos)
        material = None
        if name in codeshapes.ALIASES:
            if "copycat" in name:  # copycat stairs/doors/trapdoors: borrowed shape, copied block's texture
                mat = (data or {}).get("Material")
                if not mat or mat.get("Name") in (None, "minecraft:air"):
                    mat = {"Name": "create:copycat_base"}
                material = self._material_textures(mat)
            st = BlockState(codeshapes.ALIASES[name], st.props)
            name = st.name
        if name in codeshapes.MESH_RULES:
            self.add_mesh_rule(codeshapes.MESH_RULES[name](st.properties), data, shift)
            return
        if name in codeshapes.RULES:
            self.add_boxes(codeshapes.RULES[name](st.properties), data, shift, pos, name)
            return
        if name in codeshapes.JAR_MODELS:
            ids, _ = codeshapes.JAR_MODELS[name]
            m = _blockstate_matrix(*codeshapes.FACING_ROTATION[st.properties.get("facing", "up")])
            for mid in ids:
                model = assets.model(mid)
                if model:
                    self.add_elements(model["elements"], model["textures"], m, shift, pos, tint)
            for mid, mo, sh in codeshapes.data_parts(st, data):
                part = assets.model(mid)
                if part:
                    self.add_elements(part["elements"], part["textures"], mo, shift + sh, None, tint)
            return
        resolved = assets.resolve(st) or []
        base_textures = {}
        for model, xr, yr in resolved:
            m = _blockstate_matrix(xr, yr)
            base_textures = base_textures or model["textures"]
            self.add_elements(model["elements"], model["textures"], m, shift, pos, tint, material=material)
            if model.get("mesh") is not None:
                self._add_model_obj(model, m, shift)
            for part in codeshapes.extra_by_model(model.get("id"), assets):
                self.add_elements(part["elements"], part["textures"], m, shift, pos, tint, model["textures"])
        if resolved:
            m = _blockstate_matrix(resolved[0][1], resolved[0][2])
            extra_ids = codeshapes.extra_models(name) + codeshapes.state_extra_models(st)
            fy = codeshapes.extras_frame_yaw(name, st.properties.get("facing"))
            me = m if fy is None else _blockstate_matrix(0, fy)
            for mid in extra_ids:
                extra = assets.model(mid)
                if extra:
                    self.add_elements(extra["elements"], extra["textures"], me, shift, pos, tint, base_textures)
            for mid in codeshapes.property_parts(st):
                part = assets.model(mid)
                if part:
                    self.add_elements(part["elements"], part["textures"], np.eye(3), shift, pos, tint)
            if name in codeshapes.ATTACHMENT_BLOCKS:
                def nb(side, p=pos):
                    d = codeshapes.PIPE_DIRS[side]
                    return self.states.get((p[0] + d[0], p[1] + d[1], p[2] + d[2]))
                for mid in codeshapes.pipe_attachments(st, nb):
                    part = assets.model(mid)
                    if part:
                        self.add_elements(part["elements"], part["textures"], np.eye(3), shift, pos, tint)
            for mid, rot in codeshapes.kinetic_shafts(st):
                part = assets.model(mid)
                if part:
                    self.add_elements(part["elements"], part["textures"], _blockstate_matrix(*rot), shift, None, tint)
            for mid, mo, sh in codeshapes.offset_parts(st, resolved[0][2], resolved[0][1]):
                part = assets.model(mid)
                if part:
                    self.add_elements(part["elements"], part["textures"], mo, shift + sh, None, tint, base_textures)
            for mid, mo, sh in codeshapes.data_parts(st, data):
                part = assets.model(mid)
                if part:
                    self.add_elements(part["elements"], part["textures"], mo, shift + sh, None, tint, base_textures)
            for mid, rot in codeshapes.north_frame_parts(st):
                part = assets.model(mid)
                if part:
                    mf = _blockstate_matrix(*rot)
                    self.add_elements(part["elements"], part["textures"], mf, shift, pos, tint, base_textures)
            if not extra_ids and name not in HANDLERS:
                self.add_elements(assets.item_extra_elements(name, codeshapes.item_extras_allowed(name)), {}, m, shift, pos, tint, base_textures)
        if name == "offroad:wheel_mount":
            # Wheel parts are posed outside the block and only roughly in the voxel grid, so the
            # ray test can't judge them; they're on the outside of a vehicle anyway.
            before = {k: len(v[0]) for k, v in self.faces.groups.items()}
            self._add_wheel(st, data or {}, shift)
            for k, v in self.faces.groups.items():
                self.faces.keep.update((k, i) for i in range(before.get(k, 0), len(v[0])))

    def _add_model_obj(self, model, m, shift):
        if model.get("obj"):
            self.add_mesh(model["obj"], model["textures"], lambda t: (t - CENTER) @ m.T + CENTER + shift)

    def _add_wheel(self, st, data, shift):
        """Same layout as attachments.wheel_layout, with textures."""
        from .attachments import wheel_layout
        assets = self.assets
        parts, tire = wheel_layout(st, data)
        for mid, M in parts:
            model = assets.model(mid)
            if model:
                R, t = M[:3, :3], M[:3, 3]
                self.add_elements(model["elements"], model["textures"], R, shift + R @ CENTER + t - CENTER, None, None)
        tire_id = data.get("CurrentStack", {}).get("id")
        if tire is None or not tire_id:
            return
        ns, _, tname = tire_id.partition(":")
        item_model = json.loads(assets.raw(f"assets/{ns}/models/item/{tname}/item.json") or b"{}")
        self.add_mesh(f"{ns}:models/item/{tname}/item.obj", item_model.get("textures", {}),
                      lambda t: t @ tire[:3, :3].T + tire[:3, 3] + shift)

    def _add_moving(self, part, merge):
        """A contraption at an angle off the grid (contraptions.MovingPart): build its blocks on
        their own grid, then move and turn the triangles about the anchor's centre."""
        sub = TexturedBuilder(self.vox, part.structure).build(merge=merge)
        off = np.array(part.offset, float) * 16  # faces are in 1/16-block units
        center = np.array(part.center, float) * 16
        for key, (pos, uv) in sub.groups.items():
            g = self.faces.groups.setdefault(key, [[], []])
            for p, u in zip(pos, uv):
                if len(p):
                    g[0].append((p + off - center) @ part.matrix.T + center)
                    g[1].append(u)

    def build(self, merge=False):
        """merge: also cull hidden faces (see Faces.cull_hidden); weld when writing."""
        from .progress import report
        n = max(1, len(self.structure.blocks))
        for k, (x, y, z, st) in enumerate(self.structure.blocks):
            report(k / n, f"Building model ({k}/{n} blocks)")
            self.add_block(x, y, z, st)
        self.faces.block = None
        self.faces.separate_back_to_back()
        for mid, m, t in codeshapes.structure_parts(self.structure):
            part = self.assets.model(mid)
            if part:  # add_elements turns about the block centre: express x -> m x + t that way
                c = np.array([8.0, 8, 8])
                self.add_elements(part["elements"], part["textures"], m, t - c + m @ c, None, None)
        for moving in getattr(self.structure, "moving", []):
            self._add_moving(moving, merge)
        if merge:
            from .visibility import see_through
            data = getattr(self.structure, "block_data", None) or {}
            opaque = {p: f for (p, f), (x, y, z, st) in zip(self.full.items(), self.structure.blocks)
                      if f and not see_through(st.name, data.get((x, y, z)))}
            self.culled = self.faces.cull_hidden(opaque)
        return self.faces


def write_glb(path, faces, assets, name="Minecraft-Modded2Model", weld=False, glass_single_sided=False):
    """Write grouped faces as a .glb: one mesh, one primitive per texture.

    weld: share vertices that have the same position, UV and normal (indexed triangles), so
    each surface is one connected piece instead of separate quads."""
    groups = [(k, np.concatenate(v[0]) / 16.0, np.concatenate(v[1])) for k, v in faces.groups.items()
              if v[0] and sum(len(a) for a in v[0])]
    allpos = np.concatenate([g[1] for g in groups])
    offset = np.array([-(allpos[:, 0].min() + allpos[:, 0].max()) / 2, -allpos[:, 1].min(),
                       -(allpos[:, 2].min() + allpos[:, 2].max()) / 2])  # centred, standing on y=0

    bin_parts, views, accessors, materials, images, textures_out, primitives = [], [], [], [], [], [], []
    tex_index = {}

    def add_view(data, target=None):
        pad = (-len(b"".join(bin_parts))) % 4
        if pad:
            bin_parts.append(b"\0" * pad)
        start = len(b"".join(bin_parts))
        bin_parts.append(data)
        v = {"buffer": 0, "byteOffset": start, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    for (tex, tint), pos, uv in groups:
        pos = (pos + offset).astype(np.float32)
        a = np.cross(pos[1::3] - pos[0::3], pos[2::3] - pos[0::3])
        n = np.repeat(a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9), 3, axis=0).astype(np.float32)
        uv = uv.astype(np.float32)
        indices = None
        if weld:
            key = np.round(np.hstack([pos, uv, n]).astype(np.float64), 5)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            pos, uv, n = (uniq[:, 0:3].astype(np.float32), uniq[:, 3:5].astype(np.float32),
                          uniq[:, 5:8].astype(np.float32))
            indices = inv.reshape(-1).astype(np.uint32)
        acc_pos = len(accessors)
        accessors.append({"bufferView": add_view(pos.tobytes(), 34962), "componentType": 5126, "count": len(pos),
                          "type": "VEC3", "min": pos.min(0).tolist(), "max": pos.max(0).tolist()})
        accessors.append({"bufferView": add_view(n.tobytes(), 34962), "componentType": 5126, "count": len(n), "type": "VEC3"})
        accessors.append({"bufferView": add_view(uv.tobytes(), 34962), "componentType": 5126, "count": len(uv), "type": "VEC2"})
        acc_idx = None
        if indices is not None:
            acc_idx = len(accessors)
            accessors.append({"bufferView": add_view(indices.tobytes(), 34963), "componentType": 5125,
                              "count": len(indices), "type": "SCALAR"})
        single = glass_single_sided and tex is not None and "glass" in tex
        mat = {"name": (tex or "missing").replace(":", "_"), "doubleSided": not single,
               "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0,
                                        "baseColorFactor": list(tint or (1.0, 1.0, 1.0)) + [1.0]}}
        png = assets.texture_png(tex) if tex else None
        if png:
            if tex not in tex_index:
                images.append({"bufferView": add_view(png[0]), "mimeType": "image/png", "name": tex})
                textures_out.append({"source": len(images) - 1, "sampler": 0})
                tex_index[tex] = (len(textures_out) - 1, png[1])
            ti, has_alpha = tex_index[tex]
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": ti}
            if has_alpha:
                mat["alphaMode"], mat["alphaCutoff"] = "MASK", 0.5
        else:
            mat["pbrMetallicRoughness"]["baseColorFactor"] = [1.0, 0.0, 1.0, 1.0]  # missing texture: magenta
        materials.append(mat)
        prim = {"attributes": {"POSITION": acc_pos, "NORMAL": acc_pos + 1, "TEXCOORD_0": acc_pos + 2},
                "material": len(materials) - 1}
        if acc_idx is not None:
            prim["indices"] = acc_idx
        primitives.append(prim)

    gltf = {"asset": {"version": "2.0", "generator": "Minecraft-Modded2Model"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
            "meshes": [{"name": name, "primitives": primitives}],
            "materials": materials, "accessors": accessors, "bufferViews": views,
            "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 10497, "wrapT": 10497}],
            "buffers": [{"byteLength": 0}]}
    if images:
        gltf["images"], gltf["textures"] = images, textures_out
    binary = b"".join(bin_parts)
    binary += b"\0" * ((-len(binary)) % 4)
    gltf["buffers"][0]["byteLength"] = len(binary)
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binary), 0x004E4942) + binary)
    return sum(len(g[1]) for g in groups) // 3, len(images)  # triangles before welding (same after)
