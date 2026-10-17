"""Which faces show on the outside of the model, decided by the block-level interior fill.

The fill treats the build block by block: rooms that outside air can't reach (with doors and
windows up to a few blocks wide sealed while it checks) are filled in, and only faces between
the filled model and outside air are kept. Here the same block-level fill decides each face:
it stays if the space just in front of it is outside air, or a block that touches outside air
(so partial blocks on the surface keep their detail); faces looking into a filled room or a
buried block go.
"""
import numpy as np
from scipy import ndimage

SEE_THROUGH = ("glass", "pane", "window", "grat", "bars", "door", "trapdoor", "leaves", "ice",
               "fence", "chain", "ladder", "rail", "scaffold")


def see_through(name, data=None):
    """True for glass-like blocks, and copycats with any glass-like material in them."""
    names = [name]
    if data:
        mats = [data.get("Material")] + [v.get("material") for v in (data.get("material_data") or {}).values()]
        names += [m.get("Name", "") for m in mats if isinstance(m, dict)]
    return any(k in n.split(":")[-1] for n in names for k in SEE_THROUGH)


FILL_TEXTURE = "minecraft:block/gray_concrete"  # what filled rooms show through windows/doorways
_SIDES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


class Interior:
    """Interior fill for a structure ("Fill air bubbles" + "Seal off entrances").

    Every block with geometry is a wall (glass, doors and other entrances included). Doors and
    windows up to seal_width blocks wide are plugged while outside air floods in block by block;
    every air block it can't reach is filled, then the plugs come out again so doorways stay as
    recesses.
    """

    def __init__(self, vox, structure, seal_width=1):
        from .printprep import block_fill
        data = getattr(structure, "block_data", None) or {}
        self.size = structure.size
        self.solid = np.zeros(structure.size, bool)
        self.full = np.zeros(structure.size, bool)
        self.clear = np.zeros(structure.size, bool)  # see-through: what's behind shows
        opaque = {}
        for x, y, z, st in structure.blocks:
            shape = vox.block_shape(st)
            self.solid[x, y, z] = bool(shape.any())
            if st not in opaque:
                opaque[st] = _opaque(vox.assets, st)
            # a wall you can't see through: fills its cell, with no see-through texture
            self.clear[x, y, z] = see_through(st.name, data.get((x, y, z)))
            self.full[x, y, z] = bool(shape.all()) and opaque[st] and not self.clear[x, y, z]
        filled, outside = block_fill(self.solid, max(1, int(round(seal_width))))
        self.filled_air = filled & ~self.solid
        # what can be seen from outside: outside air, plus see-through blocks (glass, grates,
        # doors) and partial blocks (seen through their gaps) joined to it; full blocks and the
        # filled rooms stop the view
        sees = np.pad(outside | self.clear | (self.solid & ~self.full), 1, constant_values=True)
        lab, _ = ndimage.label(sees, structure=ndimage.generate_binary_structure(3, 1))
        exposed = np.isin(lab, np.unique(lab[np.pad(outside, 1, constant_values=True)]))
        exposed &= sees
        near = ndimage.binary_dilation(exposed, structure=ndimage.generate_binary_structure(3, 1))
        # where a face may look and still be seen: exposed space, or a wall block touching it
        self.expo = exposed | (near & np.pad(self.solid, 1))
        self.exposed = exposed

    def fill_faces(self):
        """Quads (corners in 1/16 units, counter-clockwise from outside) closing the filled rooms
        where they could be seen: next to outside air, see-through blocks, or blocks that don't
        fill their whole cell."""
        quads = []
        # every side not against an opaque full block (or more filled room) may be seen into
        show = ~np.pad(self.full, 1) & ~np.pad(self.filled_air, 1)
        for x, y, z in np.argwhere(self.filled_air):
            for d in _SIDES:
                if show[x + d[0] + 1, y + d[1] + 1, z + d[2] + 1]:
                    quads.append(_cell_face((x, y, z), d))
        return quads


def _opaque(assets, state):
    """True only for a block you can't see into: its model has a whole-cell cube with all six
    faces, and none of its textures has see-through pixels (glass-like frames, grilles)."""
    from .blender import resolve_texture
    cube = False
    for model, _, _ in assets.resolve(state) or []:
        for el in model.get("raw_elements", model["elements"]):
            faces = el.get("faces") or {}
            if (list(el.get("from", [])) == [0, 0, 0] and list(el.get("to", [])) == [16, 16, 16]
                    and not (el.get("rotation") or {}).get("angle") and len(faces) == 6):
                cube = True
            for f in faces.values():
                tex = resolve_texture(f.get("texture"), model["textures"])
                png = assets.texture_png(tex) if tex else None
                if png and png[1]:
                    return False
    return cube


def _cell_face(cell, d):
    """The face of block `cell` toward direction d, as 4 corners seen from outside (from d)."""
    a = int(np.argmax(np.abs(d)))
    u, v = [i for i in range(3) if i != a]
    base = np.array(cell, float) * 16
    plane = base[a] + (16 if d[a] > 0 else 0)
    pts = []
    for cu, cv in ((0, 16), (16, 16), (16, 0), (0, 0)):
        p = np.zeros(3)
        p[a], p[u], p[v] = plane, base[u] + cu, base[v] + cv
        pts.append(p)
    pts = np.array(pts)
    n = np.cross(pts[3] - pts[0], pts[1] - pts[0])
    if np.sign(n[a]) != np.sign(d[a]):
        pts = pts[::-1]
    return pts


def exposure(vox, structure, seal_width=1):
    """Where a face may look and still be seen (see Interior), as passed to visible()."""
    return Interior(vox, structure, seal_width).expo


def visible(tris, expo):
    """Boolean mask over triangles (N, 3, 3 in 1/16-block units): True if the face shows."""
    tris = np.asarray(tris, float)
    if not len(tris):
        return np.zeros(0, bool)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln == 0, 1, ln)
    cen = tris.mean(1)
    seen = np.zeros(len(tris), bool)
    shape = np.array(expo.shape)
    # the centre and a point near each corner, a hair in front of the face
    for q in [cen] + [cen + (tris[:, k] - cen) * 0.9 for k in range(3)]:
        cell = np.floor((q + n * 0.05) / 16).astype(int) + 1
        out = np.any((cell < 0) | (cell >= shape), axis=1)
        ok = ~out
        hit = np.zeros(len(tris), bool)
        hit[ok] = expo[tuple(cell[ok].T)]
        seen |= out | hit
    return seen
