"""Extract a watertight surface from a voxel grid and write it as STL or 3MF."""
import struct
import zipfile

import numpy as np

# For each face direction: axis, sign, and the 4 corner offsets in counter-clockwise
# order when viewed from outside (so normals point outward).
_FACES = [
    (0, +1, [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    (0, -1, [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
    (1, +1, [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    (1, -1, [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    (2, +1, [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    (2, -1, [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
]


def surface(grid):
    """Return (vertices int (V,3) in cell units, triangles (T,3)) for the exposed faces."""
    g = np.pad(grid, 1)
    quads = []
    for axis, sign, corners in _FACES:
        neighbor = np.roll(g, -sign, axis=axis)
        exposed = g & ~neighbor
        cells = np.argwhere(exposed) - 1  # undo padding
        if len(cells):
            quads.append(cells[:, None, :] + np.array(corners)[None])
    if not quads:
        return np.zeros((0, 3), int), np.zeros((0, 3), int)
    quads = np.concatenate(quads)  # (Q, 4, 3)
    verts, idx = np.unique(quads.reshape(-1, 3), axis=0, return_inverse=True)
    idx = idx.reshape(-1, 4)
    tris = np.concatenate([idx[:, [0, 1, 2]], idx[:, [0, 2, 3]]])
    return verts, tris


def add_meshes(verts, tris, meshes):
    """Append closed triangle meshes (each (N, 3, 3), in cell units) as extra shells.

    They may overlap the voxel body; slicers merge overlapping shells of one object.
    """
    if not meshes:
        return verts, tris
    verts = [verts.astype(float)]
    tris = [tris]
    count = len(verts[0])
    for m in meshes:
        verts.append(m.reshape(-1, 3))
        tris.append(np.arange(count, count + len(m) * 3).reshape(-1, 3))
        count += len(m) * 3
    return np.concatenate(verts), np.concatenate(tris)


def to_print_space(verts, cell_mm, z_up=True):
    """Minecraft is Y-up; printers are Z-up. Rotate (keeping handedness) unless z_up is off,
    and scale (cell_mm: file units per grid cell)."""
    v = verts.astype(float) * cell_mm
    out = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1) if z_up else v.copy()
    out -= out.min(0)
    return out


def write_stl(path, verts, tris):
    tri_v = verts[tris].astype(np.float32)
    normals = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = (normals / np.where(lens == 0, 1, lens)).astype(np.float32)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = normals
    rec["v"] = tri_v
    with open(path, "wb") as f:
        f.write(b"Minecraft-Modded2Model binary STL".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        f.write(rec.tobytes())


def write_3mf(path, verts, tris):
    vs = "".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in verts)
    ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in tris)
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        f'<resources><object id="1" type="model"><mesh><vertices>{vs}</vertices>'
        f'<triangles>{ts}</triangles></mesh></object></resources>'
        '<build><item objectid="1"/></build></model>'
    )
    types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)
