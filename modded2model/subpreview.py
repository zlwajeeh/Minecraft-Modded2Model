"""Picture of a schematic's Aeronautics sub-levels, to tell which is which before exporting one.

The overview shows the main blocks with every sub-level placed where it was saved (its
position and orientation), each tinted its own colour; below it, one labelled thumbnail per
sub-level in the same colour, with the number to pass to --sub-level.
"""
import math

import numpy as np

from . import nbt
from .blender import Faces, TexturedBuilder
from .schematic import load_structure

COLORS = [(1.0, 0.45, 0.35), (0.35, 0.75, 1.0), (0.5, 1.0, 0.45), (1.0, 0.85, 0.3), (0.85, 0.5, 1.0),
          (0.3, 1.0, 0.85), (1.0, 0.6, 0.85), (0.75, 0.75, 0.45)]


def _quat_matrix(q):
    w, x, y, z = (q.get(k, 0.0) for k in "wxyz")
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _faces(vox, structure):
    return TexturedBuilder(vox, structure).build()


def _add(dst, src, tint=None, rot=None, move=None):
    """Copy src's triangles into dst, optionally turned, moved (1/16 units) and tinted."""
    for (tex, t), (pos, uv) in src.groups.items():
        g = dst.groups.setdefault((tex, tint if tint is not None else t), [[], []])
        for p, u in zip(pos, uv):
            if not len(p):
                continue
            q = p if rot is None else p @ rot.T
            g[0].append(q + (0 if move is None else move))
            g[1].append(u)


BG = 34  # canvas grey around the views


def _crop(view, margin=0.04):
    """Trim the empty background around a rendered view (keeps a small margin)."""
    a = np.asarray(view)
    used = np.abs(a.astype(int) - a[0, 0].astype(int)).sum(2) > 6
    if not used.any():
        return a
    ys, xs = np.nonzero(used)
    m = int(margin * max(a.shape[:2]))
    return a[max(0, ys.min() - m):ys.max() + 1 + m, max(0, xs.min() - m):xs.max() + 1 + m]


def _square(views, size):
    """Views trimmed and stacked top to bottom, scaled to fit a size x size square."""
    from PIL import Image
    parts = [Image.fromarray(np.ascontiguousarray(_crop(v)).astype(np.uint8)) for v in views]
    gap = size // 40
    width = max(p.width for p in parts)
    height = sum(p.height for p in parts) + gap * (len(parts) - 1)
    scale = min(size / width, size / height)
    canvas = Image.new("RGB", (size, size), (BG, BG, BG))
    y = (size - int(height * scale)) // 2
    for p in parts:
        w, h = max(1, int(p.width * scale)), max(1, int(p.height * scale))
        canvas.paste(p.resize((w, h), Image.LANCZOS), ((size - w) // 2, y))
        y += h + int(gap * scale)
    return canvas


def render(path, vox, out_png, size=420):
    """Write the sub-level picture for schematic `path`. Returns the number of sub-levels."""
    img, n = picture(path, vox, size)
    if img is not None:
        img.save(out_png)
    return n


def structure_picture(structure, vox, size=420, title=None):
    """Two views of one structure, trimmed and stacked in a size x size PIL image (for previews of
    ships, regions and parts)."""
    from PIL import Image, ImageDraw
    from .preview_render import render_textured
    from .progress import stage
    with stage(0, 0.6, "Building model"):
        f = _faces(vox, structure)
    if not any(len(p) for g in f.groups.values() for p in g[0]):
        img = Image.new("RGB", (size, size), (BG, BG, BG))
    else:
        views = []
        for k, y in enumerate((35, 215)):
            with stage(0.6 + 0.2 * k, 0.8 + 0.2 * k, "Rendering preview"):
                views.append(render_textured(f, vox.assets, math.radians(y), math.radians(25), size, backface=True))
        img = _square(views, size)
    if title:
        ImageDraw.Draw(img).text((8, 6), title, fill=(230, 230, 230))
    return img


def picture(path, vox, size=420):
    """(PIL image, number of sub-levels) for schematic `path`; image None if it has none."""
    from PIL import Image, ImageDraw
    from .preview_render import render_textured

    root = nbt.load(path)
    raw = root.get("sub_levels", [])
    main = load_structure(path, merge_sub_levels=False)
    subs = main.sub_levels
    if not subs:
        return None, 0
    from .progress import stage
    overview = Faces()
    parts = len(subs) + 1
    if main.blocks:
        with stage(0, 0.5 / parts, "Building main build"):
            _add(overview, _faces(vox, main))
    thumbs = []
    for i, (sl, r) in enumerate(zip(subs, raw)):
        color = COLORS[i % len(COLORS)]
        with stage(0.5 * (i + 1) / parts, 0.5 * (i + 2) / parts, f"Building sub-level {i}"):
            f = _faces(vox, sl)
        p = r.get("position", {})
        rot = _quat_matrix(r.get("orientation", {}))
        move = np.array([p.get(k, 0.0) for k in "xyz"]) * 16
        _add(overview, f, tint=color, rot=rot, move=move)
        thumbs.append((i, sl, f, color, abs(abs(r.get("orientation", {}).get("w", 1.0)) - 1) > 1e-3))

    def shot(faces, s, yaw=35, pitch=25):
        if not any(len(p) for g in faces.groups.values() for p in g[0]):
            return np.full((s, s, 3), 34, np.uint8)
        return render_textured(faces, vox.assets, math.radians(yaw), math.radians(pitch), s,
                               backface=True).astype(np.uint8)

    per_row = 4 if len(thumbs) > 3 else max(1, len(thumbs))
    per_row = max(per_row, 3)
    rows_n = (len(thumbs) + per_row - 1) // per_row
    tw = size // per_row
    top_side = max(tw, size - rows_n * tw)  # the overview fills what the thumbnail rows leave
    with stage(0.5, 0.625, "Rendering overview"):
        left = shot(overview, size, 35, 25)
    with stage(0.625, 0.75, "Rendering overview"):
        right = shot(overview, size, 215, 25)
    top = np.asarray(_square([left, right], top_side))
    cells = []
    for i, sl, f, color, rotated in thumbs:
        with stage(0.75 + 0.25 * i / len(thumbs), 0.75 + 0.25 * (i + 1) / len(thumbs), f"Rendering sub-level {i}"):
            img = _square([shot(f, size)], tw)
        d = ImageDraw.Draw(img)
        c = tuple(int(255 * v) for v in color)
        d.rectangle([0, 0, tw - 1, tw - 1], outline=c, width=4)
        sx, sy, sz = sl.size
        d.text((8, 6), f"sub-level {i}", fill=c)
        d.text((8, 20), f"{sx}x{sy}x{sz}, {len(sl.blocks)} blocks", fill=(230, 230, 230))
        d.text((8, 34), "rotated (export with --sub-level)" if rotated else "merged into the main export",
               fill=(200, 200, 200))
        cells.append(np.asarray(img))
    canvas = Image.new("RGB", (size, size), (BG, BG, BG))
    canvas.paste(Image.fromarray(top), ((size - top_side) // 2, 0))
    for k, cell in enumerate(cells):
        r, c = divmod(k, per_row)
        canvas.paste(Image.fromarray(cell), (c * tw, top_side + r * tw))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), "main blocks (untinted) with each sub-level where it was saved", fill=(230, 230, 230))
    return canvas, len(subs)
