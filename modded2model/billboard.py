"""Turn billboard parts (cut-out textures on flat cards) into solid pixel boxes.

Minecraft draws torches, ladders, bars, plants and the antennas on redstone links as cards: a
full 16x16 face where only the opaque pixels show. Printed, such a card becomes a solid slab;
in Blender it's a paper-thin card. Here the opaque pixels of the card's texture become
boxes spanning the card's thickness (one pixel for a flat card), textured with those pixels,
the way Minecraft extrudes held items; neighbouring pixels share one box (a rectangle of them).
Works for any element whose faces all lie on one axis (or that has no thickness) and whose
texture has see-through pixels in the face's area.
"""
import io

import numpy as np

AXIS_FACES = {0: ("west", "east"), 1: ("down", "up"), 2: ("north", "south")}
FACE_AXIS = {f: a for a, fs in AXIS_FACES.items() for f in fs}
# (u axis, high end at u1?), (v axis, high end at v1?) for each face; see Minecraft's default UVs
FACE_UV = {"north": ((0, True), (1, True)), "south": ((0, False), (1, True)),
           "west": ((2, False), (1, True)), "east": ((2, True), (1, True)),
           "up": ((0, False), (2, False)), "down": ((0, False), (2, True))}
MAX_PIXELS = 1024
MAX_CARD_THICKNESS = 4  # px: thicker two-faced boxes (scaffold frames) aren't cards  # bigger cut-out areas stay as they are (glass panes, big grates)


def _default_uv(face, f, t):
    x1, y1, z1 = f
    x2, y2, z2 = t
    return {"north": [16 - x2, 16 - y2, 16 - x1, 16 - y1], "south": [x1, 16 - y2, x2, 16 - y1],
            "west": [z1, 16 - y2, z2, 16 - y1], "east": [16 - z2, 16 - y2, 16 - z1, 16 - y1],
            "up": [x1, z1, x2, z2], "down": [x1, 16 - z2, x2, 16 - z1]}[face]


def _resolve(ref, textures):
    for _ in range(8):
        if not isinstance(ref, str) or not ref.startswith("#"):
            break
        ref = textures.get(ref[1:])
    return ref if isinstance(ref, str) and not ref.startswith("#") else None


def _alpha(assets, tex_id, cache):
    if tex_id not in cache:
        png = assets.texture_png(tex_id)
        if png is None or not png[1]:
            cache[tex_id] = None
        else:
            from PIL import Image
            cache[tex_id] = np.asarray(Image.open(io.BytesIO(png[0])).getchannel("A")) >= 128
    return cache[tex_id]


def solidify_elements(elements, textures, assets, cache=None):
    """Elements with billboard cards replaced by per-pixel boxes (others unchanged)."""
    cache = {} if cache is None else cache
    out, seen = [], set()
    for el in elements:
        boxes = _pixel_boxes(el, textures, assets, cache)
        if boxes is None:
            out.append(el)
            continue
        for b in boxes:  # crossed cards (torches, plants) give the same pixel twice
            key = (tuple(np.round(b["from"], 4)), tuple(np.round(b["to"], 4)), str(b.get("rotation")))
            if key not in seen:
                seen.add(key)
                out.append(b)
    return out


def _pixel_boxes(el, textures, assets, cache):
    faces = el.get("faces") or {}
    f, t = el.get("from"), el.get("to")
    if not faces or not f or not t:
        return None
    if any(a > b for a, b in zip(f, t)):
        return None  # an inside-out box (from > to) is a deliberate trick: keep it as it is
    lo, hi = np.minimum(f, t).astype(float), np.maximum(f, t).astype(float)
    flat = [a for a in range(3) if hi[a] - lo[a] < 1e-6]
    if len(flat) == 1:  # faces across a flat card have no area: only its two sides count
        faces = {n: v for n, v in faces.items() if FACE_AXIS.get(n) == flat[0]}
    axes = {FACE_AXIS[n] for n in faces if n in FACE_AXIS}
    if len(flat) == 1:
        axis = flat[0]
        if axes - {axis}:
            return None
    elif len(axes) == 1 and not flat:
        axis = axes.pop()
        if hi[axis] - lo[axis] > MAX_CARD_THICKNESS:
            return None  # a box with only two faces, not a card
    else:
        return None
    name = next(n for n in AXIS_FACES[axis] if n in faces) if any(n in faces for n in AXIS_FACES[axis]) else None
    if name is None:
        return None
    face = faces[name]
    if face.get("rotation"):
        return None
    tex = _resolve(face.get("texture"), textures)
    mask = _alpha(assets, tex, cache) if tex else None
    if mask is None:
        return None
    h, w = mask.shape
    u1, v1, u2, v2 = face.get("uv") or _default_uv(name, lo, hi)
    (ua, uflip), (va, vflip) = FACE_UV[name]
    # texels covered by the face's UV rectangle
    i0, i1 = int(np.floor(min(u1, u2) * w / 16 + 1e-6)), int(np.ceil(max(u1, u2) * w / 16 - 1e-6))
    j0, j1 = int(np.floor(min(v1, v2) * h / 16 + 1e-6)), int(np.ceil(max(v1, v2) * h / 16 - 1e-6))
    i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1, w), min(j1, h)
    region = mask[j0:j1, i0:i1]
    if region.size == 0 or region.all() or region.size > MAX_PIXELS:
        return None  # nothing see-through (or too big): keep the card as it is
    if abs(u2 - u1) < 1e-9 or abs(v2 - v1) < 1e-9:
        return None

    def coord(val, a, flip, start, end):
        s = (val - start) / (end - start)  # 0 at the u1/v1 side
        return hi[a] - s * (hi[a] - lo[a]) if flip else lo[a] + s * (hi[a] - lo[a])

    def tex(pos, a, flip, start, end):  # the other way: world position -> texture coordinate
        s = (hi[a] - pos) / (hi[a] - lo[a]) if flip else (pos - lo[a]) / (hi[a] - lo[a])
        return start + s * (end - start)

    if axis in flat:
        mid = lo[axis]
        depth = (mid - 0.5, mid + 0.5)
    else:
        depth = (lo[axis], hi[axis])
    from .blender import _corners
    keep = {k: v for k, v in el.items() if k not in ("from", "to", "faces")}
    tint = {"tintindex": face["tintindex"]} if "tintindex" in face else {}
    sub = region.astype(bool)
    boxes = []
    # one box per rectangle of opaque pixels (not per pixel): far fewer triangles, same shape
    for ra, rb, ca, cb in _rectangles(sub):
        ua0, ua1 = max((i0 + ra) * 16 / w, min(u1, u2)), min((i0 + rb) * 16 / w, max(u1, u2))
        va0, va1 = max((j0 + ca) * 16 / h, min(v1, v2)), min((j0 + cb) * 16 / h, max(v1, v2))
        p = sorted((coord(ua0, ua, uflip, u1, u2), coord(ua1, ua, uflip, u1, u2)))
        q = sorted((coord(va0, va, vflip, v1, v2), coord(va1, va, vflip, v1, v2)))
        b_lo, b_hi = [0.0] * 3, [0.0] * 3
        b_lo[axis], b_hi[axis] = depth
        b_lo[ua], b_hi[ua] = p
        b_lo[va], b_hi[va] = q
        if min(b - a for a, b in zip(b_lo, b_hi)) <= 1e-6:
            continue
        # leave out sides pressed all along against more opaque pixels (a side touching them
        # only in part stays: the hidden part sits inside the solid)
        covered = set()
        for a, ends, flip, start, end, n, line in (
                (ua, ((ua0, ra - 1), (ua1, rb)), uflip, u1, u2, sub.shape[1], lambda k: sub[ca:cb, k]),
                (va, ((va0, ca - 1), (va1, cb)), vflip, v1, v2, sub.shape[0], lambda k: sub[k, ra:rb])):
            for t_end, k in ends:
                if 0 <= k < n and line(k).all():
                    w_end = coord(t_end, a, flip, start, end)
                    covered.add(AXIS_FACES[a][1 if abs(w_end - b_hi[a]) < 1e-6 else 0])
        box_faces = {}
        for name_ in ("north", "south", "west", "east", "up", "down"):
            if name_ in covered:
                continue
            # each corner shows the texel at its spot on the card, kept 0.02 inside the
            # rectangle; a side's corners all fall on the edge texels, so it shows those
            pts = [(min(max(tex(c[ua], ua, uflip, u1, u2), ua0 + 0.02), ua1 - 0.02),
                    min(max(tex(c[va], va, vflip, v1, v2), va0 + 0.02), va1 - 0.02))
                   for c in _corners(name_, b_lo, b_hi)]
            uv, rot = _face_uv(pts)
            box_faces[name_] = dict(texture=face["texture"], uv=uv, **({"rotation": rot} if rot else {}), **tint)
        boxes.append(dict(keep, **{"from": b_lo, "to": b_hi, "faces": box_faces, "__pixel": True}))
    return boxes


def _rectangles(mask):
    """Cover the True cells of a 2D mask with few rectangles (i0, i1, j0, j1): runs along rows
    grown down, or runs along columns grown sideways (blades of grass), whichever needs fewer."""
    by_row = _runs(mask)
    by_col = [(i0, i1, j0, j1) for j0, j1, i0, i1 in _runs(mask.T)]
    return by_row if len(by_row) <= len(by_col) else by_col


def _runs(mask):
    """Rectangles (i0, i1, j0, j1): the run along each row, grown down while the rows below are
    opaque all along it."""
    left = mask.copy()
    h, w = left.shape
    out = []
    for j in range(h):
        i = 0
        while i < w:
            if not left[j, i]:
                i += 1
                continue
            e = i
            while e < w and left[j, e]:
                e += 1
            k = j + 1
            while k < h and left[k, i:e].all():
                k += 1
            left[j:k, i:e] = False
            out.append((i, e, j, k))
            i = e
    return out


def _face_uv(pts):
    """uv [u1, v1, u2, v2] and rotation that give a face's corners (top-left, top-right,
    bottom-right, bottom-left) these texture points, the way blender._uv_corners reads them."""
    for k in range(4):
        (a, b), (c, d) = pts[-k % 4], pts[(2 - k) % 4]
        quad = [(a, b), (c, b), (c, d), (a, d)]
        if all(abs(quad[(m + k) % 4][0] - pts[m][0]) < 1e-6 and abs(quad[(m + k) % 4][1] - pts[m][1]) < 1e-6
               for m in range(4)):
            return [a, b, c, d], 90 * k
    raise ValueError("face corners don't form a texture rectangle")
