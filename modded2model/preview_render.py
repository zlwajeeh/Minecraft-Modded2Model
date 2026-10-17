"""Software renderer for previews: textured faces (modded2model.blender.Faces) to an RGB array.

Vectorized: triangles are grouped by their on-screen size, every candidate pixel of a whole
batch is tested at once with numpy, and one z-buffer pass at the end keeps the nearest
fragment per pixel. Much faster than drawing triangle by triangle for big builds.
"""
import io
import math

import numpy as np

_CHUNK = 3_000_000  # candidate pixels processed at once (memory bound)


def _texture(assets, tex, cache):
    if tex not in cache:
        from PIL import Image
        png = assets.texture_png(tex) if tex else None
        cache[tex] = (np.asarray(Image.open(io.BytesIO(png[0])).convert("RGBA"), np.float32) / 255
                      if png else np.array([[[1, 0, 1, 1]]], np.float32))
    return cache[tex]


def render_textured(faces, assets, yaw, pitch, size=700, frame=None, backface=False):
    """Render textured faces with nearest-neighbour sampling; returns (size, size, 3) uint8."""
    from .progress import report
    groups = [(k, np.concatenate(v[0]), np.concatenate(v[1])) for k, v in faces.groups.items()
              if v[0] and sum(len(a) for a in v[0])]
    img = np.zeros((size * size, 3), np.float32) + np.array([0.13, 0.13, 0.15], np.float32)
    if not groups:
        return (img.reshape(size, size, 3) * 255).astype(np.uint8)
    allp = np.concatenate([g[1] for g in groups])
    c = allp.mean(0) if frame is None else frame[0]
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rot = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    span = np.abs((allp - c) @ rot.T)[:, :2].max() if frame is None else frame[1]
    scale = size * 0.45 / span
    light = np.array([0.4, 0.8, 0.6]) / np.linalg.norm([0.4, 0.8, 0.6])
    tex_cache = {}
    frag_idx, frag_z, frag_col = [], [], []

    for gi, ((tex, tint), pos, uv) in enumerate(groups):
        report(gi / len(groups), "Rendering preview")
        t = _texture(assets, tex, tex_cache)
        th, tw = t.shape[:2]
        tintv = np.array(tint or (1, 1, 1), np.float32)
        p = ((pos - c) @ rot.T).reshape(-1, 3, 3)
        tuv = uv.reshape(-1, 3, 2)
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 0
        if backface:
            ok &= n[:, 2] > 0  # facing away from the camera: culled like in game
        p, tuv, n, ln = p[ok], tuv[ok], n[ok], ln[ok]
        if not len(p):
            continue
        shade = (0.45 + 0.55 * np.abs((n / ln[:, None]) @ light)).astype(np.float32)
        sx = p[:, :, 0] * scale + size / 2
        sy_ = -p[:, :, 1] * scale + size / 2
        x0 = np.floor(sx.min(1)).astype(int)
        y0 = np.floor(sy_.min(1)).astype(int)
        ext = np.maximum(np.ceil(sx.max(1)).astype(int) - x0, np.ceil(sy_.max(1)).astype(int) - y0) + 1
        den = (sy_[:, 1] - sy_[:, 2]) * (sx[:, 0] - sx[:, 2]) + (sx[:, 2] - sx[:, 1]) * (sy_[:, 0] - sy_[:, 2])
        keep = (np.abs(den) > 1e-9) & (sx.max(1) >= 0) & (sx.min(1) < size) & (sy_.max(1) >= 0) & (sy_.min(1) < size)
        # batch by on-screen size (powers of two) so each batch tests a K x K pixel window
        bucket = np.ceil(np.log2(np.maximum(ext, 1))).astype(int)
        for b in np.unique(bucket[keep]):
            K = 1 << int(b)
            ids_all = np.flatnonzero(keep & (bucket == b))
            per = max(1, _CHUNK // (K * K))
            for s0 in range(0, len(ids_all), per):
                ids = ids_all[s0:s0 + per]
                oy, ox = np.divmod(np.arange(K * K), K)
                X = x0[ids, None] + ox[None, :]
                Y = y0[ids, None] + oy[None, :]
                xs, ys = X + 0.5, Y + 0.5
                a, bb = sx[ids], sy_[ids]
                d = den[ids, None]
                w0 = ((bb[:, 1, None] - bb[:, 2, None]) * (xs - a[:, 2, None])
                      + (a[:, 2, None] - a[:, 1, None]) * (ys - bb[:, 2, None])) / d
                w1 = ((bb[:, 2, None] - bb[:, 0, None]) * (xs - a[:, 2, None])
                      + (a[:, 0, None] - a[:, 2, None]) * (ys - bb[:, 2, None])) / d
                w2 = 1 - w0 - w1
                inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6) & (X >= 0) & (X < size) & (Y >= 0) & (Y < size)
                if not inside.any():
                    continue
                r, k = np.nonzero(inside)
                W0, W1, W2 = w0[r, k], w1[r, k], w2[r, k]
                tri = ids[r]
                z = W0 * p[tri, 0, 2] + W1 * p[tri, 1, 2] + W2 * p[tri, 2, 2]
                u = W0 * tuv[tri, 0, 0] + W1 * tuv[tri, 1, 0] + W2 * tuv[tri, 2, 0]
                v = W0 * tuv[tri, 0, 1] + W1 * tuv[tri, 1, 1] + W2 * tuv[tri, 2, 1]
                tx = np.clip((u % 1.0001 * tw).astype(int), 0, tw - 1)
                ty = np.clip((v % 1.0001 * th).astype(int), 0, th - 1)
                col = t[ty, tx]
                solid = col[:, 3] > 0.5
                if not solid.any():
                    continue
                frag_idx.append((Y[r, k] * size + X[r, k])[solid])
                frag_z.append(z[solid])
                frag_col.append(col[solid, :3] * tintv * shade[tri][solid, None])

    if frag_idx:
        idx = np.concatenate(frag_idx)
        z = np.concatenate(frag_z)
        col = np.concatenate(frag_col)
        zbuf = np.full(size * size, -np.inf)
        np.maximum.at(zbuf, idx, z)
        win = z >= zbuf[idx]
        img[idx[win]] = col[win]
    return (np.clip(img.reshape(size, size, 3), 0, 1) * 255).astype(np.uint8)
