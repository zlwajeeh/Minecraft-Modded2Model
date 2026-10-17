"""Render a block's exported geometry to a PNG (flat shading, z-buffer, two views).

python tools_render.py <instance> <block id> [key=value ...] -o out.png
"""
import argparse
import math
import struct
import zlib

import numpy as np

from modded2model import instance
from modded2model.assets import Assets
from modded2model.exact import ExactModels
from modded2model.schematic import BlockState
from modded2model.voxelize import Voxelizer


def write_png(path, img):
    h, w, _ = img.shape
    raw = b"".join(b"\0" + img[y].tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def render(tris, yaw, pitch, size=520):
    c = tris.reshape(-1, 3).mean(0)
    t = tris - c
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    t = t @ (rx @ ry).T
    scale = size * 0.8 / (np.abs(t[..., :2]).max() * 2)
    img = np.zeros((size, size, 3)) + np.array([0.13, 0.13, 0.15])
    zbuf = np.full((size, size), -np.inf)
    light = np.array([0.4, 0.8, 0.6]) / np.linalg.norm([0.4, 0.8, 0.6])
    for tri in t:
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        ln = np.linalg.norm(n)
        if ln == 0 or n[2] <= 0:  # back-facing (camera looks down -z toward +z viewer)
            continue
        n /= ln
        shade = 0.35 + 0.65 * max(0.0, n @ light)
        p = np.column_stack([tri[:, 0] * scale + size / 2, -tri[:, 1] * scale + size / 2, tri[:, 2]])
        x0, x1 = int(max(0, p[:, 0].min())), int(min(size - 1, p[:, 0].max() + 1))
        y0, y1 = int(max(0, p[:, 1].min())), int(min(size - 1, p[:, 1].max() + 1))
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay, az), (bx, by, bz), (cx, cy_, cz) = p
        den = (by - cy_) * (ax - cx) + (cx - bx) * (ay - cy_)
        if abs(den) < 1e-9:
            continue
        w0 = ((by - cy_) * (xs - cx) + (cx - bx) * (ys - cy_)) / den
        w1 = ((cy_ - ay) * (xs - cx) + (ax - cx) * (ys - cy_)) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        z = w0 * az + w1 * bz + w2 * cz
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        draw = inside & (z > sub)
        sub[draw] = z[draw]
        img[y0:y1 + 1, x0:x1 + 1][draw] = np.array([0.72, 0.74, 0.78]) * shade
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("instance")
    p.add_argument("block")
    p.add_argument("props", nargs="*")
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()
    state = BlockState(a.block, tuple(sorted(tuple(kv.split("=", 1)) for kv in a.props)))
    vox = Voxelizer(Assets(instance.asset_paths(a.instance)), 16)
    tris = ExactModels(vox).block_tris(state)
    print(f"{a.block}: {len(tris)} triangles; unknown={vox.unknown or '-'}")
    views = [render(tris, math.radians(-35), math.radians(25)), render(tris, math.radians(145), math.radians(25))]
    write_png(a.output, np.concatenate(views, 1))


if __name__ == "__main__":
    main()


