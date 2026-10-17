"""Top-down world map, ported from Mineways (MinewaysMap.cpp draw(), initColors(), and
ObjFileManip.cpp GetMinimumSelectionHeight()/analyzeChunk()).

Each map pixel is one block column. From the top of the view (the "height" slider) down, the
first visible block gives the colour; see-through blocks (water, glass) are blended "under"
until an opaque block is hit. Mineways' shading follows: a column lit from the west gets
light 12, +2 if the column to the west is lower, -5 if it is higher (clamped 1..15), applied
in YUV so colours darken evenly; deeper blocks can be darkened by elevation. The selection
box is tinted purple, and its border drawn stronger.

Block colours come from the instance's own models (see BlockColors and blocktable), so modded
blocks get their real colour. Chunks are decoded a section at a time, only as far down as the
view needs: a top-down view usually touches just the top one or two sections of each chunk.
"""
import hashlib
import io
import json
import os
import threading
from collections import OrderedDict

import numpy as np

from .schematic import AIR, BlockState, Structure
from .world import _unpack_indices, palette_states

# MinewaysMap.cpp: highlight colour and strengths
H_ALPHA, H_ALPHA_BORDER = 0.3, 0.8
H_COLOR = np.array([205, 50, 255], np.float32)
WATER_TINT = np.array([0x3F, 0x76, 0xE4], np.float32) / 255
WATER_ALPHA = 0.6
EMPTY = -1
# properties that don't change how a block looks from above
SAME_LOOK = {"waterlogged", "distance", "persistent", "level", "power", "powered", "age", "moisture",
             "stage", "snowy", "note", "instrument", "delay", "locked", "triggered", "enabled", "inverted",
             "occupied", "signal_fire", "has_book", "honey_level", "bites", "charges", "candles", "pickles",
             "eggs", "hatch", "rotation", "in_wall", "open", "conditional", "mode", "lit"}
TABLE_DIR = os.path.join(os.path.expanduser("~"), ".modded2model", "blocktables")


def _blank_tile():
    """Mineways' gray checkerboard for ungenerated chunks (4x4 checks, tones 150/140)."""
    r = np.arange(16)
    check = ((r[:, None] // 4) % 2) ^ ((r[None, :] // 4) % 2)
    tone = np.where(check, 140, 150).astype(np.uint8)
    return np.repeat(tone[:, :, None], 3, 2)


BLANK = _blank_tile()
_AIR_LAYER = np.zeros((16, 16), np.uint16)


def shade(rgb, light):
    """initColors(): darken colour(s) to light level 0..15 in YUV, as gBlockColors[type*16+light]."""
    rgb = np.asarray(rgb, np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = (b - y) * 0.565
    v = (r - y) * 0.713
    y = np.asarray(light, np.float32) * (y / 15)
    out = np.stack([y + 1.403 * v, y - 0.344 * u - 0.714 * v, y + 1.770 * u], -1)
    return np.clip(out, 0, 255)


def _sources_signature(asset_paths):
    """Names, sizes and dates of the mod jars, resource packs and game jar: if any change
    (a mod added, removed or updated), the instance's block table is rebuilt."""
    sig = []
    for p in asset_paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                full = os.path.join(p, f)
                if f.lower().endswith((".jar", ".zip")) and os.path.isfile(full):  # mods, not their configs
                    st = os.stat(full)
                    sig.append(f"{f}:{st.st_size}:{int(st.st_mtime)}")
        elif os.path.exists(p):
            st = os.stat(p)
            sig.append(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha1(chr(10).join(sig).encode()).hexdigest()


def table_path(instance):
    """Where an instance's block table is kept (one file per Minecraft instance)."""
    key = hashlib.sha1(os.path.normcase(os.path.abspath(instance)).encode()).hexdigest()[:16]
    return os.path.join(TABLE_DIR, f"{key}.json")


def _key(state):
    return state.name + ("[" + ",".join(f"{k}={v}" for k, v in state.props) + "]" if state.props else "")


def _entry(values):
    return np.array(values[:3], np.float32), float(values[3])


class BlockColors:
    """Map colour and opacity of each block state, from its model.

    Like Mineways' block table, but built from the instance's own models (blocktable.build
    fills it for every block of every mod). A block the table doesn't have is measured from its
    model the first time it is seen and added to the table."""

    def __init__(self, vox, instance=None, asset_paths=()):
        """vox: a Voxelizer, or a function making one. It is only needed (and the mods only
        loaded) for blocks the instance's table doesn't have."""
        self._vox = vox
        self.cache = {}
        self._tex = {}
        self._table, self._dirty = {}, False
        self._jobs, self._variants, self._names = {}, {}, {}
        self._lock = threading.Lock()
        self.path = table_path(instance) if instance else None
        self._sig = _sources_signature(asset_paths) if instance else None
        self.reload()

    @property
    def vox(self):
        if callable(self._vox):
            self._vox = self._vox()
        return self._vox

    def reload(self):
        """Read the instance's table from disk (after blocktable.build has filled it)."""
        if not (self.path and os.path.isfile(self.path)):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        if data.get("sources") != self._sig:
            return
        with self._lock:
            data.setdefault("blocks", {}).update(self._table)
            self._table = data["blocks"]
            self._jobs = data.get("jobs", {})
            self._variants = data.get("variants", {})
            self._names = data.get("names", {})
        self.cache.clear()

    def save(self):
        """Add newly measured blocks to the instance's table file."""
        with self._lock:
            if not (self.path and self._dirty):
                return
            data = {}
            try:  # the full build may have written more since we read it: only add our entries
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                pass
            if data.get("sources") != self._sig:
                data = {"sources": self._sig}
            data.setdefault("blocks", {}).update(self._table)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
            self._dirty = False

    def get(self, state):
        """(rgb 0..255, alpha 0..1). alpha 0 means not shown on the map (air, tiny details)."""
        out = self.cache.get(state)
        if out is not None:
            return out
        out = self._lookup(state)
        if out is None:
            reduced = BlockState(state.name, tuple(kv for kv in state.props if kv[0] not in SAME_LOOK))
            default = self._jobs.get(self._names.get(state.name, ""))
            if reduced != state:
                out = self.get(reduced)
            elif default is not None:  # a state its blockstate file doesn't list: the block's default
                out = _entry(default)
            else:
                out = self._uncached(state)
                with self._lock:
                    self._table[_key(state)] = [round(float(c), 1) for c in out[0]] + [round(float(out[1]), 3)]
                    self._dirty = True
        self.cache[state] = out
        return out

    def _lookup(self, state):
        """The table's entry for this exact state, or for the blockstate variant it matches."""
        hit = self._table.get(_key(state))
        if hit is None and state.name in self._variants:
            from .assets import _variant_matches
            props = state.properties
            for key, job in self._variants[state.name]:
                if _variant_matches(key, props):
                    hit = self._jobs.get(job)
                    break
        return None if hit is None else _entry(hit)

    def _uncached(self, state):
        """Measure a block state from its model (no table)."""
        name = state.name
        if name in AIR or name.endswith((":light", "structure_void")):
            return np.zeros(3), 0.0
        if name in ("minecraft:water", "minecraft:bubble_column"):
            return WATER_TINT * 255, WATER_ALPHA
        if name == "minecraft:lava":
            return np.array([207, 92, 20], np.float32), 1.0
        return self._from_model(state)

    def _tex_mean(self, tex):
        if tex not in self._tex:
            from PIL import Image
            png = self.vox.assets.texture_png(tex) if tex else None
            if not png:
                self._tex[tex] = (np.array([0.6, 0.6, 0.6]), 1.0)
            else:
                a = np.asarray(Image.open(io.BytesIO(png[0])).convert("RGBA"), np.float32) / 255
                a = a[: a.shape[1]]  # first frame of animated textures
                w = a[..., 3].sum()
                rgb = (a[..., :3] * a[..., 3:]).sum((0, 1)) / w if w else np.zeros(3)
                self._tex[tex] = (rgb, float(a[..., 3].mean()))
        return self._tex[tex]

    def _from_model(self, state):
        from .blender import TexturedBuilder
        s = Structure((1, 1, 1))
        s.blocks.append((0, 0, 0, state))
        try:
            faces = TexturedBuilder(self.vox, s).build()
        except Exception:
            return np.array([150, 150, 150], np.float32), 1.0
        hp, hcol = [], []  # horizontal triangles (x, y, z corners) and their (rgb, alpha)
        vlo, vhi, vrgb, varea = np.full(2, 9.0), np.full(2, -9.0), np.zeros(3), 0.0
        for (tex, tint), (pos, _uv) in faces.groups.items():
            if not pos:
                continue
            p = np.concatenate(pos).reshape(-1, 3, 3) / 16  # model units are 0..16
            n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            ln = np.linalg.norm(n, axis=1)
            rgb, alpha = self._tex_mean(tex)
            if tint is not None:
                rgb = rgb * np.array(tint)
            horiz = (ln > 1e-9) & (np.abs(n[:, 1]) > 0.7 * ln)
            hp.extend(p[horiz])
            hcol.extend([(rgb, alpha)] * int(horiz.sum()))
            vert = (ln > 1e-9) & ~horiz
            if vert.any():
                xz = p[vert][..., [0, 2]].reshape(-1, 2)
                vlo, vhi = np.minimum(vlo, xz.min(0)), np.maximum(vhi, xz.max(0))
                varea += ln[vert].sum() / 2
                vrgb += rgb * ln[vert].sum() / 2
        # look down at 8x8 points of the column: each sees the highest horizontal face over it
        g = (np.arange(8) + 0.5) / 8
        px, pz = [a.ravel() for a in np.meshgrid(g, g, indexing="ij")]
        best_y = np.full(64, -9.0)
        best = np.full(64, -1)
        for k, t in enumerate(hp):
            (ax, ay, az), (bx, by, bz), (cx, cy, cz) = t
            den = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
            if abs(den) < 1e-12:
                continue
            w0 = ((bz - cz) * (px - cx) + (cx - bx) * (pz - cz)) / den
            w1 = ((cz - az) * (px - cx) + (ax - cx) * (pz - cz)) / den
            w2 = 1 - w0 - w1
            y = w0 * ay + w1 * by + w2 * cy
            hit = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6) & (y > best_y)
            best_y[hit], best[hit] = y[hit], k
        seen = best[best >= 0]
        billboard = varea > 0.5 and (vhi - vlo).max() >= 0.5  # grass, flowers: Mineways shows those
        if len(seen):
            rgb = np.mean([hcol[k][0] for k in seen], 0)
            alpha = float(np.mean([hcol[k][1] for k in seen]))
            covered = len(seen) / 64
        elif billboard:
            rgb, alpha, covered = vrgb / varea, 1.0, 1.0
        else:
            return np.zeros(3), 0.0
        # Mineways only maps blocks that fill a good part of their column (BLF_WHOLE, HALF, STAIRS,
        # PANE, BILLBOARD...), skipping small things like torches and buttons
        if covered < 0.25:
            if not billboard:
                return np.zeros(3), 0.0
            rgb, alpha = vrgb / varea, 1.0  # billboard drawn as pixel boxes
        elif alpha < 0.98:
            alpha = max(0.25, alpha) if alpha < 0.6 else 1.0  # glass is see-through, cutout leaves are not
        return np.clip(rgb * 255, 0, 255), alpha


_AIR_NAMES = tuple(name.encode() for name in AIR)


def _all_air(key):
    """An empty section, told from the palette's raw bytes (one entry, an air block) without
    parsing it: most of a world's sections are sky."""
    return int.from_bytes(key[2:6], "big", signed=True) == 1 and any(a in key for a in _AIR_NAMES)


class _Chunk:
    """One chunk's blocks, decoded a 16-block section at a time and only when asked for."""
    __slots__ = ("map", "sections", "_decoded", "_top")

    def __init__(self, wmap, data):
        self.map = wmap
        self.sections = {}  # section index from the map's bottom -> (palette key, packed indices)
        base = wmap.min_y >> 4
        for y, key, packed in data:
            i = y - base
            if 0 <= i < wmap.n_sections and not _all_air(key):
                self.sections[i] = (key, packed)
        self._decoded = {}
        self._top = None

    def section(self, i):
        """(16, 16, 16) map state ids [y][z][x] of section i, or None if it is all air."""
        if i not in self._decoded:
            ids = None
            if i in self.sections:
                key, packed = self.sections[i]
                states = palette_states(key)
                ids = self.map._lut(key, states)[_unpack_indices(packed, len(states))].reshape(16, 16, 16)
            self._decoded[i] = ids
        return self._decoded[i]

    def layer(self, i):
        """(16, 16) state ids of grid layer i (y = min_y + i)."""
        sec = self.section(i >> 4)
        return _AIR_LAYER if sec is None else sec[i & 15]

    def top(self):
        """Highest grid layer with any block, or -1."""
        if self._top is None:
            self._top = -1
            for i in sorted(self.sections, reverse=True):
                rows = np.nonzero(self.section(i).reshape(16, -1).any(1))[0]
                if len(rows):
                    self._top = i * 16 + int(rows.max())
                    break
        return self._top

    def stack(self, hi):
        """(hi + 1, 16, 16) state ids of grid layers 0..hi."""
        out = np.zeros((hi + 1, 16, 16), np.uint16)
        for i in self.sections:
            n = min(16, hi + 1 - i * 16)
            if n > 0:
                out[i * 16:i * 16 + n] = self.section(i)[:n]
        return out


class _Lru(OrderedDict):
    """A dict that forgets its oldest entries past `limit` (tiles for old Height settings)."""

    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.limit:
            self.popitem(last=False)


class WorldMap:
    """Draws map tiles (one per chunk) and answers height queries for a World."""

    def __init__(self, world, vox, min_y=-64, max_y=319, max_chunks=3000, instance=None, asset_paths=()):
        self.world = world
        self.colors = BlockColors(vox, instance, asset_paths)
        self.min_y, self.max_y = min_y, max_y
        self.n_sections = (max_y - min_y + 1) // 16
        self.states, self._ids, self._luts = [BlockState("minecraft:air", ())], {}, {}
        self._chunks = OrderedDict()  # (cx, cz) -> _Chunk, or None (not generated)
        self._max_chunks = max_chunks
        self._tiles = _Lru(40000)  # (cx, cz, top) -> (rgb 16x16x3, height, first-solid height)
        self._hts = _Lru(40000)  # (cx, cz, top) -> (height, first solid, first visible) or None
        self.elevation_shading = False  # off by default, as in Mineways
        self._rgb = np.zeros((1, 3), np.float32)
        self._alpha = np.zeros(1, np.float32)
        self._lit = np.zeros((1, 3, 3), np.float32)

    def set_elevation_shading(self, on):
        if on != self.elevation_shading:
            self.elevation_shading = on
            self._tiles.clear()

    # -- chunk data ------------------------------------------------------------------------------
    def _lut(self, key, states):
        """Palette index -> map state id (air is 0), one table per palette."""
        lut = self._luts.get(key)
        if lut is None:
            ids = []
            for st in states:
                i = 0
                if st.name not in AIR:
                    i = self._ids.get(st)
                    if i is None:
                        i = self._ids[st] = len(self.states)
                        self.states.append(st)
                ids.append(i)
            lut = self._luts[key] = np.array(ids, np.uint16)
        return lut

    def chunk(self, cx, cz):
        key = (cx, cz)
        if key in self._chunks:
            self._chunks.move_to_end(key)
            return self._chunks[key]
        data = self.world.section_data(cx, cz)
        ch = self._chunks[key] = _Chunk(self, data) if data else None
        if len(self._chunks) > self._max_chunks:
            self._chunks.popitem(last=False)
        return ch

    def _props(self):
        """Opacity per state id, growing the colour tables as new states appear."""
        n = len(self.states)
        if len(self._alpha) < n:
            rgb = np.zeros((n, 3), np.float32)
            alpha = np.zeros(n, np.float32)
            rgb[:len(self._alpha)], alpha[:len(self._alpha)] = self._rgb, self._alpha
            for i in range(len(self._alpha), n):
                rgb[i], alpha[i] = self.colors.get(self.states[i])
            self._rgb, self._alpha = rgb, alpha
            # premultiplied colour at light 7 (shadow), 12 (normal) and 14 (lit from the west)
            self._lit = np.stack([shade(rgb, light) * alpha[:, None] for light in (7, 12, 14)], 1)
        return self._alpha

    # -- draw() ----------------------------------------------------------------------------------
    def _heights(self, cx, cz, top):
        """Pass 1 of draw(): per column [z][x] the displayed height (first opaque block from the
        top), the first solid block (for the selection) and the first visible one, as grid
        layers. None if the chunk isn't generated."""
        key = (cx, cz, top)
        if key in self._hts:
            return self._hts[key]
        ch = self.chunk(cx, cz)
        out = None
        if ch is not None:
            height = np.full((16, 16), EMPTY, np.int32)
            first = np.full((16, 16), EMPTY, np.int32)
            for i in range(min(top - self.min_y, ch.top()), -1, -1):
                ids = ch.layer(i)  # decoding a section can add new states,
                a = self._props()[ids]  # so look opacity up after it
                first[(a > 0) & (first == EMPTY)] = i
                height[(a >= 1) & (height == EMPTY)] = i
                if (height != EMPTY).all():
                    break
            out = (height, first, first)  # the first solid block is also the first visible one
        self._hts[key] = out
        return out

    def tile(self, cx, cz, top):
        """Mineways draw() for one chunk, without the highlight: (rgb uint8 16x16x3 [z][x],
        displayed height [z][x] (EMPTY if none), first solid height [z][x] for selection)."""
        key = (cx, cz, top)
        if key in self._tiles:
            return self._tiles[key]
        hts = self._heights(cx, cz, top)
        if hts is None:
            out = self._tiles[key] = (BLANK, None, None)
            return out
        ch = self.chunk(cx, cz)
        height, sel, first = hts
        # the west neighbour's heights, for shading (Mineways reads the west chunk's heightmap)
        west = np.empty((16, 16), np.int32)
        west[:, 1:] = height[:, :-1]
        wh = self._heights(cx - 1, cz, top)
        west[:, 0] = wh[0][:, 15] if wh is not None else EMPTY
        west = np.where(west == EMPTY, first, west)  # no block to the west: compare with itself

        # pass 2: colours, blending see-through blocks "under" the ones above
        alpha_of = self._props()  # every state in reach was registered by pass 1
        r = np.zeros((16, 16, 3), np.float32)
        acc = np.zeros((16, 16), np.float32)
        done = np.zeros((16, 16), bool)
        for i in range(min(top - self.min_y, ch.top()), -1, -1):
            ids = ch.layer(i)
            a = alpha_of[ids]
            take = (a > 0) & ~done
            if not take.any():
                continue
            li = np.where(west < i, 2, np.where(west > i, 0, 1))  # light 14 / 7 / 12
            r[take] += (1 - acc[take])[:, None] * self._lit[ids[take], li[take]]  # premultiplied
            acc[take] += a[take] * (1 - acc[take])
            done |= take & (a >= 1)
            if done.all():
                break

        if self.elevation_shading:  # View > Elevation shading: darken deeper blocks (grid heights)
            h_alloc = top - self.min_y
            num = height + 50 - int((256 - h_alloc) / 5)
            den = h_alloc + 50 - int((256 - h_alloc) / 5)
            if den <= 0:
                num, den = np.ones_like(height), 1
            r = np.where((height >= 0)[..., None], r * (num / den)[..., None], r)
        empty = (height == EMPTY) & (acc == 0)
        r[empty] = BLANK[empty]
        out = self._tiles[key] = (np.clip(r, 0, 255).astype(np.uint8), height, sel)
        return out

    def render(self, x0, z0, w, h, top, box=None, progress=None, stop=None):
        """Map of blocks x0..x0+w-1, z0..z0+h-1 as (h, w, 3) uint8; box = (minx, miny, minz,
        maxx, maxy, maxz) is highlighted like Mineways' selection.

        Chunks drawn before (in any earlier view) are reused straight away; only new ones are
        drawn, nearest the centre first, calling progress(fraction, partial image) as they come.
        stop() is checked between new chunks: if it returns True, None is returned (the chunks
        drawn so far stay cached for the next view)."""
        img = np.empty((h, w, 3), np.uint8)
        img[:] = (30, 30, 34)  # not drawn yet (for progressive display)
        sel = np.full((h, w), -2, np.int32)
        mx, mz = (x0 + w / 2) / 16, (z0 + h / 2) / 16
        chunks = [(cx, cz) for cz in range(z0 >> 4, ((z0 + h - 1) >> 4) + 1)
                  for cx in range(x0 >> 4, ((x0 + w - 1) >> 4) + 1)]
        cached = [c for c in chunks if (c[0], c[1], top) in self._tiles]
        new = sorted((c for c in chunks if (c[0], c[1], top) not in self._tiles),
                     key=lambda c: (c[0] + 0.5 - mx) ** 2 + (c[1] + 0.5 - mz) ** 2)
        for k, (cx, cz) in enumerate(cached + new, 1):
            if k > len(cached):
                if stop and stop():
                    return None
                if progress and (k - len(cached)) % 16 == 1:
                    progress(k / len(chunks), img)
            rgb, _, s = self.tile(cx, cz, top)
            bx, bz = cx * 16 - x0, cz * 16 - z0
            sx0, sz0 = max(0, -bx), max(0, -bz)
            sx1, sz1 = min(16, w - bx), min(16, h - bz)
            img[bz + sz0:bz + sz1, bx + sx0:bx + sx1] = rgb[sz0:sz1, sx0:sx1]
            if s is not None:
                sel[bz + sz0:bz + sz1, bx + sx0:bx + sx1] = s[sz0:sz1, sx0:sx1]
        if new:
            self.colors.save()
        return self._highlight(img, sel, x0, z0, box) if box else img

    def _highlight(self, img, sel, x0, z0, box):
        minx, miny, minz, maxx, maxy, maxz = box
        h, w = sel.shape
        xs = np.arange(w)[None, :] + x0
        zs = np.arange(h)[:, None] + z0
        inside = (xs >= minx) & (xs <= maxx) & (zs >= minz) & (zs <= maxz)
        edge = inside & ((xs == minx) | (xs == maxx) | (zs == minz) | (zs == maxz))
        y = np.where(sel >= 0, sel + self.min_y, sel)
        none = sel < 0  # void or ungenerated: always highlighted, for readability
        in_range = inside & (none | ((y >= miny) & (y <= maxy)))
        on_y_edge = (y == miny) | (y == maxy)
        blend = np.where(edge | (on_y_edge & ~none), H_ALPHA_BORDER, H_ALPHA)[..., None]
        f = img.astype(np.float32)
        f = np.where(in_range[..., None], f * (1 - blend) + blend * H_COLOR, f)
        below = inside & ~none & (y < miny) & edge
        f = np.where(below[..., None], f * 0.5, f)
        return f.astype(np.uint8)

    # -- queries ---------------------------------------------------------------------------------
    def block_at(self, x, z, top):
        """IDBlock(): (y, BlockState) of the block shown at column x, z, or (None, None)."""
        _, _, sel = self.tile(x >> 4, z >> 4, top)
        if sel is None or sel[z & 15, x & 15] < 0:
            return None, None
        i = int(sel[z & 15, x & 15])
        return i + self.min_y, self.states[int(self.chunk(x >> 4, z >> 4).layer(i)[z & 15, x & 15])]

    def _boxes(self, minx, minz, maxx, maxz, maxy):
        """Per generated chunk overlapping the box: its state ids for layers 0..maxy, cut to the box."""
        hi = maxy - self.min_y
        if hi < 0:
            return
        for cx in range(minx >> 4, (maxx >> 4) + 1):
            for cz in range(minz >> 4, (maxz >> 4) + 1):
                ch = self.chunk(cx, cz)
                if ch is None:
                    continue
                x0, x1 = max(minx, cx * 16) - cx * 16, min(maxx, cx * 16 + 15) - cx * 16
                z0, z1 = max(minz, cz * 16) - cz * 16, min(maxz, cz * 16 + 15) - cz * 16
                yield ch.stack(hi)[:, z0:z1 + 1, x0:x1 + 1]

    def min_selection_height(self, minx, minz, maxx, maxz, maxy, expand_by_one=True, ignore_transparent=True):
        """GetMinimumSelectionHeight(): the lowest first-opaque block seen from maxy down in the
        box (grown by one block so a selection on a cliff edge reaches the bottom). 0 if none."""
        if expand_by_one:
            minx, minz, maxx, maxz = minx - 1, minz - 1, maxx + 1, maxz + 1
        found = None
        for g in self._boxes(minx, minz, maxx, maxz, maxy):
            col = self._props()[g]  # after _boxes decoded the sections
            hit = (col >= 1) if ignore_transparent else (col > 0)
            any_hit = hit.any(0)
            if any_hit.any():
                topmost = len(hit) - 1 - np.argmax(hit[::-1], 0)  # first hit from maxy down
                low = int(topmost[any_hit].min()) + self.min_y
                found = low if found is None else min(found, low)
        return 0 if found is None else found

    def max_selection_height(self, minx, minz, maxx, maxz, maxy):
        """Highest non-air block at or below maxy in the box (so an export isn't padded with sky)."""
        best = None
        for g in self._boxes(minx, minz, maxx, maxz, maxy):
            rows = np.nonzero(g.reshape(len(g), -1).any(1))[0]
            if len(rows):
                top = int(rows.max()) + self.min_y
                best = top if best is None else max(best, top)
        return best

    def height_range(self, cx, cz):
        """Lowest and highest non-air y in a chunk (for the sliders), or None."""
        ch = self.chunk(cx, cz)
        if ch is None or ch.top() < 0:
            return None
        g = ch.stack(ch.top())
        rows = np.nonzero(g.reshape(len(g), -1).any(1))[0]
        return int(rows.min()) + self.min_y, int(rows.max()) + self.min_y
