"""Text drawn on blocks, in Minecraft's own bitmap font (assets/minecraft/font/default.json).

Glyphs come from the font's bitmap sheets (ascii.png and the accented/European sheets); like the
game, each glyph is as wide as its rightmost non-transparent pixel column and advances one more
pixel; a space advances 4. The large Unifont fallback isn't loaded: characters outside the
bitmap sheets are left out.

Used for Aeronautics nameplates (NameplateRenderer, Simulated 1.3): the controller plate's name,
cut to fit the row of plates and centred, in the dye's text colour (darkened unless glowing).
"""
import io
import json

import numpy as np

# DyeColor.getTextColor(), by dye index (white = 0 ... black = 15)
DYE_TEXT = [0xFFFFFF, 0xFF681F, 0xFF00FF, 0x9AC0CD, 0xFFFF00, 0xBFFF00, 0xFF69B4, 0x808080,
            0xD3D3D3, 0x00FFFF, 0xA020F0, 0x0000FF, 0x8B4513, 0x00FF00, 0xFF0000, 0x000000]
_FONTS = {}


class Font:
    def __init__(self, assets):
        self.glyphs = {}  # char -> (texture, (u0, v0, u1, v1) in 0..16, width, height, top, advance)
        self.advances = {}  # spaces
        self._load(assets, "minecraft:default", set())

    def _load(self, assets, font_id, seen):
        if font_id in seen:
            return
        seen.add(font_id)
        ns, _, path = font_id.partition(":")
        raw = assets.raw(f"assets/{ns}/font/{path}.json")
        if raw is None:
            return
        for p in json.loads(raw.decode("utf-8-sig")).get("providers", []):
            kind = p.get("type")
            if kind == "reference":
                if "unifont" not in p.get("id", ""):
                    self._load(assets, p["id"], seen)
            elif kind == "space":
                for ch, adv in p.get("advances", {}).items():
                    self.advances.setdefault(ch, adv)
            elif kind == "bitmap":
                self._bitmap(assets, p)

    def _bitmap(self, assets, p):
        from PIL import Image
        ns, _, path = p["file"].partition(":")
        raw = assets.raw(f"assets/{ns}/textures/{path}")
        rows = p.get("chars", [])
        if raw is None or not rows:
            return
        img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))
        ih, iw = img.shape[:2]
        cw, ch = iw // max(len(r) for r in rows), ih // len(rows)
        height = p.get("height", 8)
        scale = height / ch
        ascent = p.get("ascent", 7)
        texture = f"{ns}:{path[:-4] if path.endswith('.png') else path}"
        for r, line in enumerate(rows):
            for c, char in enumerate(line):
                if char in ("\u0000", " ") or char in self.glyphs:
                    continue
                cell = img[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw, 3]
                cols = np.nonzero(cell.any(0))[0]
                if not len(cols):
                    continue
                w = int(cols.max()) + 1
                uv = (c * cw / iw * 16, r * ch / ih * 16, (c * cw + w) / iw * 16, (r + 1) * ch / ih * 16)
                self.glyphs[char] = (texture, uv, w * scale, height, 7 - ascent, int(0.5 + w * scale) + 1)

    def advance(self, char):
        if char in self.glyphs:
            return self.glyphs[char][5]
        return self.advances.get(char, 4 if char.isspace() else 0)

    def width(self, text):
        return sum(self.advance(c) for c in text)

    def fit(self, text, max_width):
        """Font.plainSubstrByWidth: the longest start of `text` no wider than max_width."""
        out, w = "", 0
        for c in text:
            w += self.advance(c)
            if w > max_width:
                break
            out += c
        return out

    def layout(self, text):
        """[(texture, uv, x, y, width, height)] in font pixels (x right, y down), pen at 0, 0."""
        x, out = 0.0, []
        for c in text:
            g = self.glyphs.get(c)
            if g:
                tex, uv, w, h, top, adv = g
                out.append((tex, uv, x, top, w, h))
            x += self.advance(c)
        return out


def font(assets):
    key = id(assets)
    if key not in _FONTS:
        _FONTS[key] = Font(assets)
    return _FONTS[key]


def _rgb(color):
    return tuple(((color >> s) & 0xFF) / 255.0 for s in (16, 8, 0))


def nameplate_text(state, data, assets):
    """Glyph quads for an Aeronautics nameplate: [(texture, tint, corners (4x3, block units,
    TL TR BR BL seen from the front), uvs)] for the row's left (or single) plate, else []."""
    from .sable import _plain
    p = state.properties
    data = data or {}
    name = _plain(data.get("Name"))
    if p.get("position", "single") not in ("left", "single") or not name:
        return []
    f = font(assets)
    glowing = bool(data.get("Glow"))
    px = 5 if glowing else 6  # text height in block pixels
    width = int(data.get("Width") or 1)
    max_w = (width * 16 - 6) * 7 // px + 1
    text = f.fit(name, max_w)
    shift_x = (max_w - 1) / 2 - f.width(text) / 2
    s = px / 16 / 7  # one font pixel in blocks
    dye = int(data.get("TextColor", 0)) % 16
    color = DYE_TEXT[dye]
    if not glowing:  # NameplateBlockEntity.getDarkColor
        color = sum(int(((color >> sh) & 0xFF) * 0.4) << sh for sh in (16, 8, 0))
    tint = _rgb(color)
    # the plate's frame (facing north = the model as authored): pose rotated about the block centre
    to_y = {"south": 0, "west": 90, "north": 180, "east": 270}.get(p.get("facing", "north"), 180)
    a = np.radians(180 - to_y)
    c, sn = np.cos(a), np.sin(a)
    rot = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])  # Axis.YP.rotationDegrees
    out = []
    for tex, (u0, v0, u1, v1), gx, gy, gw, gh in f.layout(text):
        quad = []
        for fx, fy in ((gx, gy), (gx + gw, gy), (gx + gw, gy + gh), (gx, gy + gh)):
            fx += shift_x
            local = np.array([1 - 0.1875 - s * fx, 1 - (16 - px) / 32 - s * fy, 1 - 0.253125])
            quad.append((local - 0.5) @ rot.T + 0.5)
        out.append((tex, tint, np.array(quad), [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]))
    return out
