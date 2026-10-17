"""Create-style connected textures (CT), as Create's ConnectedTextureBehaviour draws them.

A texture `X` with a sibling sheet `X_connected` is connected: each face picks one tile of
the sheet from which neighbours in the face's plane are the same block. The rules below are
read from Create's own classes (AllCTTypes.getTextureIndex, ConnectedTextureBehaviour
.buildContext / getUpDirection / getRightDirection / isBeingBlocked):

- right/up on a face: up is world UP for side faces and NORTH for top/bottom faces; right is
  SOUTH for x-axis faces, otherwise WEST; positive faces reverse right; DOWN faces reverse both.
- a neighbour connects when it is the same block and no full block covers it on that side.
- sheet size picks the type: 8x8 OMNIDIRECTIONAL, 4x4 RECTANGLE, 2x2 HORIZONTAL.
Used by Create, Steam 'n' Rails, Design n' Decor, Create Deco and other Create addons.
"""
DIRS = {"north": (0, 0, -1), "south": (0, 0, 1), "west": (-1, 0, 0), "east": (1, 0, 0),
        "up": (0, 1, 0), "down": (0, -1, 0)}
OPP = {"north": "south", "south": "north", "west": "east", "east": "west", "up": "down", "down": "up"}


def face_axes(face):
    """(horizontal, vertical) world directions of a face's texture right and up."""
    up = "up" if face in ("north", "south", "east", "west") else "north"
    right = "south" if face in ("east", "west") else "west"
    horizontal = OPP[right] if face in ("south", "east", "up") else right
    vertical = up
    if face == "down":
        horizontal, vertical = OPP[horizontal], OPP[vertical]
    return horizontal, vertical


def omnidirectional(c):
    up, down, left, right = c["up"], c["down"], c["left"], c["right"]
    tl, tr, bl, br = c["topLeft"], c["topRight"], c["bottomLeft"], c["bottomRight"]
    x = y = 0
    borders = (not up) + (not down) + (not left) + (not right)
    if up:
        x += 1
    if down:
        x += 2
    if left:
        y += 1
    if right:
        y += 2
    if borders == 0:
        if tr:
            x += 1
        if tl:
            x += 2
        if br:
            y += 2
        if bl:
            y += 1
    if borders == 1:
        if not right and (tl or bl):
            y, x = 4, -1 + bl + 2 * tl
        if not left and (tr or br):
            y, x = 5, -1 + br + 2 * tr
        if not down and (tl or tr):
            y, x = 6, -1 + tl + 2 * tr
        if not up and (bl or br):
            y, x = 7, -1 + bl + 2 * br
    if borders == 2:
        if (up and left and not tl) or (down and left and not bl) or (up and right and not tr) or \
                (down and right and not br):
            x += 3
    return x + 8 * y


def rectangle(c):
    x = 2 if c["left"] and c["right"] else 3 if c["left"] else 1 if c["right"] else 0
    y = 1 if c["up"] and c["down"] else 2 if c["up"] else 0 if c["down"] else 3
    return x + 4 * y


def horizontal(c):
    return (1 if c["right"] else 0) + (2 if c["left"] else 0)


TYPES = {8: omnidirectional, 4: rectangle, 2: horizontal}


class ConnectedTextures:
    def __init__(self, assets, identity, full, copycat=None, face_mask=None):
        """identity: (x, y, z) -> what the block connects as; full: (x, y, z) -> full cube?

        copycat: positions of copycat blocks. Copycats+ (ICopycatBlock.checkConnection ->
        BlockFaceUtils.faceMatch) only lets a copycat connect across a side when both blocks'
        face outlines on that side are identical and not empty. face_mask(pos, direction)
        gives that outline (bytes) for any block.
        """
        self.assets = assets
        self.identity = identity
        self.full = full
        self.copycat = copycat or set()
        self.face_mask = face_mask
        self._sheets = {}

    def _faces_match(self, a, b):
        """Copycats+ faceMatch between neighbouring blocks a and b (b next to a on one side)."""
        d = tuple(bb - aa for aa, bb in zip(a, b))
        side = {v: k for k, v in DIRS.items()}.get(d)
        if side is None or self.face_mask is None:
            return True
        fa, fb = self.face_mask(a, side), self.face_mask(b, OPP[side])
        return bool(fa) and fa == fb

    # Textures with a `_connected` sibling that is NOT a connected-texture sheet (Create's
    # pipes use pipes_connected for joined pipe ends, picked by the pipe model itself).
    NOT_CT = ("create:block/pipes", "create:block/glass_fluid_pipe")

    def sheet(self, texture):
        """(connected texture id, tiles per row) or None."""
        if texture in self._sheets:
            return self._sheets[texture]
        result = None
        if texture and texture not in self.NOT_CT:
            ct = texture + "_connected"
            png = self.assets.texture_png(ct)
            base = self.assets.texture_png(texture)
            if png and base:
                import io
                from PIL import Image
                cw = Image.open(io.BytesIO(self.assets.raw_texture(ct))).size[0]
                bw = Image.open(io.BytesIO(base[0])).size[0]
                n = cw // bw if bw else 0
                if n in TYPES:
                    result = (ct, n)
        self._sheets[texture] = result
        return result

    def tile(self, pos, face, n):
        """Tile (column, row) on an n x n sheet for the face of the block at pos."""
        me = self.identity.get(pos)
        h, v = face_axes(face)
        dh, dv, df = DIRS[h], DIRS[v], DIRS[face]

        def step(p, a, b):
            return (p[0] + dh[0] * a + dv[0] * b, p[1] + dh[1] * a + dv[1] * b, p[2] + dh[2] * a + dv[2] * b)

        def test(a, b):
            q = step(pos, a, b)
            if self.identity.get(q) != me:
                return False
            front = (q[0] + df[0], q[1] + df[1], q[2] + df[2])
            if self.full.get(front):  # covered on this side: no connection
                return False
            if pos in self.copycat or q in self.copycat:
                # Copycats+: touching faces must match, directly or through each in-between block
                if a and b:
                    via_h, via_v = step(pos, a, 0), step(pos, 0, b)
                    return (self._faces_match(pos, via_h) and self._faces_match(via_h, q)) or                            (self._faces_match(pos, via_v) and self._faces_match(via_v, q))
                return self._faces_match(pos, q)
            return True

        c = {"up": test(0, 1), "down": test(0, -1), "left": test(-1, 0), "right": test(1, 0)}
        c["topLeft"] = c["up"] and c["left"] and test(-1, 1)
        c["topRight"] = c["up"] and c["right"] and test(1, 1)
        c["bottomLeft"] = c["down"] and c["left"] and test(-1, -1)
        c["bottomRight"] = c["down"] and c["right"] and test(1, -1)
        index = TYPES[n](c)
        return index % n, index // n
