"""Shapes for blocks that mods draw with code instead of model files (copycats, pipes...).

Each rule takes the block's properties and returns boxes [x1, y1, z1, x2, y2, z2] in
0..16 block units, optionally with a 7th entry naming the copycat part whose material the
box shows (a key of the block entity's material_data); or an alias to another block whose
model files have the same shape.

Which side the material sits on was measured from real builds (which neighbour each block touches):
layers, panels and byte panels sit against the side OPPOSITE `facing` (the surface they were placed on);
slices, vertical slices, half layers, steps and vertical stairs sit against the `facing` side.
Rules marked "approx" are best guesses at the mod's real shape.
"""

OPPOSITE = {"up": "down", "down": "up", "north": "south", "south": "north", "east": "west", "west": "east"}
AXIS_OF = {"up": 1, "down": 1, "east": 0, "west": 0, "south": 2, "north": 2}
POSITIVE = {"up", "east", "south"}


def _against(side, thickness, lo=(0, 0, 0), hi=(16, 16, 16)):
    """Box `thickness` deep against `side`, limited to the lo..hi region on the other axes."""
    a = AXIS_OF[side]
    lo, hi = list(lo), list(hi)
    if side in POSITIVE:
        lo[a] = 16 - thickness
        hi[a] = 16
    else:
        lo[a] = 0
        hi[a] = thickness
    return lo + hi


def _half(axis, positive, lo=(0, 0, 0), hi=(16, 16, 16)):
    lo, hi = list(lo), list(hi)
    if positive:
        lo[axis] = 8
    else:
        hi[axis] = 8
    return lo, hi


def _perp(facing):
    """The horizontal axis across a horizontal facing direction."""
    return 2 if AXIS_OF[facing] == 0 else 0


def layer(p):
    return [_against(OPPOSITE[p["facing"]], 2 * int(p["layers"]))]


def panel(p):
    return [_against(OPPOSITE[p["facing"]], 3)]


def board(p):
    return [_against(s, 1) + [s] for s in OPPOSITE if p.get(s) == "true"]


def byte(p):
    boxes = []
    for key, val in p.items():
        if val != "true":
            continue
        vert, _, hz = key.partition("_")
        lo, hi = _half(1, vert == "top")
        lo, hi = _half(2, "south" in hz, lo, hi)
        lo, hi = _half(0, "east" in hz, lo, hi)
        boxes.append(lo + hi + [key])
    return boxes


def slab(p):  # Copycats+ slab: two parts, material per half ("bottom" / "top")
    axis = "xyz".index(p.get("axis", "y"))
    t = p.get("type", "bottom")
    halves = ["bottom", "top"] if t == "double" else [t]
    boxes = []
    for half in halves:
        lo, hi = _half(axis, half == "top")
        boxes.append(lo + hi + [half])
    return boxes

def beam(p):
    a = "xyz".index(p.get("axis", "y"))
    lo, hi = [4, 4, 4], [12, 12, 12]
    lo[a], hi[a] = 0, 16
    return [lo + hi]


# Copycats+ (CCShapes) defines these for facing south and turns them about the vertical
# axis for other facings. L = 2 * layers pixels.
_SOUTH_TO = {"south": 0, "west": 90, "north": 180, "east": 270}  # clockwise degrees seen from above


def _from_south(box, facing):
    """Turn a box defined for facing south to `facing` (about the block's vertical centre line)."""
    x1, y1, z1, x2, y2, z2 = box
    for _ in range(_SOUTH_TO[facing] // 90):  # each step: south -> west (x, z) -> (16 - z, x)
        x1, z1, x2, z2 = 16 - z2, x1, 16 - z1, x2
    return [x1, y1, z1, x2, y2, z2]


def _vertical_band(p, size):
    return (16 - size, 16) if p.get("half") == "top" else (0, size)


def slice_(p):  # full-width bar, L tall and L deep, along the facing wall, at the bottom or top
    n = 2 * int(p["layers"])
    y1, y2 = _vertical_band(p, n)
    return [_from_south([0, y1, 16 - n, 16, y2, 16], p["facing"])]


def vertical_slice(p):  # full-height column, L by L, in the corner of the facing and its counter-clockwise side
    n = 2 * int(p["layers"])
    return [_from_south([16 - n, 0, 16 - n, 16, 16, 16], p["facing"])]


def half_layer(p):  # approx: two independent layer stacks side by side along `axis`
    axis = "xyz".index(p["axis"])
    side = "up" if p.get("half") == "top" else "down"
    boxes = []
    for positive, key in ((False, "negative_layers"), (True, "positive_layers")):
        n = int(p[key])
        if n:
            lo, hi = _half(axis, positive)
            boxes.append(_against(side, 2 * n, lo, hi) + [key])
    return boxes


# Side of a horizontal facing that is counter-clockwise from it, seen from above.
COUNTER_CLOCKWISE = {"north": "west", "west": "south", "south": "east", "east": "north"}


def vertical_half_layer(p):  # wall layers split left/right across the wall
    wall = p["facing"]
    # positive_layers sits on the facing's counter-clockwise side and negative_layers on
    # its clockwise side, so mirrored builds match (checked against mm_pickup_truck_1 in
    # game: east/west side panels and the south-facing windshield frame).
    ccw = COUNTER_CLOCKWISE[wall]
    boxes = []
    for positive, key in ((False, "negative_layers"), (True, "positive_layers")):
        n = int(p[key])
        if n:
            side = ccw if positive else OPPOSITE[ccw]
            lo, hi = _half(_perp(wall), side in POSITIVE)
            boxes.append(_against(wall, 2 * n, lo, hi) + [key])
    return boxes


def stacked_half_layer(p):  # approx: wall layers split into bottom (negative) and top (positive) halves
    wall = p["facing"]
    boxes = []
    for positive, key in ((False, "negative_layers"), (True, "positive_layers")):
        n = int(p[key])
        if n:
            lo, hi = _half(1, positive)
            boxes.append(_against(wall, 2 * n, lo, hi) + [key])
    return boxes


def vertical_step(p):  # approx: full-height quarter column against the back wall
    lo, hi = _half(_perp(p["facing"]), False)
    return [_against(p["facing"], 8, lo, hi)]


def vertical_stairs(p):  # approx: back half plus a quarter column on `side`
    f = p["facing"]
    back = _against(f, 8)
    lo, hi = _half(_perp(f), (p.get("side") == "right") == (f in ("north", "east")))
    return [back, _against(OPPOSITE[f], 8, lo, hi)]


def step(p):  # approx: Create's copycat step: quarter block in the top/bottom half, against the facing side
    lo, hi = _half(1, p.get("half") == "top")
    return [_against(p["facing"], 8, lo, hi)]


def byte_panel(p):
    """Copycats+ CopycatBytePanelBlock.calculateMultiFaceShape: four 8x8 quarters, 3 px thick,
    against the facing side. Quarter (i, j) sits i*8 along the panel's horizontal direction and
    j*8 along its vertical one (mirrored on negative directions); bottom_right = (0, 0),
    bottom_left = (1, 0), top_right = (0, 1), top_left = (1, 1)."""
    f = p["facing"]
    horiz = {"down": "east", "up": "east", "north": "west", "south": "east", "west": "north", "east": "south"}[f]
    vert = {"down": "south", "up": "north"}.get(f, "up")
    positive = ("east", "up", "south")
    boxes = []
    for key, (i, j) in {"bottom_right": (0, 0), "bottom_left": (1, 0), "top_right": (0, 1), "top_left": (1, 1)}.items():
        if p.get(key) != "true":
            continue
        lo, hi = [0.0] * 3, [0.0] * 3
        for d, k in ((horiz, i), (vert, j)):
            a_ = AXIS_OF[d]
            start = k * 8 if d in positive else 8 - k * 8
            lo[a_], hi[a_] = start, start + 8
        a_ = AXIS_OF[f]
        start = 13 if f in positive else 0
        lo[a_], hi[a_] = start, start + 3
        boxes.append(lo + hi + [key])
    return boxes


def corner_slice(p):  # L cube in the same corner as a vertical slice, at the bottom or top
    n = 2 * int(p["layers"])
    y1, y2 = _vertical_band(p, n)
    return [_from_south([16 - n, y1, 16 - n, 16, y2, 16], p["facing"])]


def belt(p):  # approx: Create-style belt, 8 px thick through the middle
    if p.get("slope") == "vertical":
        a = AXIS_OF[p["facing"]]
        lo, hi = [0, 0, 0], [16, 16, 16]
        lo[a], hi[a] = 4, 12
        return [lo + hi]
    return [[0, 4, 0, 16, 12, 16]]


def pipe(p):
    boxes = [[4, 4, 4, 12, 12, 12]]
    for side in OPPOSITE:
        if p.get(side) == "true":
            lo, hi = [4, 4, 4], [12, 12, 12]
            boxes.append(_against(side, 4, lo, hi))
    if len(boxes) == 1:  # unconnected pipes render as a straight pipe
        boxes.append([4, 0, 4, 12, 16, 12])
    return boxes


# --- smooth slopes (exact geometry, not boxes) ---------------------------------------
# From Copycats+ CopycatSlopeModelCore / CopycatSlopeLayerModelCore: the top is a straight
# ramp from height h0 at the back edge to h1 at the facing edge, built for facing south
# (z = 16 is the facing edge), turned by the facing's yaw and flipped upside down for
# half=top. Slope: 0 -> 16. Slope layer n <= 4: 0 -> 4n; n > 4: 4(n - 4) -> 16.

def _ramp_tris(h0, h1, facing, top):
    import numpy as np
    def P(x, z, y):
        return [x, y, z]
    v = {  # base frame: back edge z=0 (height h0), facing edge z=16 (height h1)
        "b0": P(0, 0, 0), "b1": P(16, 0, 0), "b2": P(16, 16, 0), "b3": P(0, 16, 0),
        "t0": P(0, 0, h0), "t1": P(16, 0, h0), "t2": P(16, 16, h1), "t3": P(0, 16, h1)}
    quads = [("b0", "b1", "b2", "b3"),            # bottom (wound to face down)
             ("t0", "t3", "t2", "t1"),            # sloped top
             ("b3", "b2", "t2", "t3"),            # facing (high) end
             ("b0", "t0", "t1", "b1"),            # back (low) end, may be zero height
             ("b0", "b3", "t3", "t0"),            # west side
             ("b1", "t1", "t2", "b2")]            # east side
    tris = []
    for a, b, c, d in quads:
        pa, pb, pc, pd = (np.array(v[k], float) for k in (a, b, c, d))
        for t in ((pa, pb, pc), (pa, pc, pd)):
            if np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) > 1e-9:  # skip degenerate (zero-height end)
                tris.append(t)
    tris = np.array(tris)
    if top:  # hang from the ceiling: mirror in y, then fix winding
        tris[..., 1] = 16 - tris[..., 1]
        tris = tris[:, ::-1]
    # turn from facing south to `facing` about the vertical centre line (south -> west -> north -> east)
    for _ in range(_SOUTH_TO[facing] // 90):
        x, z = tris[..., 0].copy(), tris[..., 2].copy()
        tris[..., 0], tris[..., 2] = 16 - z, x
    return tris


def slope_mesh(p):
    return _ramp_tris(0, 16, p["facing"], p.get("half") == "top")


def slope_layer_mesh(p):
    n = int(p["layers"])
    h0, h1 = (0, 4 * n) if n <= 4 else (4 * (n - 4), 16)
    return _ramp_tris(h0, h1, p["facing"], p.get("half") == "top")


MESH_RULES = {
    "copycats:copycat_slope": slope_mesh,
    "copycats:copycat_slope_layer": slope_layer_mesh,
}


# Rule-shaped blocks that aren't copycats: the texture to draw them with.
RULE_TEXTURES = {}


# Parts chosen by block properties, drawn unrotated on top of the blockstate model.
# Create's fluid pipe: the multipart blockstate gives the core; PipeAttachmentModel adds a
# connection piece for every connected side.
def property_parts(state):
    if state.name == "create:fluid_pipe":
        on = {d for d in ("north", "south", "east", "west", "up", "down") if state.properties.get(d) == "true"}
        parts = []  # side attachments depend on neighbours: see pipe_attachments
        # The multipart only covers 1-2 connections around each axis; with 3-4 there is no piece
        # for that axis's two faces. Use the straight-through piece (plain copper side); the
        # core_<axis> pieces show the dark pipe interior (an open end) instead.
        around = {"x": ("down", "north", "south", "up"), "y": ("north", "south", "east", "west"),
                  "z": ("east", "west", "up", "down")}
        for axis, sides in around.items():
            if len(on & set(sides)) >= 3:
                parts.append(f"create:block/fluid_pipe/lr_{axis}")
        return parts
    return []


# --- Create fluid pipe side attachments (FluidTransportBehaviour.getRenderedRimAttachment,
# StandardPipeFluidTransportBehaviour override, AttachmentTypes -> ComponentPartials) ---
PIPE_DIRS = {"north": (0, 0, -1), "south": (0, 0, 1), "east": (1, 0, 0), "west": (-1, 0, 0),
             "up": (0, 1, 0), "down": (0, -1, 0)}
_OPP = {"north": "south", "south": "north", "east": "west", "west": "east", "up": "down", "down": "up"}
PIPE_LIKE = ("create:fluid_pipe", "create:glass_fluid_pipe", "create:encased_fluid_pipe",
             "create:smart_fluid_pipe", "create:fluid_valve")
# blocks that hold or move fluid (have a fluid capability): a pipe ends in a drain against them
FLUID_HOLDERS = ("tank", "engine", "basin", "spout", "drain", "boiler", "barrel", "pump", "ballast",
                 "electrolyzer", "generator", "hatch", "fluid_interface", "tap", "steam", "boiler")
ATTACHMENT_PARTS = {"CONNECTION": ("connection",), "DETAILED_CONNECTION": ("rim_connector",),
                    "RIM": ("rim_connector", "rim"), "PARTIAL_RIM": ("rim",), "DRAIN": ("rim_connector", "drain"),
                    "PARTIAL_DRAIN": ("drain",), "NONE": ()}


_SIDE_AXIS = {"north": "z", "south": "z", "east": "x", "west": "x", "up": "y", "down": "y"}
# Blocks with PipeAttachmentModel: their open ends get rims, drains and connection pieces
ATTACHMENT_BLOCKS = ("create:fluid_pipe", "create:encased_fluid_pipe", "create:glass_fluid_pipe",
                     "create:smart_fluid_pipe", "create:fluid_valve")
# StraightPipeFluidTransportBehaviour (and the valve's and smart pipe's behaviours built on it)
STRAIGHT_PIPES = ("create:glass_fluid_pipe", "create:smart_fluid_pipe", "create:fluid_valve")
# blocks with a FluidTransportBehaviour (pipes, pumps, valves)
_TRANSPORT = PIPE_LIKE + ("create:mechanical_pump",)


def _pipe_open(state, side):
    p = state.properties
    if state.name in ("create:fluid_pipe", "create:encased_fluid_pipe"):
        return p.get(side) == "true"
    if state.name == "create:glass_fluid_pipe":
        return _SIDE_AXIS[side] == p.get("axis", "y")
    if state.name == "create:fluid_valve":  # FluidValveBlock.getPipeAxis
        shaft = _SIDE_AXIS[p.get("facing", "north")]
        others = [a for a in "xyz" if a != shaft]
        return _SIDE_AXIS[side] == (others[0] if p.get("axis_along_first") == "true" else others[1])
    if state.name in ("create:smart_fluid_pipe", "create:mechanical_pump"):
        facing = p.get("facing", "north")
        if state.name == "create:smart_fluid_pipe" and p.get("face") in ("floor", "ceiling"):
            return _SIDE_AXIS[side] == "y"
        return _SIDE_AXIS[side] == _SIDE_AXIS[facing]
    return False


def _pipe_axis(state):
    """IAxisPipe.getAxisOf: the axis of a straight pipe (glass pipe, valve, smart pipe), else None."""
    p = state.properties
    if state.name == "create:glass_fluid_pipe":
        return p.get("axis", "y")
    if state.name in ("create:fluid_valve", "create:smart_fluid_pipe"):
        return next((a for a in "xyz" if _pipe_open(state, {"x": "east", "y": "up", "z": "south"}[a])), None)
    return None


def _base_attachment(state, side, n):
    """FluidTransportBehaviour.getRenderedRimAttachment (Create 6)."""
    if not _pipe_open(state, side):
        return "NONE"
    if n is not None and "pump" in n.name and n.properties.get("facing") == _OPP[side]:
        return "NONE"
    if n is not None and n.name == "create:encased_fluid_pipe" and _pipe_open(n, _OPP[side]):
        return "RIM"
    if n is not None and n.name not in _TRANSPORT and any(k in n.name for k in FLUID_HOLDERS) \
            and "hose_pulley" not in n.name:  # FluidPropagator.hasFluidCapability
        return "DRAIN"
    return "RIM"


def _standard_attachment(state, side, n, straight):
    """FluidPipeBlockEntity.StandardPipeFluidTransportBehaviour (fluid pipes, encased pipes)."""
    kind = _base_attachment(state, side, n)
    if state.name == "create:encased_fluid_pipe" and kind != "DRAIN":
        return "NONE"  # the casing covers the joints
    if kind == "RIM":
        if n is not None and n.name not in ("create:fluid_pipe", "create:encased_fluid_pipe",
                                             "create:glass_fluid_pipe") \
                and n.name in _TRANSPORT and _pipe_open(n, _OPP[side]):
            return "DETAILED_CONNECTION"  # pumps, valves, smart pipes joined to this side
        # FluidPipeBlock.shouldDrawRim: only another fluid pipe joined back to this side has no rim
        draw_rim = n is None or n.name != "create:fluid_pipe" or not _pipe_open(n, _OPP[side])
        if not draw_rim:
            return "CONNECTION" if straight == _SIDE_AXIS[side] else "DETAILED_CONNECTION"
        return "RIM"
    if kind == "NONE" and _pipe_open(state, side):
        return "DETAILED_CONNECTION"
    return kind


def _straight_attachment(state, side, n):
    """StraightPipeBlockEntity.StraightPipeFluidTransportBehaviour (glass pipes, valves, smart pipes)."""
    kind = _base_attachment(state, side, n)
    if kind == "RIM" and state.name == "create:fluid_valve":
        return "NONE"
    if kind == "RIM" and state.name != "create:glass_fluid_pipe" and n is not None \
            and n.name == "create:glass_fluid_pipe":
        return "PARTIAL_RIM"
    if kind == "RIM" and n is not None and n.name == "create:fluid_pipe":
        return "NONE"
    own, other = _pipe_axis(state), (_pipe_axis(n) if n is not None else None)
    if own is not None and own == other:
        return "NONE"
    if n is not None and n.name == "create:fluid_valve" and _pipe_axis(n) == _SIDE_AXIS[side]:
        return "NONE"
    return {"RIM": "PARTIAL_RIM", "DRAIN": "PARTIAL_DRAIN"}.get(kind, "NONE")  # withoutConnector


def should_draw_casing(open_sides):
    """FluidPipeBlock.shouldDrawCasing: a pipe junction (more than two openings around one axis,
    e.g. a T or a cross) gets the bulky casing cube (PipeAttachmentModel adds FLUID_PIPE_CASING)."""
    return any(sum(1 for d in open_sides if _SIDE_AXIS[d] != axis) > 2 for axis in "xyz")


def pipe_attachments(state, neighbour):
    """Side parts of a Create fluid pipe and the other PipeAttachmentModel blocks, following
    Create 6's rendered rim attachments. neighbour(side) -> BlockState or None."""
    if state.name not in ATTACHMENT_BLOCKS:
        return []
    on = [d for d in PIPE_DIRS if _pipe_open(state, d)]
    straight = _SIDE_AXIS[on[0]] if len(on) == 2 and _OPP[on[0]] == on[1] else None
    out = []
    if state.name == "create:fluid_pipe" and should_draw_casing(on):
        out.append("create:block/fluid_pipe/casing")
    for side in PIPE_DIRS:
        n = neighbour(side)
        if state.name in STRAIGHT_PIPES:
            kind = _straight_attachment(state, side, n)
        else:
            kind = _standard_attachment(state, side, n, straight)
        out += [f"create:block/fluid_pipe/{part}/{side}" for part in ATTACHMENT_PARTS[kind]]
    return out


# Create kinetic blocks whose shafts the renderer draws (KineticBlockEntityRenderer), not the
# model: "axis" = a full shaft along the block's axis; "gearbox" = half shafts out of the four
# sides around the axis; "six" = half shafts out of all six sides.
KINETIC_SHAFTS = {
    "create:encased_chain_drive": "axis", "create:adjustable_chain_gearshift": "axis",
    "create:gearshift": "axis", "create:clutch": "axis",
    "create:andesite_encased_shaft": "axis", "create:brass_encased_shaft": "axis",
    "create_connected:inverted_clutch": "axis", "create_connected:overstress_clutch": "axis",
    "create_connected:encased_chain_cogwheel": "axis", "create_connected:parallel_gearbox": "gearbox",
    "simulated:directional_gearshift": "axis",
    "create:gearbox": "gearbox", "create_connected:brass_gearbox": "gearbox",
    "create_connected:six_way_gearbox": "six",
}
# (x, y) blockstate rotation turning Create's south-pointing half shaft / y-axis shaft
_HALF_TO = {"south": (0, 0), "west": (0, 90), "north": (0, 180), "east": (0, 270), "up": (270, 0), "down": (90, 0)}
_AXIS_TO = {"y": (0, 0), "x": (90, 90), "z": (90, 0)}


def kinetic_shafts(state):
    """[(model id, (x rot, y rot))] for the shafts Create draws on this block."""
    mode = KINETIC_SHAFTS.get(state.name)
    if not mode:
        return []
    axis = state.properties.get("axis") or {"north": "z", "south": "z", "east": "x", "west": "x"}.get(
        state.properties.get("facing", ""), "y")
    if mode == "axis":
        return [("create:block/shaft", _AXIS_TO[axis])]
    sides = list(_HALF_TO) if mode == "six" else [d for d in _HALF_TO
                                                  if {"x": ("east", "west"), "y": ("up", "down"),
                                                      "z": ("north", "south")}[axis].count(d) == 0]
    return [("create:block/shaft_half", _HALF_TO[d]) for d in sides]


RULES = {
    # Create Connected's copycats behave like the Copycats+ ones of the same shape
    "create_connected:copycat_block": lambda p: [[0, 0, 0, 16, 16, 16]],
    "create_connected:copycat_slab": slab,
    "create_connected:copycat_board": board,
    "create_connected:copycat_beam": beam,
    "create_connected:copycat_vertical_step": vertical_step,
    "copycats:copycat_block": lambda p: [[0, 0, 0, 16, 16, 16]],
    "copycats:copycat_layer": layer,
    "copycats:copycat_board": board,
    "copycats:copycat_byte": byte,
    "copycats:copycat_slab": slab,
    "copycats:copycat_beam": beam,
    "copycats:copycat_slice": slice_,
    "copycats:copycat_vertical_slice": vertical_slice,
    "copycats:copycat_half_layer": half_layer,
    "copycats:copycat_vertical_half_layer": vertical_half_layer,
    "copycats:copycat_stacked_half_layer": stacked_half_layer,
    "copycats:copycat_vertical_step": vertical_step,
    "copycats:copycat_vertical_stairs": vertical_stairs,
    "create:copycat_panel": panel,
    "create:copycat_step": step,
    "copycats:copycat_byte_panel": byte_panel,
    "copycats:copycat_corner_slice": corner_slice,
    "copycats:copycat_fluid_pipe": pipe,
    "dndecor:belt": belt,
}

# Blocks whose blockstate points at an empty model but whose real parts ship as
# model files in the jar (drawn at runtime by a block entity renderer).
# name -> (model ids, the direction the models face when unrotated)
JAR_MODELS = {
    # drawn only from its block data: see _headlamps
    "bits_n_bobs:headlamp": ([], "up"),
    # drawn only by BeltRenderer: see _belt
    "create:belt": ([], "up"),
}

# Blocks whose blockstate model is only a base: the block entity draws extra parts
# (authored in block space, so they take the blockstate's rotation). Keys may be a
# "namespace:*suffix" pattern to cover colour variants.
EXTRA_MODELS = {
    "simulated:*_portable_engine": [
        f"simulated:block/portable_engine/{p}" for p in (
            "exhaust_outlet_left", "exhaust_outlet_right", "exhaust_pipe_left", "exhaust_pipe_right",
            "hatch_top", "hatch_bottom", "mouth")],
}


# Parts a block entity draws in the frame of one particular placed model, keyed by that
# model's id: they take the same blockstate rotation as the model they belong to.
EXTRA_BY_MODEL = {
    # Rope winch: the shaft runs through the side plates' holes (the rope drum is left out).
    "simulated:block/rope_winch/block_horizontal": ["simulated:block/rope_winch/shaft"],
    "simulated:block/rope_winch/block_vertical": ["create:block/shaft"],
    # Analog transmission: the cog and the shaft through it turn, so the renderer draws them.
    # Huge diesel engine (createdieselgenerators HugeDieselEngineRenderer): the piston, drawn in
    # the base model's frame and pushed 0.53475 blocks out along the engine axis at rest.
    "simulated:block/analog_transmission/block": ["simulated:block/analog_transmission/gear", "create:block/shaft"],
    "simulated:block/analog_transmission/block_on": ["simulated:block/analog_transmission/gear", "create:block/shaft"],
    # Diesel engines (createdieselgenerators DieselEngineRenderer, ModularDieselEngineRenderer):
    # the piston tops capping the cylinders, one of five animation frames turned to the facing.
    "createdieselgenerators:block/diesel_engine/block": ["createdieselgenerators:block/diesel_engine/pistons/pistons_0"],
    "createdieselgenerators:block/diesel_engine/block_vertical": ["createdieselgenerators:block/diesel_engine/pistons/vertical_0"],
    "createdieselgenerators:block/modular_diesel_engine/block": ["createdieselgenerators:block/modular_diesel_engine/pistons/pistons_0"],
}

# Parts a block entity draws outside its own block (offset in blocks along y), turned by the
# blockstate's yaw. From createbigcannons CannonMountBlockEntityRenderer: the rotating mount
# one block above (below, flipped, when the mount points up) and the cannon axle two above.
OFFSET_PARTS = {
    "createbigcannons:cannon_mount": [("createbigcannons:block/cannon_mount/rotating_mount", 1),
                                      ("createbigcannons:block/cannon_carriage/carriage_axle", 2)],
}


def _diesel_parts(state, xr, yr):
    """Huge diesel engine piston, linkage and shaft connector (HugeDieselEngineRenderer).

    Parts are drawn in the base model's frame (pointing up, turned like the placed model).
    With no stored crank angle the renderer's starting angle for the facing is used:
    piston up (1 - s)/4 + 0.4375 blocks, linkage one block further, pivoting at (0.25, 0.5)
    by 23 degrees * c, and the connector two blocks out on the shaft, turned by -a + 90 deg.
    """
    import math
    import numpy as np
    from .voxelize import _blockstate_matrix

    facing = state.properties.get("facing", "north")
    a = math.radians({"down": -90, "up": 90, "west": -90, "east": 90}.get(facing, 0))
    sign = -1 if facing in ("up", "down") else 1
    s1, s2 = math.sin(a) * sign, math.sin(a - math.pi / 2) * sign
    py = ((1 - s1) / 4 + 0.4375) * 16

    def T(x, y, z):
        m = np.eye(4); m[:3, 3] = (x, y, z); return m

    def RX(deg):
        c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        m = np.eye(4); m[1:3, 1:3] = [[c, -s_], [s_, c]]; return m

    R = np.eye(4); R[:3, :3] = _blockstate_matrix(xr, yr)
    C, Ci = T(8, 8, 8), T(-8, -8, -8)
    base = C @ R @ Ci  # turn the up-facing part like the placed model, about the block centre
    parts = {
        "piston": base @ T(0, py, 0),
        "linkage": base @ T(0, 16, 0) @ T(0, py, 0) @ T(0, 4, 8) @ RX(s2 * 23) @ T(0, -4, -8),
        "shaft_connector": base @ T(0, 32, 0) @ C @ RX(math.degrees(-a + math.pi / 2)) @ Ci,
    }
    out = []
    for name, M in parts.items():
        m3, t = M[:3, :3], M[:3, 3]
        # express x -> m3 x + t as m3 (x - c) + c + shift, the form the exporters use
        shift = m3 @ np.array([8.0, 8, 8]) + t - np.array([8.0, 8, 8])
        out.append((f"createdieselgenerators:block/huge_diesel_engine/{name}", m3, shift))
    return out


_CDG = "createdieselgenerators:block/"
# Diesel engine upgrades (createdieselgenerators EngineUpgrades.renderPartial): model per engine
# kind, horizontal or vertical.
_ENGINE_UPGRADES = {
    "silencer": {"createdieselgenerators:diesel_engine": ("diesel_engine/silencer", "diesel_engine/vertical_silencer"),
                 "createdieselgenerators:large_diesel_engine": ("modular_diesel_engine/silencer", None)},
    "turbocharger": {"createdieselgenerators:diesel_engine": ("diesel_engine/turbocharger",
                                                              "diesel_engine/vertical_turbocharger"),
                     "createdieselgenerators:large_diesel_engine": ("modular_diesel_engine/turbocharger", None)},
}


def _clockwork_blades(state, data):
    """VS Clockwork propeller blades on a blade controller (BladeControllerRenderer.renderShared /
    renderBlade, Clockwork 0.5). Blade i (an item in "Blades") is turned to slot i * 360 / count
    about the controller's axis, pitched by "BladeAngle" about the block's z axis, and built from a
    base, an extension stretched to "BladeLength" and a tip. Per part, applied last-to-first:
      facing:  rotateCentered(horizontalAngle(facing.opposite), up)  (horizontal facings)
               rotateCentered(-90 - verticalAngle(facing), east)
      slot:    rotateCentered(i * 360 / count, up)
      pitch:   rotate about z by BladeAngle through the block centre
      part:    base: z - 0.3;  extension: z * L, then z + (L - 1) * 0.3 - 0.3;  tip: z + (L-1)*0.5 + 1 - L - 0.3
    Returns (model id, matrix, shift) in this module's data_parts form."""
    import math
    import numpy as np
    data = data or {}
    items = [b for b in (data.get("Blades") or {}).get("Items", []) if b.get("id")]
    if not items:
        return []
    facing = state.properties.get("facing", "north")
    pitch = float(data.get("BladeAngle", 0.0))
    step = 360.0 / len(items)

    def rot(axis, deg):
        t = math.radians(deg)
        c, s = math.cos(t), math.sin(t)
        r = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
             "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
        m = np.eye(4)
        m[:3, :3] = r
        return m

    def tr(x, y, z):
        m = np.eye(4)
        m[:3, 3] = (x, y, z)
        return m

    def centered(axis, deg):
        return tr(.5, .5, .5) @ rot(axis, deg) @ tr(-.5, -.5, -.5)
    opposite = {"north": "south", "south": "north", "east": "west", "west": "east", "up": "down", "down": "up"}
    y_rot = {"south": 0, "west": 90, "north": 180, "east": 270}  # Direction.toYRot
    head = np.eye(4)
    if facing not in ("up", "down"):
        o = opposite[facing]
        h = -y_rot[o] if o in ("east", "west") else y_rot[o]  # AngleHelper.horizontalAngle
        head = head @ centered("y", h)
    vertical = {"up": -90, "down": 90}.get(facing, 0)  # AngleHelper.verticalAngle
    head = head @ centered("x", -90 - vertical)
    out = []
    for i, item in enumerate(items):
        length = float((item.get("tag") or {}).get("BladeLength", 1.0))
        wide = "wide" in str(item.get("id"))
        name = "wideblade" if wide else "blade"
        m = head @ centered("y", i * step) @ centered("z", pitch)
        scale = np.eye(4)
        scale[2, 2] = length
        parts = {f"{name}_base": tr(0, 0, -0.3),
                 f"{name}_extension": tr(0, 0, (length - 1) * 0.3) @ scale @ tr(0, 0, -0.3),
                 f"{name}_tip": tr(0, 0, (length - 1) * 0.5 + 1 - length - 0.3)}
        for part, local in parts.items():
            a = m @ local  # block units, origin at the block corner
            lin = a[:3, :3]
            c = np.full(3, 8.0)
            shift = 16 * a[:3, 3] + lin @ c - c  # data_parts turn about the block centre (1/16 units)
            out.append((f"vs_clockwork:block/blade_controller/blade/{part}", lin, shift))
    return out


def _headlamps(state, data):
    """Bits 'n' Bobs headlamps (HeadlampBlockEntityRenderer / HeadlampModelBuilder): up to nine
    lamps on one block in a 3x3 grid. headlampPlacements[i] is 0 for an empty slot, else a lamp
    (2+ = dyed); slot i = vertical * 3 + horizontal, each alignment -4, 0 or +4 pixels. Lamps are
    authored facing up, moved to their slot, then turned up -> facing and spun -90 degrees per
    the facing's 2D index."""
    import math
    import numpy as np
    slots = list((data or {}).get("headlampPlacements") or [])
    if not any(slots):
        slots = [0, 0, 0, 0, 1, 0, 0, 0, 0]  # no data: one lamp in the middle
    p = state.properties
    on = p.get("forced_on") == "true" or str(p.get("power", "0")) not in ("0", "")
    mid = "bits_n_bobs:block/headlamp/headlight_on" if on else "bits_n_bobs:block/headlamp/headlight"
    facing = p.get("facing", "up")
    dirs = {"up": (0, 1, 0), "down": (0, -1, 0), "north": (0, 0, -1), "south": (0, 0, 1),
            "east": (1, 0, 0), "west": (-1, 0, 0)}
    a, b = np.array(dirs["up"], float), np.array(dirs[facing], float)
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-9:  # parallel: none for up, half turn about x for down
        to = np.eye(3) if facing == "up" else np.diag([1.0, -1, -1])
    else:
        k = axis / np.linalg.norm(axis)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        ang = math.acos(max(-1.0, min(1.0, float(a @ b))))
        to = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
    if facing != "up":
        d2 = {"south": 0, "west": 1, "north": 2, "east": 3}.get(facing, -1)
        t = math.radians(d2 * -90 if d2 != -1 else 0)
        c, s_ = math.cos(t), math.sin(t)
        to = to @ np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
    offs = (-4, 0, 4)  # RIGHT_OR_BOTTOM, MIDDLE, LEFT_OR_TOP
    out = []
    for i, v in enumerate(slots[:9]):
        if not v:
            continue
        local = np.array([offs[i % 3], 0.0, offs[i // 3]])  # (horizontal, 0, vertical) pixels
        out.append((mid, np.round(to, 9), np.round(to @ local, 9)))
    return out


def _belt(state):
    """Create BeltRenderer: per segment a top and a bottom belt piece (start / middle / end, or
    one diagonal piece on slopes), turned by the facing's horizontal angle (+180 upward, +270
    sideways), tipped 90 about z when sideways and 90 about x when vertical; start and end swap
    on downward and positive vertical belts. Pulley ends add the pulley, pointing clockwise
    of the facing (up when sideways)."""
    import math
    import numpy as np
    p = state.properties
    facing = p.get("facing", "north")
    slope, part = p.get("slope", "horizontal"), p.get("part", "middle")
    down, up, side = slope == "downward", slope == "upward", slope == "sideways"
    diagonal = down or up
    start, end = part == "start", part == "end"
    if down or (slope == "vertical" and facing in ("south", "east")):
        start, end = end, start

    def rot(axis, d):
        c, s_ = math.cos(math.radians(d)), math.sin(math.radians(d))
        return {"x": np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]),
                "y": np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]),
                "z": np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])}[axis]

    horizontal = {"south": 0, "west": -90, "north": 180, "east": 90}[facing]
    m = (rot("y", horizontal + (180 if up else 0) + (270 if side else 0)) @ rot("z", 90 if side else 0)
         @ rot("x", 90 if not diagonal and slope != "horizontal" else 0))
    m = np.round(m, 9)
    kind = "start" if start else "end" if end else "middle"
    if diagonal:
        names = [f"diagonal_{kind}"]
    else:
        names = [kind + "_bottom", kind]
    out = [(f"create:block/belt/{n}", m, np.zeros(3)) for n in names]
    if part in ("start", "end", "pulley"):
        cw = {"north": "east", "east": "south", "south": "west", "west": "north"}[facing]
        out.append(("create:block/belt_pulley", _blockstate_rot(*NORTH_TO_FACING["up" if side else cw]), np.zeros(3)))
    return out


def _blockstate_rot(xr, yr):
    from .voxelize import _blockstate_matrix
    return _blockstate_matrix(xr, yr)


def _valve_pointer(state):
    """Create FluidValveRenderer: the pointer marks, turned by the facing's horizontal angle,
    tipped 90 degrees unless the valve faces up/down, then turned about their own axis: 90
    when the pipe is vertical or the shaft runs along x, and -90 more when the valve is open."""
    import math
    import numpy as np
    p = state.properties
    facing = p.get("facing", "north")
    axis_of = {"north": 2, "south": 2, "east": 0, "west": 0, "up": 1, "down": 1}
    shaft = axis_of[facing]
    others = [a for a in range(3) if a != shaft]
    pipe = others[0] if p.get("axis_along_first") == "true" else others[1]
    offset = 90 if (pipe != 1 and shaft == 0) or pipe == 1 else 0
    turn = offset + (-90 if p.get("enabled") == "true" else 0)

    def ry(d):
        c, s_ = math.cos(math.radians(d)), math.sin(math.radians(d))
        return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])

    def rx(d):
        c, s_ = math.cos(math.radians(d)), math.sin(math.radians(d))
        return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])

    horizontal = {"south": 0, "west": -90, "north": 180, "east": 90}.get(facing, 0)  # AngleHelper.horizontalAngle
    tip = {"up": 0, "down": 180}.get(facing, 90)
    m = ry(horizontal) @ rx(tip) @ ry(turn)
    return [("create:block/fluid_valve/pointer", np.round(m, 9), np.zeros(3))]


def data_parts(state, data):
    """[(model id, rotation matrix, shift)] for parts a renderer draws from block data or
    properties that the cached per-state model can't know about (diesel engine upgrades)."""
    import numpy as np
    from .voxelize import _blockstate_matrix
    if state.name == "create:fluid_valve":
        return _valve_pointer(state)
    if state.name == "bits_n_bobs:headlamp":
        return _headlamps(state, data)
    if state.name == "create:belt":
        return _belt(state)
    if state.name == "vs_clockwork:blade_controller":
        return _clockwork_blades(state, data)
    if state.name not in _ENGINE_UPGRADES["silencer"]:
        return []
    p = state.properties
    upgrade = str((data or {}).get("Upgrade", "")).rpartition(":")[2]
    kinds = [k for k in _ENGINE_UPGRADES if upgrade == k or p.get("silenced" if k == "silencer" else "turbocharged") == "true"]
    facing = p.get("facing", "north")
    out = []
    for kind in kinds:
        horizontal, vertical = _ENGINE_UPGRADES[kind][state.name]
        if facing in ("up", "down"):
            if vertical is None:
                continue
            mid, yaw = vertical, 90 if facing == "up" else 180  # rotateY(270 or 180), counter-clockwise
        else:
            # rotateY(facing.toYRot()), counter-clockwise; as a clockwise blockstate yaw:
            mid, yaw = horizontal, {"south": 0, "west": 270, "north": 180, "east": 90}[facing]
        out.append((_CDG + mid, _blockstate_matrix(0, yaw), np.zeros(3)))
    return out


def offset_parts(state, yaw, xrot=0):
    """[(model id, rotation matrix, shift in 1/16 units)] for OFFSET_PARTS of a block."""
    import numpy as np
    from .voxelize import _blockstate_matrix, _rot_matrix
    if state.name == "createdieselgenerators:huge_diesel_engine":
        return _diesel_parts(state, xrot, yaw)
    out = []
    flipped = state.properties.get("vertical_direction") == "up"
    m = _blockstate_matrix(0, yaw)
    if flipped:  # upside down: half-turn about the block's z axis first
        m = m @ _rot_matrix("z", 180)
    for mid, dy in OFFSET_PARTS.get(state.name, []):
        out.append((mid, m, np.array([0.0, (-dy if flipped else dy) * 16.0, 0.0])))
    return out


def extra_by_model(model_id, assets):
    """Part models (with any offset already applied, in the model's own frame) for a placed model."""
    out = []
    for entry in EXTRA_BY_MODEL.get(model_id, []):
        mid, off = (entry, None) if isinstance(entry, str) else entry
        part = assets.model(mid)
        if part is None:
            continue
        if off:
            moved = []
            for el in part["elements"]:
                el = dict(el, **{"from": [a + b for a, b in zip(el["from"], off)],
                                 "to": [a + b for a, b in zip(el["to"], off)]})
                if el.get("rotation") and "origin" in el["rotation"]:
                    el["rotation"] = dict(el["rotation"], origin=[a + b for a, b in zip(el["rotation"]["origin"], off)])
                moved.append(el)
            part = dict(part, elements=moved)
        out.append(part)
    return out


# Parts authored in a north-facing frame (Create's partialFacing): they are turned to the
# block's `facing` rather than taking the blockstate's own model rotation.
NORTH_FRAME_MODELS = {
    "create:hand_crank": ["create:block/hand_crank/handle"],
    "create:mechanical_pump": ["create:block/mechanical_pump/cog"],
    # EncasedFanRenderer: blades and the half shaft face away from the fan's front
    "create:encased_fan": [("create:block/encased_fan/propeller", "opposite"), ("create:block/shaft_half", "opposite")],
    "create_connected:crank_wheel": ["create_connected:block/crank_wheel/handle"],
    "create_connected:large_crank_wheel": ["create_connected:block/large_crank_wheel/handle"],
}

_OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east", "up": "down", "down": "up"}


def north_frame_parts(state):
    """[(model id, (x, y) rotation)] for NORTH_FRAME_MODELS of a block."""
    facing = state.properties.get("facing", "north")
    out = []
    for entry in NORTH_FRAME_MODELS.get(state.name, []):
        mid, mode = (entry, None) if isinstance(entry, str) else entry
        out.append((mid, NORTH_TO_FACING[_OPPOSITE[facing] if mode == "opposite" else facing]))
    return out


# Rotation (x, y) that turns a north-facing part to each facing.
NORTH_TO_FACING = {"north": (0, 0), "east": (0, 90), "south": (0, 180), "west": (0, 270),
                   "up": (270, 0), "down": (90, 0)}

# Blocks whose item model shows their moving parts in the placed model's frame even though
# it redraws some casing pieces (checked by rendering): keep the item's extra elements.
ITEM_EXTRAS_ALLOWED = ("create:speedometer", "create:stressometer", "simulated:throttle_lever")

# Propellers (Aeronautics, Create Submarine): the blades are their own part model, drawn in the
# placed model's frame (pointing up); "reversed" picks the reversed blades.
PROPELLERS = ("aeronautics:*_propeller", "create_submarine:submarine_propeller")


def state_extra_models(state):
    """Extra part models chosen from the block's properties (placed model's rotation)."""
    if any(_matches(state.name, p) for p in PROPELLERS):
        ns, _, path = state.name.partition(":")
        rev = "_reversed" if state.properties.get("reversed") == "true" else ""
        return [f"{ns}:block/{path}/propeller{rev}"]
    return []


def _matches(name, pattern):
    ns, _, path = name.partition(":")
    pns, _, ppath = pattern.partition(":")
    if pns != ns:
        return False
    return path.endswith(ppath[1:]) if ppath.startswith("*") else path == ppath


def item_extras_allowed(name):
    return any(_matches(name, p) for p in ITEM_EXTRAS_ALLOWED)


# EXTRA_MODELS whose parts are authored facing south and turned to the block's facing by the
# renderer (Simulated's PortableEngineRenderer.rotateToFacing uses Create's horizontalAngle),
# rather than taking the placed model's own rotation.
SOUTH_FRAME_EXTRAS = ("simulated:*_portable_engine",)


def extras_frame_yaw(name, facing):
    """Clockwise yaw for EXTRA_MODELS of `name`, or None to use the placed model's rotation."""
    if any(_matches(name, p) for p in SOUTH_FRAME_EXTRAS):
        return {"south": 0, "west": 90, "north": 180, "east": 270}.get(facing, 0)
    return None


def extra_models(name):
    if name in EXTRA_MODELS:
        return EXTRA_MODELS[name]
    ns, _, path = name.partition(":")
    for key, models in EXTRA_MODELS.items():
        kns, _, kpath = key.partition(":")
        if kns == ns and kpath.startswith("*") and path.endswith(kpath[1:]):
            return models
    return []


# Blockstate x/y rotation that turns an "up"-facing model to face each direction.
FACING_ROTATION = {"up": (0, 0), "down": (180, 0), "north": (90, 0), "east": (90, 90),
                   "south": (90, 180), "west": (90, 270)}

# Blocks shaped exactly like another block that has model files.
ALIASES = {
    "create_connected:copycat_stairs": "minecraft:oak_stairs",
    "create_connected:copycat_fence": "minecraft:oak_fence",
    "create_connected:copycat_fence_gate": "minecraft:oak_fence_gate",
    "create_connected:copycat_wall": "minecraft:cobblestone_wall",
    "copycats:copycat_stairs": "minecraft:oak_stairs",
    "copycats:copycat_trapdoor": "minecraft:oak_trapdoor",
    "copycats:copycat_fence": "minecraft:oak_fence",
    "copycats:copycat_wall": "minecraft:cobblestone_wall",
    "copycats:copycat_door": "minecraft:oak_door",
    "copycats:copycat_folding_door": "minecraft:oak_door",
    "copycats:copycat_pane": "minecraft:glass_pane",
    "copycats:copycat_button": "minecraft:oak_button",
    "copycats:copycat_pressure_plate": "minecraft:oak_pressure_plate",
    "dndecor:black_industrial_cogwheel": "create:cogwheel",
    "dndecor:industrial_cogwheel": "create:cogwheel",
    "dndecor:dark_metal_cogwheel": "create:cogwheel",
    "copycats:copycat_shaft": "create:shaft",
    "copycats:copycat_cogwheel": "create:cogwheel",
    "copycats:copycat_large_cogwheel": "create:large_cogwheel",
}


# --- Parts drawn between two blocks (need the whole structure) ---
_DIR = {"up": (0, 1, 0), "down": (0, -1, 0), "north": (0, 0, -1), "south": (0, 0, 1),
        "east": (1, 0, 0), "west": (-1, 0, 0)}
_HYD = "aeronautics_utility_objects:block/"


def _rotate_to(a, b):
    """Rotation matrix turning unit vector a onto b (shortest arc)."""
    import math
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-9:
        if a @ b > 0:
            return np.eye(3)
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 0, 1.0])
        k = np.cross(a, perp)
        k /= np.linalg.norm(k)
        return 2 * np.outer(k, k) - np.eye(3)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.acos(max(-1.0, min(1.0, float(a @ b))))
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def structure_parts(structure):
    """[(model id, matrix, translation)] in structure 1/16 units (x -> matrix @ x + translation)
    for parts that span blocks. Hydraulic pistons (Aeronautics Utility Objects): the head with
    RenderOwner draws the rod to the head facing it, a sleeve on each end (approximation of
    HydraulicConnectionHeadRenderer at rest: the rod model stretched between the two head tips)."""
    import numpy as np
    heads = {(x, y, z): st for x, y, z, st in structure.blocks
             if st.name == "aeronautics_utility_objects:hydraulic_connection_head"}
    out = []
    for pos, st in heads.items():
        data = structure.block_data.get(pos) or {}
        if not data.get("RenderOwner"):
            continue
        d = np.array(_DIR[st.properties.get("facing", "up")], float)
        other = None
        for step in range(1, 33):
            q = tuple(int(v) for v in np.array(pos) + d * step)
            if q in heads:
                if np.allclose(_DIR[heads[q].properties.get("facing", "up")], -d):
                    other = q
                break
        if other is None:
            continue
        tip = 0.75  # head top: 8.75 px from the base, 0.75 past the cell centre
        a = np.array(pos, float) * 16 + 8 + d * tip
        b = np.array(other, float) * 16 + 8 - d * tip
        length = np.linalg.norm(b - a)
        R = _rotate_to((0, 1, 0), d)
        S = np.diag([1.0, length / 22.0, 1.0])  # rod model runs y = 1..23 around x = z = 0
        m = R @ S
        out.append((_HYD + "hydraulic_rod", m, a - m @ np.array([0, 1.0, 0])))
        out.append((_HYD + "hydraulic_hinge_sleeve", R, a))
        out.append((_HYD + "hydraulic_hinge_sleeve", _rotate_to((0, 1, 0), -d), b))
    out += _propeller_cluster_parts(structure)
    out += _thruster_multiblock_parts(structure)
    return out


def hidden_blocks(structure):
    """Positions whose own model isn't drawn because a multiblock draws them as a whole
    (Create Deep Seas propeller clusters: SubmarinePropellerClusterModel hides the members)."""
    return set().union(*[members for _, members in propeller_clusters(structure).values()])


# --- Create Deep Seas (create_submarine) multi-block propellers ---
# SubmarinePropellerBlockEntity: an isolated, completely filled 3x3 (tried first) or 2x2 square of
# propellers with the same facing and "reversed" turns into one big propeller. The square lies
# across planeAxes(facing); the ring of blocks around it in that plane must hold no matching
# propeller. The master (the 2x2's corner nearest -axes, or the 3x3's centre) draws it all.
SUB_PROPELLER = "create_submarine:submarine_propeller"
_PLANE_AXES = {"x": ((0, 1, 0), (0, 0, 1)), "y": ((1, 0, 0), (0, 0, 1)), "z": ((1, 0, 0), (0, 1, 0))}
_AXIS_OF = {"up": "y", "down": "y", "north": "z", "south": "z", "east": "x", "west": "x"}
_SUB_PROP = "create_submarine:block/submarine_propeller/"


def _direction_rotation(facing):
    """Minecraft's Direction.getRotation(): turns +y to face `facing` (right-handed matrices)."""
    from .voxelize import _rot_matrix
    rx, rz = (lambda d: _rot_matrix("x", d)), (lambda d: _rot_matrix("z", d))
    return {"up": lambda: rx(0), "down": lambda: rx(180), "south": lambda: rx(90),
            "north": lambda: rx(90) @ rz(180), "west": lambda: rx(90) @ rz(90),
            "east": lambda: rx(90) @ rz(-90)}[facing]()


def propeller_clusters(structure):
    """{master pos: (size, frozenset of member positions)} for Deep Seas propeller clusters."""
    cache = getattr(structure, "_m2m_prop_clusters", None)
    if cache is not None and cache[0] == len(structure.blocks):
        return cache[1]
    key = {(x, y, z): (st.properties.get("facing", "up"), st.properties.get("reversed", "false"))
           for x, y, z, st in structure.blocks if st.name == SUB_PROPELLER}

    def isolated(corner, axes, k, n):
        a0, a1 = axes
        for i in range(-1, n + 1):
            for j in range(-1, n + 1):
                inside = 0 <= i < n and 0 <= j < n
                q = tuple(corner[c] + a0[c] * i + a1[c] * j for c in range(3))
                if (key.get(q) == k) != inside:
                    return False
        return True

    found = {}
    for pos, k in key.items():
        axes = _PLANE_AXES[_AXIS_OF.get(k[0], "y")]
        a0, a1 = axes
        for n in (3, 2):
            master = None
            for i in range(n):
                for j in range(n):
                    corner = tuple(pos[c] - a0[c] * i - a1[c] * j for c in range(3))
                    if isolated(corner, axes, k, n):
                        master = corner if n == 2 else tuple(corner[c] + a0[c] + a1[c] for c in range(3))
                        break
                if master:
                    break
            if master:
                found.setdefault(master, (n, set()))[1].add(pos)
                break
    clusters = {m: (n, frozenset(ms)) for m, (n, ms) in found.items() if len(ms) == n * n}
    structure._m2m_prop_clusters = (len(structure.blocks), clusters)
    return clusters


def _propeller_cluster_parts(structure):
    """SubmarinePropellerRenderer.renderCluster at rest (blade angle 0)."""
    import numpy as np
    out = []
    states = {(x, y, z): st for x, y, z, st in structure.blocks}
    for master, (n, _) in propeller_clusters(structure).items():
        facing = states[master].properties.get("facing", "up")
        a0, a1 = (np.array(a, float) for a in _PLANE_AXES[_AXIS_OF.get(facing, "y")])
        centre = np.array(master, float) * 16 + 8 + (8 * (a0 + a1) if n == 2 else 0)
        rf = _direction_rotation(facing)
        s = 16.0 if n == 2 else 8.0
        out.append((f"{_SUB_PROP}{n}x{n}_base", rf, centre + rf @ np.array([-s, -8, -s])))
        for blades in (["2x2_propeller"] if n == 2 else ["3x3_gear", "3x3_propeller"]):
            out.append((_SUB_PROP + blades, rf, centre + rf @ np.array([0, -8.0, 0])))
    return out


# --- Create Propulsion: Simulated multi-block thrusters ---
# A 2x2x2 or 3x3x3 cube of thrusters (multi=true; their own models are empty) is drawn by its
# controller, the cube's lowest corner (the others store ControllerOffX/Y/Z to it), as the
# partial model scaled by Width and turned like the block (ThrusterRenderer.applyFacingRotation).
THRUSTERS = ("createpropulsion:thruster", "createpropulsion:creative_thruster")


def empty_by_design(state):
    """Blocks whose model is empty on purpose: another block draws them (multiblock thrusters)."""
    return state.name in THRUSTERS and state.properties.get("multi") == "true"


def thruster_multiblocks(structure):
    """[(controller pos, width, state)] of multiblock thrusters."""
    data = getattr(structure, "block_data", None) or {}
    multi = {(x, y, z): st for x, y, z, st in structure.blocks
             if st.name in THRUSTERS and st.properties.get("multi") == "true"}
    out, seen = [], set()
    for pos, st in multi.items():  # saved block entity data says which block is the controller
        d = data.get(pos) or {}
        w = int(d.get("Width", 0) or 0)
        if w in (2, 3) and "ControllerOffX" not in d:
            out.append((pos, w, st))
            seen.update(_cube(pos, w))
    for pos in sorted(multi):  # no data: a full same-facing cube from its lowest corner
        if pos in seen:
            continue
        for w in (3, 2):
            cube = _cube(pos, w)
            if all(q in multi and q not in seen and multi[q] == multi[pos] for q in cube):
                out.append((pos, w, multi[pos]))
                seen.update(cube)
                break
    return out


def _cube(pos, w):
    return [(pos[0] + i, pos[1] + j, pos[2] + k) for i in range(w) for j in range(w) for k in range(w)]


def _thruster_multiblock_parts(structure):
    import numpy as np
    from .voxelize import _blockstate_matrix
    out = []
    for pos, w, st in thruster_multiblocks(structure):
        ns, _, path = st.name.partition(":")
        r = _blockstate_matrix(*NORTH_TO_FACING[st.properties.get("facing", "north")])
        h = np.full(3, 8.0 * w)  # turned about the cube's centre, after scaling by the width
        out.append((f"{ns}:partial/{path}_multiblock_{w}x{w}x{w}", r * w, np.array(pos, float) * 16 + h - r @ h))
    return out
