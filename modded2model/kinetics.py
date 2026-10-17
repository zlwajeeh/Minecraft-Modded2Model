"""Create kinetic networks: which blocks turn, how fast, and which way (for rigged GLB exports).

Speeds follow Create 6 (RotationPropagator.getRotationSpeedModifier): `to` turns at
`from * modifier`, where the modifier is
  1        shafts joined end to end (both have a shaft toward each other); a gearbox turns its
           sides by getAxisModifier (relative to the side its rotation comes from);
  -1       two small cogs side by side on the same axis (the step isn't along that axis);
  -2/-0.5  large cog to a small cog on the same axis, one block diagonally (and back);
  +-1      two large cogs at right angles, one block apart diagonally;
  1        joined chain drives (an adjustable chain gearshift, when powered, 2 / 0.5);
  0        anything else (not connected).
Saved speeds (every kinetic block entity stores "Speed", in RPM) are used when the build was
running when saved; otherwise each network's speeds are worked out from the rules, starting at
a source (motor, engine, crank...) turning at `base_rpm`.

Rendering (KineticBlockEntityRenderer): a block turns 6 * RPM degrees per second about its axis,
starting at Create's rotation offset (22.5 degrees where the coordinates off the axis add up to
an even number, else 11.25 for large cogs) so neighbouring cogs mesh.
"""
import math
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction

from .codeshapes import ALIASES, KINETIC_SHAFTS

DIRS = {"east": (1, 0, 0), "west": (-1, 0, 0), "up": (0, 1, 0), "down": (0, -1, 0),
        "south": (0, 0, 1), "north": (0, 0, -1)}
AXIS_OF = {"east": "x", "west": "x", "up": "y", "down": "y", "south": "z", "north": "z"}
POSITIVE = ("east", "up", "south")
OPP = {"east": "west", "west": "east", "up": "down", "down": "up", "south": "north", "north": "south"}

SMALL_COGS = ("create:cogwheel",)
LARGE_COGS = ("create:large_cogwheel",)
SHAFTS = ("create:shaft",)
# rotation sources: where a stopped network is assumed to be driven from
SOURCES = ("creative_motor", "water_wheel", "windmill_bearing", "hand_crank", "steam_engine",
           "powered_engine_shaft", "diesel_engine", "flywheel", "motor", "generator")
# blocks whose shaft comes out of their back (the side opposite `facing`)
BACK_SHAFT = ("bearing", "hand_crank", "encased_fan", "drill", "saw", "deployer", "mechanical_press",
              "mechanical_mixer", "millstone", "crushing_wheel", "propeller")


def _base(name):
    return ALIASES.get(name, name)


@dataclass
class Node:
    pos: tuple
    name: str
    axis: str  # rotation axis
    kind: str  # small_cog, large_cog, shaft, gearbox, chain, part
    props: dict = field(default_factory=dict)
    saved: float = 0.0  # saved Speed (RPM)
    source: str = None  # gearbox: side its rotation comes from

    def shaft_towards(self, d):
        """IRotate.hasShaftTowards."""
        if self.kind == "gearbox":
            return AXIS_OF[d] != self.axis
        if self.kind in ("shaft", "small_cog", "large_cog", "chain"):
            return AXIS_OF[d] == self.axis
        facing = self.props.get("facing")
        if facing in DIRS:
            if "creative_motor" in self.name or "motor" in self.name:
                return d == facing
            if any(k in self.name for k in BACK_SHAFT):
                return d == OPP[facing]
        return AXIS_OF[d] == self.axis


def classify(pos, state, data):
    """A Node for a kinetic block (one with a saved Speed, or a known kinetic shape), else None."""
    name, p = _base(state.name), state.properties
    data = data or {}
    if "Speed" not in data and name not in SMALL_COGS + LARGE_COGS + SHAFTS and name not in KINETIC_SHAFTS:
        return None
    axis = p.get("axis")
    if axis not in ("x", "y", "z"):
        axis = AXIS_OF.get(p.get("facing", ""), "y")
    if name in SMALL_COGS or (name.endswith("cogwheel") and "large" not in name and "encased" not in name):
        kind = "small_cog"
    elif name in LARGE_COGS or (name.endswith("large_cogwheel") and "encased" not in name):
        kind = "large_cog"
    elif KINETIC_SHAFTS.get(name) in ("gearbox", "six"):
        kind = "gearbox"
    elif "chain" in name:
        kind = "chain"
    elif name in SHAFTS or name.endswith("_shaft") or KINETIC_SHAFTS.get(name) == "axis":
        kind = "shaft"
    else:
        kind = "part"
    return Node(pos, name, axis, kind, dict(p), float(data.get("Speed", 0.0) or 0.0))


def _gear_axis_modifier(node, d):
    """getAxisModifier for a gearbox with its rotation coming from side `node.source`."""
    s = node.source
    if node.kind != "gearbox" or s is None:
        return 1.0
    if AXIS_OF[d] == AXIS_OF[s]:
        if "parallel_gearbox" in node.name:  # Create Connected: reversed at the input side
            return -1.0 if d == s else 1.0
        return 1.0 if d == s else -1.0
    same = -1.0 if (d in POSITIVE) == (s in POSITIVE) else 1.0
    return -same if "six_way_gearbox" in node.name else same  # Create Connected: right angles reversed


def modifier(a, b):
    """RotationPropagator.getRotationSpeedModifier(a -> b) for the two Nodes (0 = not connected)."""
    diff = tuple(q - p for p, q in zip(a.pos, b.pos))
    nz = [i for i in range(3) if diff[i]]
    aligned = len(nz) == 1 and abs(diff[nz[0]]) == 1
    d = next((k for k, v in DIRS.items() if v == diff), None) if aligned else None
    if d is not None and a.shaft_towards(d) and b.shaft_towards(OPP[d]):
        if b.kind == "gearbox" and b.source is None:
            b.source = OPP[d]  # its rotation comes in from a
        out = _gear_axis_modifier(a, d)
        back = _gear_axis_modifier(b, OPP[d])
        return out / back if back else 0.0
    if a.kind == "chain" and b.kind == "chain" and d is not None and a.axis == b.axis and AXIS_OF[d] != a.axis:
        ratio = 1.0
        if "gearshift" in a.name and a.props.get("powered") == "true":
            ratio = 2.0
        elif "gearshift" in b.name and b.props.get("powered") == "true":
            ratio = 0.5
        return ratio
    ax = "xyz"
    if a.kind == "large_cog" and b.kind == "large_cog" and a.axis != b.axis:
        ok = all((diff[i] != 0) == (ax[i] in (a.axis, b.axis)) for i in range(3))
        if ok:
            return -1.0 if (diff[ax.index(a.axis)] > 0) != (diff[ax.index(b.axis)] > 0) else 1.0
    if a.axis == b.axis and diff[ax.index(a.axis)] == 0 and all(abs(diff[i]) == 1 for i in range(3)
                                                                 if ax[i] != a.axis):
        if a.kind == "large_cog" and b.kind == "small_cog":
            return -2.0
        if a.kind == "small_cog" and b.kind == "large_cog":
            return -0.5
    if a.kind == "small_cog" and b.kind == "small_cog" and d is not None:
        if AXIS_OF[d] != a.axis and a.axis == b.axis:
            return -1.0
    return 0.0


_NEIGHBOURS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if 0 < abs(dx) + abs(dy) + abs(dz) <= 2]


def networks(structure, base_rpm=32.0):
    """{pos: (Node, rpm)} for every kinetic block, and a list of networks [[pos, ...], ...]."""
    nodes = {}
    for x, y, z, st in structure.blocks:
        n = classify((x, y, z), st, structure.block_data.get((x, y, z)))
        if n is not None:
            nodes[(x, y, z)] = n
    speed, groups = {}, []
    for start in nodes:
        if start in speed:
            continue
        # gather the network (connected by any non-zero modifier), resetting gearbox sources
        comp, todo = [start], deque([start])
        seen = {start}
        while todo:
            p = todo.popleft()
            for dx, dy, dz in _NEIGHBOURS:
                q = (p[0] + dx, p[1] + dy, p[2] + dz)
                if q in nodes and q not in seen:
                    a, b = nodes[p], nodes[q]
                    ga, gb = a.source, b.source
                    m = modifier(a, b) or modifier(b, a)
                    a.source, b.source = ga, gb
                    if m:
                        seen.add(q)
                        comp.append(q)
                        todo.append(q)
        for p in comp:
            nodes[p].source = None
        running = [p for p in comp if abs(nodes[p].saved) > 1e-6]
        if running:  # saved while running: Create's own speeds
            for p in comp:
                speed[p] = nodes[p].saved
            _sources_from_saved(nodes, comp)
        else:  # stopped: work the speeds out from a source
            root = next((p for p in comp if any(k in nodes[p].name for k in SOURCES)), comp[0])
            speed[root] = base_rpm
            todo = deque([root])
            while todo:
                p = todo.popleft()
                for dx, dy, dz in _NEIGHBOURS:
                    q = (p[0] + dx, p[1] + dy, p[2] + dz)
                    if q in nodes and q not in speed:
                        m = modifier(nodes[p], nodes[q])
                        if m:
                            speed[q] = speed[p] * m
                            todo.append(q)
            for p in comp:
                speed.setdefault(p, 0.0)
        groups.append(comp)
    return {p: (nodes[p], speed.get(p, 0.0)) for p in nodes}, groups


def _sources_from_saved(nodes, comp):
    """Gearboxes in a running network: their input side is a neighbour whose speed matches."""
    for p in comp:
        n = nodes[p]
        if n.kind != "gearbox":
            continue
        for d, v in DIRS.items():
            q = (p[0] + v[0], p[1] + v[1], p[2] + v[2])
            if q in nodes and AXIS_OF[d] != n.axis and nodes[q].shaft_towards(OPP[d]) \
                    and abs(abs(nodes[q].saved) - abs(n.saved)) < 1e-3:
                n.source = d
                break


def half_speed(node, rpm, side):
    """Speed a gearbox's (or other block's) half shaft on `side` shows (GearboxRenderer)."""
    return rpm * _gear_axis_modifier(node, side)


def phase(node):
    """KineticBlockEntityVisual.rotationOffset: starting angle so neighbouring cogs mesh."""
    off = [c for i, c in enumerate(node.pos) if "xyz"[i] != node.axis]
    if sum(off) % 2 == 0:
        return 22.5
    return 11.25 if node.kind == "large_cog" else 0.0


@dataclass
class KineticPart:
    """One turning piece for a rigged GLB (same fields write_glb reads from contraption parts)."""
    offset: tuple  # (0, 0, 0): its faces are already in export coordinates
    center: tuple  # the block's centre (blocks)
    matrix: object  # rest turn (Create's rotation offset) about the axis
    axis: str
    angle: float  # rest angle, degrees
    kind: str  # block id
    type: str  # "kinetic"
    rpm: float  # signed speed of this piece
    network: int
    label: str  # e.g. "create:gearbox (north half)"


def whole_block(node):
    """Does the block's whole model turn (shafts, cogs), rather than parts drawn on it?"""
    return node.kind in ("small_cog", "large_cog") or (node.kind == "shaft" and node.name not in KINETIC_SHAFTS)


def loop_seconds(speeds, limit=60.0):
    """Shortest time after which every speed has made whole turns (6 * rpm degrees a second),
    so the animation loops without a jump; `limit` if that would be longer."""
    fr = [Fraction(abs(s)).limit_denominator(64) for s in speeds if abs(s) > 1e-6]
    if not fr:
        return 0.0
    num = 0
    den = 1
    for f in fr:
        num = math.gcd(num, f.numerator)
        den = den * f.denominator // math.gcd(den, f.denominator)
    g = Fraction(num, den)  # gcd of all speeds
    t = 60 / g
    return float(min(t, limit))
