"""Snap-on rolling wheels for 3D printing (Offroad wheel mounts, as on Aeronautics cars).

Instead of printing each tire fused to the car, the car gets a round peg axle on every wheel
mount and each wheel prints on its own with a bore through its hub, lying flat beside the model.
Push a wheel over its peg's barbed head and it snaps on and spins, like a Lego wheel.

Along each wheel's axle (from the mount outwards):
  shaft    radius PEG, from inside the wheel mount to just past the wheel's outer face
  collar   a stop just inside the wheel's inner face, so the wheel doesn't rub the car
  head     a barb past the outer face: CLEARANCE + lip wider than the bore at its base (a flat
           shoulder that holds the wheel), tapering outwards so the wheel can be pushed over it
  slot     an upright slit from the tip back to the collar, splitting the head and shaft into
           two prongs that squeeze together as the wheel goes over the head, then spring back.
           Hard plastic can't stretch a wheel over a solid barb; pegs too thin to split get a
           much smaller lip instead.
The wheel's bore is PEG + CLEARANCE in radius. Sizes are in millimetres, so they print the
same at any scale; the peg grows with the wheel.
"""
import numpy as np

from .attachments import FACING_Y, wheel_layout, _apply
from .voxelize import _blockstate_matrix

CLEARANCE = 0.2  # mm, gap between peg and bore, and between wheel and collar/head
LIP = 0.3  # mm, how far the snap head sticks out past the bore (split pegs)
SOLID_LIP = 0.12  # mm, the same for pegs too thin to split: the wheel has to stretch over it
SPLIT_FROM = 2.2  # mm, pegs at least this thick are split by a slot
HEAD = 1.0  # mm, snap head length
COLLAR = 0.8  # mm, collar ring width (radius past the bore) and length
WHEEL_CELL = 0.08  # mm, voxel size used to cut the bore into the wheel


class Wheel:
    def __init__(self, tris, centre, axis, up, mount_w, pos=None):
        self.pos = pos  # the wheel mount's block position
        self.tris = tris  # tire mesh, structure 1/16-block units
        self.centre, self.axis, self.up = centre, axis, up
        self.side = np.cross(axis, up)  # (up, side, axis) is right-handed
        local = self.to_local(tris.reshape(-1, 3))
        self.radius = float(np.hypot(local[:, 0], local[:, 1]).max())  # as printed
        self.tire_radius = self.radius  # as in game
        self.scale = 1.0  # how much smaller make_room made the wheel (across, not in width)
        self.w0, self.w1 = float(local[:, 2].min()), float(local[:, 2].max())  # the seat on the peg
        self.tire0, self.tire1 = self.w0, self.w1  # the tire as it sits in game
        self.mount_w = mount_w  # where the axle starts, on the axis (inside the mount)
        self.cut = 0.0  # how much make_room trimmed off the inner side
        self.moved = 0.0  # how far make_room moved the seat out

    def to_local(self, p):
        d = np.asarray(p, float) - self.centre
        return np.stack([d @ self.up, d @ self.side, d @ self.axis], -1)

    def to_world(self, p):
        p = np.asarray(p, float)
        return self.centre + p[..., :1] * self.up + p[..., 1:2] * self.side + p[..., 2:3] * self.axis


def find_wheels(structure, assets):
    """Wheels (tire meshes with their axles) of the Offroad wheel mounts in a structure."""
    out = []
    for x, y, z, st in structure.blocks:
        if st.name != "offroad:wheel_mount":
            continue
        data = structure.block_data.get((x, y, z)) or {}
        parts, tire = wheel_layout(st, data)
        tire_id = data.get("CurrentStack", {}).get("id")
        if tire is None or not tire_id:
            continue
        ns, _, name = tire_id.partition(":")
        mesh = assets.obj_mesh(f"{ns}:models/item/{name}/item.obj")
        if mesh is None:
            mesh = assets.obj_mesh(f"{ns}:models/item/{name}/block.obj")
        if mesh is None:
            continue
        shift = np.array([x, y, z], float) * 16
        tris = _apply(tire, mesh) + shift
        face = _blockstate_matrix(0, FACING_Y[st.properties.get("facing", "north")])
        axis = face @ np.array([0.0, 0, -1])  # out along the facing
        pts = tris.reshape(-1, 3)
        up = np.array([0.0, 1, 0])
        side = np.cross(axis, up)
        # wheel centre: middle of the tire's extent across and along the axis
        lo = np.array([(pts @ up).min(), (pts @ side).min(), (pts @ axis).min()])
        hi = np.array([(pts @ up).max(), (pts @ side).max(), (pts @ axis).max()])
        mid = (lo + hi) / 2
        centre = mid[0] * up + mid[1] * side + mid[2] * axis
        mount = shift + 8.0  # the mount block's centre: the axle starts inside it
        w = Wheel(tris, centre, axis, up, float((mount - centre) @ axis), (x, y, z))
        if w.mount_w > 0:  # mount on the far side: flip so the car is at -w
            w = Wheel(tris, centre, -axis, up, -w.mount_w, (x, y, z))
        out.append(w)
    return out


def _frustum(wheel, r0, r1, a, b, n=48):
    """Closed frustum along a wheel's axis from w=a (radius r0) to w=b (radius r1), as world
    triangles with outward normals."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    bot = np.stack([r0 * c, r0 * s, np.full(n, a)], 1)
    top = np.stack([r1 * c, r1 * s, np.full(n, b)], 1)
    j = np.roll(np.arange(n), -1)
    tris = [np.stack([bot, bot[j], top[j]], 1), np.stack([bot, top[j], top], 1)]
    cb, ct = np.array([0.0, 0, a]), np.array([0.0, 0, b])
    tris.append(np.stack([np.broadcast_to(cb, bot.shape), bot[j], bot], 1))
    tris.append(np.stack([np.broadcast_to(ct, top.shape), top, top[j]], 1))
    return wheel.to_world(np.concatenate(tris))


def sizes(wheel, unit_mm):
    """(peg radius, bore radius) in 1/16-block units for a wheel printed at unit_mm per unit."""
    k = 1.0 / unit_mm  # units per mm
    r_mm = wheel.radius / k
    # about a third of the wheel; at least thick enough to split, unless the wheel is tiny
    peg = float(np.clip(0.3 * r_mm, min(SPLIT_FROM / 2, 0.45 * r_mm), 2.4)) * k
    return peg, peg + CLEARANCE * k


def snap(wheel, unit_mm):
    """(lip, slot width) in mm for a wheel's peg; slot 0 = a solid peg."""
    peg, _ = sizes(wheel, unit_mm)
    if 2 * peg * unit_mm >= SPLIT_FROM:
        return LIP, 2 * LIP + 0.1  # the prongs must close by the lip on each side
    return SOLID_LIP, 0.0


def without_hidden_blocks(structure, wheels, vox):
    """The structure without blocks mostly inside a tire (in game the tire hides them, e.g. a
    shaft standing in for a hub); they would stop the printed wheel turning."""
    from .schematic import with_blocks
    keep, dropped = [], []
    for x, y, z, st in structure.blocks:
        shape = vox.block_shape(st)
        cells = np.argwhere(shape)
        hidden = False
        if len(cells) and st.name != "offroad:wheel_mount":
            pts = (cells + 0.5) * (16 / shape.shape[0]) + np.array([x, y, z], float) * 16
            for w in wheels:
                loc = w.to_local(pts)
                inside = ((np.hypot(loc[:, 0], loc[:, 1]) < w.tire_radius) & (loc[:, 2] > w.tire0)
                          & (loc[:, 2] < w.tire1))
                if inside.mean() >= 0.5:
                    hidden = True
                    break
        (dropped if hidden else keep).append((x, y, z, st))
    if not dropped:
        return structure, []
    return with_blocks(structure, keep), dropped


def clear_mount_parts(tris, wheel):
    """A wheel mount's own drawn parts (hub plate, suspension arms, springs; triangles in
    structure units) without those whose middle is inside the tire: in game the tire hides
    them, and here the peg takes their place. Parts that only reach into the tire (the square
    hub plate, the arms) stay; make_room trims the wheel's hidden inner side to clear them."""
    from .exact import _shells
    keep = []
    for shell in (_shells(tris) if len(tris) else []):
        lo, hi = shell.reshape(-1, 3).min(0), shell.reshape(-1, 3).max(0)
        mid = wheel.to_local((lo + hi) / 2)
        inside = np.hypot(mid[0], mid[1]) < wheel.tire_radius and wheel.tire0 < mid[2] < wheel.tire1
        if not inside:
            keep.append(shell)
    return np.concatenate(keep) if keep else np.zeros((0, 3, 3))


MIN_WIDTH = 1.5  # mm, a trimmed wheel stays at least this wide (and at least half its width)
MIN_SCALE = 0.6  # a wheel is never made smaller than this fraction of its size


def _near_points(tris, wheel, reach):
    """Points covering the triangles that come within `reach` of a wheel's axis, about one unit
    (1/16 block) apart, in the wheel's local frame."""
    size = np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=2).max(1)  # longest edge
    mid = wheel.to_local(tris.mean(1))
    near = ((np.hypot(mid[:, 0], mid[:, 1]) < reach + size) & (mid[:, 2] > wheel.mount_w - size)
            & (mid[:, 2] < wheel.tire1 + 4 * reach + size))
    small = near & (size <= 1.5)  # fine meshes (a voxel surface): their corners cover them
    out = [tris[small].reshape(-1, 3)]
    for a, b, c in tris[near & ~small]:
        n = int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)))) + 1
        u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        keep = u + v <= 1
        out.append(a + np.outer(u[keep], b - a) + np.outer(v[keep], c - a))
    return wheel.to_local(np.concatenate(out))


def make_room(wheels, body_tris, unit_mm, shrink=True):
    """Give each wheel room to spin where the car is in its way.

    In game a tire may be tucked into the car's side or clip into the blocks next to it; a
    printed wheel can't. Each wheel is fitted in three ways, mildest first:
      trim    its inner side is cut back to CLEARANCE off the car's side: that part was hidden
              in the car anyway (never below half its width)
      shrink  blocks beside the tire (fenders it clips into) make it smaller across, down to
              MIN_SCALE; wheels that started the same size all shrink alike, so the car stays level
      move    whatever is still in the way moves the seat (w0..w1) out along the peg
    shrink=False keeps every wheel its in-game size: only trim and move (the wheels stick out
    further instead). body_tris: the car's triangles (N, 3, 3) in structure units, without pegs
    or tires."""
    body = np.asarray(body_tris, float)
    if not len(body):
        return
    gap = CLEARANCE / unit_mm
    close = {}
    for w in wheels:
        local = _near_points(body, w, w.radius + gap)
        r = np.hypot(local[:, 0], local[:, 1])
        sel = (r < w.radius + gap) & (local[:, 2] > w.mount_w)
        close[id(w)] = r[sel], local[sel, 2]  # everything along the peg, for the move step
        sel &= local[:, 2] < w.tire1 + gap  # beside the tire as it is in game, for trim and shrink
        r, cw = r[sel], local[sel, 2]
        if not len(r):
            continue
        width = w.tire1 - w.tire0
        max_cut = max(0.0, min(width / 2, width - MIN_WIDTH / unit_mm))

        def fit(cut):  # how big the wheel can stay once trimmed by `cut`
            left = r[cw > w.tire0 + cut - gap]
            return min(1.0, (float(left.min()) - gap) / w.radius) if len(left) else 1.0
        # every trim that just clears one more part of the car; take the smallest that leaves
        # the wheel (nearly) as big as the largest trim would
        cuts = np.unique(np.clip(np.concatenate([[0.0], cw + gap - w.tire0]), 0.0, max_cut))
        if not shrink:  # full size: the trim that leaves the least to move out past, the smallest such
            def still(cut):
                ahead = cw[(r < w.radius + gap) & (cw > w.tire0 + cut - gap)]
                return max(0.0, float(ahead.max()) + gap - (w.tire0 + cut)) if len(ahead) else 0.0
            need = [still(c) for c in cuts]
            w.cut = float(cuts[next(i for i, n in enumerate(need) if n <= min(need) + 1e-6)])
            w.w0 += w.cut
            continue
        fits = [fit(c) for c in cuts]
        if max(fits) >= MIN_SCALE:
            best = next(i for i, f in enumerate(fits) if f >= max(fits) - 0.01)
            w.cut, w.scale = float(cuts[best]), fits[best]
        else:  # too much in the way to shrink: trim what's allowed, and move below
            w.cut = float(cuts[-1])
        w.w0 += w.cut
    for w in wheels:  # the same tires stay the same size
        w.scale = min(o.scale for o in wheels if abs(o.tire_radius - w.tire_radius) < 1e-6)
        w.radius = w.tire_radius * w.scale
        r, cw = close[id(w)]
        cw = cw[r < w.radius + gap]
        for _ in range(16):  # a seat moved out can meet more of the car further along the peg
            near = cw[(cw > w.w0 - gap) & (cw < w.w1 + gap)]
            need = float(near.max()) + gap - w.w0 if len(near) else 0.0
            if need <= 1e-9:
                break
            w.w0 += need
            w.w1 += need
            w.moved += need


def _loft(wheel, stations, half_slot, side, n=48):
    """A closed solid along a wheel's axis through `stations` [(w, radius)]: round, or with
    half_slot > 0 the part of the circle on one side (side = +1 or -1) of a slot that wide,
    cut flat along the slot. World triangles with outward normals."""
    rings = []
    for w, r in stations:
        if half_slot:
            a0 = np.arcsin(min(0.999, half_slot / r))
            t = np.linspace(a0, np.pi - a0, n)
            ring = np.stack([r * np.cos(t), side * r * np.sin(t), np.full(n, w)], 1)
        else:
            t = np.linspace(0, 2 * np.pi, n, endpoint=False)
            ring = np.stack([r * np.cos(t), r * np.sin(t), np.full(n, w)], 1)
        rings.append(ring)
    j = np.roll(np.arange(n), -1)
    tris = []
    for p0, p1 in zip(rings, rings[1:]):  # the sides (and, for a slotted prong, its flat face)
        tris += [np.stack([p0, p1, p1[j]], 1), np.stack([p0, p1[j], p0[j]], 1)]
    c0 = np.broadcast_to(rings[0].mean(0), rings[0].shape)  # the ends, facing opposite ways
    c1 = np.broadcast_to(rings[-1].mean(0), rings[-1].shape)
    tris += [np.stack([c0, rings[0], rings[0][j]], 1), np.stack([c1, rings[-1][j], rings[-1]], 1)]
    tris = np.concatenate(tris)
    # wound so the faces look out
    vol = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum()
    if vol < 0:
        tris = tris[:, ::-1]
    return wheel.to_world(tris)


def peg_tris(wheel, unit_mm):
    """The axle that stays on the car: shaft, collar, snap head and slot (world triangles,
    smooth round shells that overlap where they join, like the rest of an exact export).

    The shaft from inside the wheel mount to just past the collar is one round piece. From
    there to the tip it is two prongs either side of an upright slot (prongs side by side, so
    neither prints hanging over the other), each with its half of the snap head."""
    k = 1.0 / unit_mm
    peg, bore = sizes(wheel, unit_mm)
    lip, slot = snap(wheel, unit_mm)
    gap = CLEARANCE * k
    shaft_end = wheel.w1 + gap  # the head's shoulder: just clear of the wheel's outer face
    tip = shaft_end + HEAD * k
    collar_end = wheel.w0 - gap
    collar_start = max(wheel.mount_w, collar_end - COLLAR * k)
    head_r, tip_r = bore + lip * k, peg * 0.7
    root = min(collar_end + 0.4 * k, shaft_end - 0.4 * k) if slot else shaft_end  # where the prongs start

    out = [_frustum(wheel, peg, peg, wheel.mount_w, root + 0.05 * k)]
    if collar_end > collar_start:
        out.append(_frustum(wheel, bore + COLLAR * k, bore + COLLAR * k, collar_start, collar_end))
    head = [(root, peg), (shaft_end, peg), (shaft_end, head_r), (tip, tip_r)]
    if slot:
        for side in (1, -1):
            out.append(_loft(wheel, head, slot * k / 2, side))
    else:
        out.append(_loft(wheel, head[2:], 0, 1))
    return np.concatenate(out)


SEGMENTS = 120  # sides of a printed wheel's circle


def tire_profile(wheel):
    """The tire's outline as (w, radius) stations along its axis, from the model's own vertices
    (its widest radius at each position along the axle): shoulders and bulges follow the real
    tire, but the wheel is made perfectly round, so it rolls smoothly."""
    pts = wheel.to_local(wheel.tris.reshape(-1, 3))
    r = np.hypot(pts[:, 0], pts[:, 1])
    w = np.round(pts[:, 2], 3)
    stations = sorted(set(w))
    return np.array([(x, float(r[w == x].max())) for x in stations])


def wheel_part(wheel, unit_mm):
    """The wheel on its own: a smooth round solid of revolution following the tire's outline,
    with the bore cut through it. (verts, faces) in the wheel's local frame (axis = z), a
    closed mesh with outward normals."""
    k = 1.0 / unit_mm
    _, bore = sizes(wheel, unit_mm)
    prof = tire_profile(wheel)
    a, b = wheel.tire0 + wheel.cut, wheel.tire1  # the inner side may have been trimmed
    ws = np.concatenate([[a], prof[(prof[:, 0] > a + 1e-6) & (prof[:, 0] < b - 1e-6), 0], [b]])
    outer = np.interp(ws, prof[:, 0], prof[:, 1]) * wheel.scale
    outer = np.maximum(outer, bore + 0.4 * k)  # always some tire around the bore
    n = SEGMENTS
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s_ = np.cos(t), np.sin(t)
    out_rings = [np.stack([r * c, r * s_, np.full(n, w)], 1) for w, r in zip(ws, outer)]
    in_rings = [np.stack([bore * c, bore * s_, np.full(n, w)], 1) for w in ws]
    j = np.roll(np.arange(n), -1)
    tris = []
    for p0, p1 in zip(out_rings, out_rings[1:]):  # tread and sidewalls
        tris += [np.stack([p0, p1, p1[j]], 1), np.stack([p0, p1[j], p0[j]], 1)]
    for p0, p1 in zip(in_rings, in_rings[1:]):  # the bore, facing the axis
        tris += [np.stack([p0, p1[j], p1], 1), np.stack([p0, p0[j], p1[j]], 1)]
    o, i = out_rings[0], in_rings[0]  # the two faces, wound to match the tread and the bore
    tris += [np.stack([o, o[j], i], 1), np.stack([i, o[j], i[j]], 1)]
    o, i = out_rings[-1], in_rings[-1]
    tris += [np.stack([o[j], o, i], 1), np.stack([o[j], i, i[j]], 1)]
    tris = np.concatenate(tris)
    vol = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum()
    if vol < 0:  # wound so the faces look out
        tris = tris[:, ::-1]
    verts = tris.reshape(-1, 3)
    return verts, np.arange(len(verts)).reshape(-1, 3)


def laid_out(parts, floor_y, start_x, start_z, gap):
    """Place wheel parts (local frame, axis = z) flat on the floor in a row starting at
    (start_x, start_z): the axis becomes +y. Returns [(verts, faces)] in structure units."""
    out, x = [], start_x
    for verts, faces in parts:
        lo, hi = verts.min(0), verts.max(0)
        # local (u, v, w) -> (x, y, z) = (u, w, -v): a proper rotation, so faces keep their winding
        placed = np.stack([verts[:, 0] - lo[0] + x, verts[:, 2] - lo[2] + floor_y,
                           -(verts[:, 1] - hi[1]) + start_z], 1)
        out.append((placed, faces))
        x += (hi[0] - lo[0]) + gap
    return out
