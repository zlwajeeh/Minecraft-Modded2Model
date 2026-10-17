"""Geometry that blocks draw outside their own cell from stored block data (block entities).

Example: an Offroad wheel mount stores which tire is fitted and draws the axle and the
tire one block out in front of itself. These parts never appear as blocks in a schematic.
Each handler returns (voxel_tris, smooth_tris), both (N, 3, 3) in the block's 0..16 units
and free to extend past 0..16 into neighbouring cells. voxel_tris are blocky parts merged
into the voxel grid; smooth_tris are closed meshes (like round tires) written to the output
as they are, so they keep their real shape instead of being stair-stepped.
"""
import numpy as np

from .voxelize import _blockstate_matrix, _rot_matrix

FACING_Y = {"north": 0, "east": 90, "south": 180, "west": 270}

# 12 triangles of a unit cube, as corner indices (x, y, z bits)
# wound counter-clockwise seen from outside, so normals point outward
_CUBE_TRIS = [(0, 3, 1), (0, 2, 3), (4, 7, 6), (4, 5, 7), (0, 5, 4), (0, 1, 5),
              (2, 7, 3), (2, 6, 7), (0, 6, 2), (0, 4, 6), (1, 7, 5), (1, 3, 7)]


def element_tris(el):
    # min/max: a box saved with from > to (drawn inside out by Minecraft) is still a closed,
    # outward-facing solid for printing
    f, t = np.array(el["from"], float), np.array(el["to"], float)
    lo, hi = np.minimum(f, t), np.maximum(f, t)
    corners = np.array([[(hi if (i >> b) & 1 else lo)[b] for b in range(3)] for i in range(8)])
    rot = el.get("rotation")
    if rot and rot.get("angle"):
        origin = np.array(rot.get("origin", [8, 8, 8]), float)
        corners = (corners - origin) @ _rot_matrix(rot["axis"], rot["angle"]).T + origin
    return corners[np.array(_CUBE_TRIS)]


def model_tris(assets, model_id):
    m = assets.model(model_id)
    if m is None:
        return np.zeros((0, 3, 3))
    parts = [element_tris(el) for el in m["elements"]]
    if m.get("mesh") is not None:
        parts.append(m["mesh"])
    return np.concatenate(parts) if parts else np.zeros((0, 3, 3))


WHEEL_OUT = 1.375  # blocks from the mount's centre to the tire's centre, along the facing
REST_EXTENSION = 0.5  # WheelMountBlockEntity's initial suspension extension (blocks)


def _T(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _RX(rad):
    c, s = np.cos(rad), np.sin(rad)
    m = np.eye(4)
    m[1:3, 1:3] = [[c, -s], [s, c]]
    return m


def _S(x, y, z):
    return np.diag([x, y, z, 1.0])


def wheel_layout(state, data):
    """Parts of an Offroad wheel mount as Create Tracks' AdjustableWheelMountRenderer draws them.

    Returns (parts, tire): parts = [(model id, 4x4 matrix)], tire = 4x4 matrix for the tire item
    model (or None). Matrices work in 1/16-block units on the part's own coordinates, already
    turned to the block's facing. Decoded rest pose (suspension extension 0), in blocks, in a
    frame where -z points out along the facing; h/lat/lon are the Create Tracks offsets:
      tele_outer  T(lat, -0.375, lon) . RX(atan2(h + 0.375, 0.75)) about the block centre
      tele_inner  the same, then T(0, 0, -(|(h + 0.375, 0.75)| - 1))
      mount       T(lat, h, lon + 0.25)
      springs     about (0.5 + lat, 0.9375, 0.0625 + lon) by atan2(h - 0.3125, 0.1875) + 90 deg;
                  middle stretched in y by (len - 0.25) / 0.875 about y = 0.8125, lower moved
                  down by (len - 0.25 - 0.875), len = |(h - 0.3125, 0.1875)|
      diodes      as authored
      tire        T(lat, h, lon + 0.25) then T(0, 0, -1.625), lying flat item turned 90 deg on X
    """
    data = data or {}
    lat = float(data.get("TracksLateralOffset", 0.0))
    lon = float(data.get("TracksLongitudinalOffset", 0.0))
    # The renderer's height is -extension + TracksHeightOffset. Extension is the suspension's
    # live travel (not saved); WheelMountBlockEntity starts it at 0.5, the resting position.
    h = float(data.get("TracksHeightOffset", 0.0)) - REST_EXTENSION
    hidden = bool(data.get("TracksVisualSuspensionHidden"))
    B = 16.0  # blocks -> 1/16 units

    def about(m, pivot):  # apply m around a pivot point (in 1/16 units)
        return _T(*pivot) @ m @ _T(*(-np.asarray(pivot)))

    face = np.eye(4)
    face[:3, :3] = _blockstate_matrix(0, FACING_Y[state.properties.get("facing", "north")])
    F = about(face, (8, 8, 8))
    c = (8.0, 8.0, 8.0)
    parts = []
    if not hidden:
        a = np.arctan2(h + 0.375, 0.75)
        length = np.hypot(h + 0.375, 0.75)
        outer = _T(lat * B, -0.375 * B, lon * B) @ about(_RX(a), c)
        parts.append(("offroad:block/wheel_mount/tele_outer", F @ outer))
        parts.append(("offroad:block/wheel_mount/tele_inner", F @ outer @ _T(0, 0, -(length - 1) * B)))
        parts.append(("offroad:block/wheel_mount/mount", F @ _T(lat * B, h * B, (lon + 0.25) * B)))
        a2 = np.arctan2(h - 0.3125, 0.1875) + np.pi / 2
        f2 = np.hypot(h - 0.3125, 0.1875) - 0.25
        spring = about(_RX(a2), ((0.5 + lat) * B, 0.9375 * B, (0.0625 + lon) * B))
        parts.append(("offroad:block/wheel_mount/spring_upper", F @ spring))
        parts.append(("offroad:block/wheel_mount/spring_middle",
                      F @ spring @ about(_S(1, f2 / 0.875, 1), (0, 0.8125 * B, 0))))
        parts.append(("offroad:block/wheel_mount/spring_lower", F @ spring @ _T(0, -(f2 - 0.875) * B, 0)))
    for d in ("diode_left", "diode_right"):
        parts.append((f"offroad:block/wheel_mount/{d}", F))
    tire = None
    if data.get("CurrentStack", {}).get("id"):
        # item model centred on the wheel position, lying flat, turned 90 degrees about X
        tire = F @ _T(lat * B, h * B, (lon + 0.25 - 1.625) * B) @ about(_RX(np.pi / 2), c)
    return parts, tire


def _apply(m, tris):
    t = np.asarray(tris, float)
    return t @ m[:3, :3].T + m[:3, 3]


def offroad_wheel(state, data, assets):
    """Suspension, axle and tire of an Offroad wheel mount (see wheel_layout)."""
    data = data or {}
    parts, tire = wheel_layout(state, data)
    out = [_apply(m, model_tris(assets, mid)) for mid, m in parts]
    smooth = np.zeros((0, 3, 3))
    tire_id = data.get("CurrentStack", {}).get("id")
    if tire is not None and tire_id:
        ns, _, name = tire_id.partition(":")
        mesh = assets.obj_mesh(f"{ns}:models/item/{name}/item.obj")
        if mesh is None:
            mesh = assets.obj_mesh(f"{ns}:models/item/{name}/block.obj")
        if mesh is not None:
            smooth = _apply(tire, mesh)
    out = [t for t in out if len(t)]
    return (np.concatenate(out) if out else np.zeros((0, 3, 3))), smooth


HANDLERS = {
    "offroad:wheel_mount": offroad_wheel,
}
