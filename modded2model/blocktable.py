"""Build the block table for a whole Minecraft instance: the map colour and opacity of every
block state of every mod, worked out from the blocks' own models.

This is the instance's equivalent of Mineways' built-in block table. It is built once when the
game folder is chosen (in the background, on all CPU cores) and saved per instance; the world
map then only looks colours up, so it draws immediately. It is rebuilt when a mod jar or
resource pack changes.

Work is shared: block states that use the same model with the same tilt look the same from
above (turning a block to face another way doesn't change its colour), so each distinct
(model, x-rotation) is measured once. Blocks drawn by code (copycats and others in
codeshapes.RULES) depend on their properties, so their states are measured one by one.
"""
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

from .schematic import BlockState

_BLOCKSTATE = re.compile(r"assets/([^/]+)/blockstates/(.+)\.json$")
_worker = None  # per-process BlockColors


def block_names(assets):
    """Every block with a blockstate file in the instance (mods, resource packs, the game)."""
    names = set()
    for s in assets.sources:
        if s.is_dir:
            continue
        for n in s.names:
            m = _BLOCKSTATE.match(n)
            if m:
                names.add(f"{m.group(1)}:{m.group(2)}")
    return sorted(names)


def _props(key):
    """Variant key 'facing=north,half=top' -> sorted ((k, v), ...)."""
    if not key or key == "normal":
        return ()
    return tuple(sorted(tuple(kv.split("=", 1)) for kv in key.split(",") if "=" in kv))


def plan(assets):
    """The work list: [(job key, BlockState)] to measure, and for each block its variant keys
    with the job each one uses, plus each block's default job."""
    from .codeshapes import RULES
    jobs, variants, defaults = {}, {}, {}
    for name in block_names(assets):
        bs = assets.blockstate(name)
        if not isinstance(bs, dict):
            continue
        by_code = name in RULES
        entries = []
        if "variants" in bs and isinstance(bs["variants"], dict):
            for key, v in bs["variants"].items():
                v = v[0] if isinstance(v, list) and v else v
                if not isinstance(v, dict):
                    continue
                props = _props(key)
                job = f"{name}|{key}" if by_code else f"model|{v.get('model')}|{v.get('x', 0)}"
                jobs.setdefault(job, BlockState(name, props))
                entries.append((key, job))
        if entries:
            variants[name] = entries
            defaults[name] = entries[0][1]
        else:  # multipart (fences, walls, pipes...) or empty: the block with no properties set
            job = f"{name}|"
            jobs.setdefault(job, BlockState(name, ()))
            defaults[name] = job
    return jobs, variants, defaults


def _init(asset_paths):
    global _worker
    import contextlib
    import io
    from .assets import Assets
    from .voxelize import Voxelizer
    from .worldmap import BlockColors
    with contextlib.redirect_stdout(io.StringIO()):
        _worker = BlockColors(Voxelizer(Assets(asset_paths), 16))


def _measure(batch):
    import contextlib
    import io
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        for job, state in batch:
            try:
                rgb, alpha = _worker._uncached(state)
                out.append((job, [round(float(c), 1) for c in rgb] + [round(float(alpha), 3)]))
            except Exception:
                out.append((job, [150.0, 150.0, 150.0, 1.0]))
    return out


def _worker_count():
    """Each worker holds every mod's assets in memory (roughly 1 GB for a big pack), so the
    count is set by free memory as well as by CPU cores."""
    try:
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("free", ctypes.c_ulonglong)] + [(f"r{i}", ctypes.c_ulonglong) for i in range(5)]
        m = _Mem()
        m.len = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        by_memory = int(m.free / 2 ** 30 / 1.2)
    except Exception:
        by_memory = 2
    return max(1, min(4, (os.cpu_count() or 2) - 1, by_memory))


def build(instance, asset_paths, progress=None, workers=None, force=False):
    """Build (or confirm up to date) the instance's block table. progress(fraction, text).
    Returns the table file path."""
    from .assets import Assets
    from .worldmap import _sources_signature, table_path
    path = table_path(instance)
    sig = _sources_signature(asset_paths)
    old = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
        except (OSError, ValueError):
            old = {}
    if not force and old.get("sources") == sig and old.get("complete"):
        if progress:
            progress(1.0, f"Block map up to date ({len(old.get('names', {}))} blocks)")
        return path

    t0 = time.time()
    if progress:
        progress(0.0, "Reading the instance's blocks")
    assets = Assets(asset_paths)
    jobs, variants, defaults = plan(assets)
    done = old.get("jobs", {}) if old.get("sources") == sig else {}  # resume a half-built table
    todo = [(k, s) for k, s in jobs.items() if k not in done]
    if progress:
        progress(0.02, f"Measuring {len(jobs)} block models for {len(defaults)} blocks")

    def save(complete):
        data = {"instance": os.path.abspath(instance), "sources": sig, "complete": complete,
                "jobs": done, "variants": variants, "names": defaults, "blocks": old.get("blocks", {})
                if old.get("sources") == sig else {}}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    if todo:
        n = workers or _worker_count()
        batches = [todo[i:i + 40] for i in range(0, len(todo), 40)]
        last_save = time.time()
        with ProcessPoolExecutor(max_workers=n, initializer=_init, initargs=(list(asset_paths),)) as ex:
            for k, result in enumerate(ex.map(_measure, batches)):
                done.update(result)
                if progress:
                    progress(0.02 + 0.98 * (k + 1) / len(batches),
                             f"Building block map: {len(done)}/{len(jobs)} models")
                if time.time() - last_save > 20:  # keep progress if the window is closed
                    save(False)
                    last_save = time.time()
    save(True)
    if progress:
        progress(1.0, f"Block map ready: {len(defaults)} blocks, {len(jobs)} models, "
                      f"in {time.time() - t0:.0f}s")
    return path
