"""Mineways-style map window: pick a world area by dragging a box on a top-down map.

Controls follow Mineways (Mineways.cpp): left-drag pans, right-drag (or Ctrl+left-drag) selects,
the mouse wheel zooms. "Height" is the top of the view and of the selection (blocks above it
are cut away, so you can look inside buildings); "Depth" is the bottom of the selection,
changed with its slider or the [ and ] keys. After a new selection the depth is fitted to the
lowest block visible in it (GetMinimumSelectionHeight), as Mineways' auto-correction does;
press the spacebar to fit it again.
"""
import os
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from . import nbt

ZOOMS = (0.125, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16)
SEA_LEVEL = 62


_MAPS = {}  # (world folder, dimension) -> WorldMap, kept for the whole session


def _instance_of(world_path):
    """The instance folder a world belongs to (<instance>/minecraft/saves/<world>), for its block table."""
    saves = os.path.dirname(os.path.abspath(world_path))
    game = os.path.dirname(saves)
    if os.path.basename(saves).lower() == "saves" and os.path.basename(game).lower() in ("minecraft", ".minecraft"):
        return os.path.dirname(game)
    return game


class MapWindow(tk.Toplevel):
    def __init__(self, parent, world_path, dimension, asset_paths, on_use, box=None):
        super().__init__(parent)
        self.title(f"Pick an area - {os.path.basename(world_path)}")
        self.geometry("1000x720")
        self.minsize(640, 480)
        self.on_use = on_use
        self.world_path, self.dimension, self.asset_paths = world_path, dimension, asset_paths
        self.map = None
        self.zoom_i = ZOOMS.index(2)
        self.cx, self.cz = 0.0, 0.0  # world coords at the canvas centre
        self.box = None  # (minx, minz, maxx, maxz) on the map
        self.top_y, self.min_y, self.max_y = 319, -64, 319
        self._gen = 0
        self._lock = threading.Lock()  # one map render at a time (the caches aren't thread-safe)
        self._photo = None
        self._drag = None
        self._autofix = tk.BooleanVar(value=True)
        self._shade = tk.BooleanVar(value=False)
        self._build()
        try:
            pos = nbt.load(os.path.join(world_path, "level.dat"))["Data"]["Player"]["Pos"]
            self.player = tuple(int(v) for v in pos)
        except Exception:
            self.player = (0, SEA_LEVEL, 0)
        self.cx, self.cz = self.player[0] + 0.5, self.player[2] + 0.5
        if box:
            (x0, y0, z0), (x1, y1, z1) = box
            self.box = (min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1))
            self.depth.set(min(y0, y1))
            self.height.set(max(y0, y1))
            self.cx, self.cz = (x0 + x1) / 2, (z0 + z1) / 2
        self.status_var.set("Loading the world and block colours...")
        threading.Thread(target=self._load, daemon=True).start()

    # -- layout ----------------------------------------------------------------------------------
    def _build(self):
        bar = ttk.Frame(self, padding=(8, 6))
        bar.pack(fill="x")
        ttk.Label(bar, text="Height").pack(side="left")
        self.height = tk.IntVar(value=319)
        self.height_scale = ttk.Scale(bar, from_=-64, to=319, length=220, orient="horizontal",
                                      command=lambda v: self._slid("height", v))
        self.height_scale.pack(side="left", padx=4)
        self.height_lbl = ttk.Label(bar, width=5)
        self.height_lbl.pack(side="left")
        ttk.Label(bar, text="Depth").pack(side="left", padx=(16, 0))
        self.depth = tk.IntVar(value=SEA_LEVEL)
        self.depth_scale = ttk.Scale(bar, from_=-64, to=319, length=220, orient="horizontal",
                                     command=lambda v: self._slid("depth", v))
        self.depth_scale.pack(side="left", padx=4)
        self.depth_lbl = ttk.Label(bar, width=5)
        self.depth_lbl.pack(side="left")
        ttk.Checkbutton(bar, text="Fit depth to selection", variable=self._autofix).pack(side="left", padx=(16, 0))
        ttk.Checkbutton(bar, text="Elevation shading", variable=self._shade,
                        command=self._shade_changed).pack(side="left", padx=(12, 0))

        self.canvas = tk.Canvas(self, background="#1e1e22", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        c = self.canvas
        c.bind("<Configure>", lambda e: self.redraw())
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._motion)
        c.bind("<ButtonRelease-1>", self._release)
        c.bind("<ButtonPress-3>", lambda e: self._press(e, select=True))
        c.bind("<B3-Motion>", self._motion)
        c.bind("<ButtonRelease-3>", self._release)
        c.bind("<MouseWheel>", self._wheel)
        c.bind("<Motion>", self._hover)
        self.bind("<bracketleft>", lambda e: self._nudge_depth(-1))
        self.bind("<bracketright>", lambda e: self._nudge_depth(1))
        self.bind("<space>", lambda e: self._fit_depth(force=True))

        bottom = ttk.Frame(self, padding=(8, 6))
        bottom.pack(fill="x")
        self.status_var = tk.StringVar()
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left", fill="x", expand=True)
        ttk.Button(bottom, text="Go to player", command=self._to_player).pack(side="left", padx=4)
        ttk.Button(bottom, text="Use selection", command=self._use).pack(side="left", padx=4)
        ttk.Button(bottom, text="Cancel", command=self.destroy).pack(side="left")
        self.sel_var = tk.StringVar(value="Right-drag (or Ctrl+drag) to select an area.")
        ttk.Label(self, textvariable=self.sel_var, padding=(8, 0, 8, 6), style="Hint.TLabel").pack(fill="x")
        self._sync_sliders()

    def _load(self):
        try:
            from .assets import Assets
            from .voxelize import Voxelizer
            from .world import World
            from .worldmap import WorldMap
            paths = self.asset_paths
            key = (os.path.normcase(os.path.abspath(self.world_path)), self.dimension)
            m = _MAPS.get(key)  # reopening the window keeps everything already drawn
            if m is None:
                # the mods are only opened if a block is missing from the instance's block table
                m = _MAPS[key] = WorldMap(World(self.world_path, self.dimension),
                                          lambda: Voxelizer(Assets(paths), 16),
                                          instance=_instance_of(self.world_path), asset_paths=paths)
            px, _, pz = self.player
            rng = m.height_range(px >> 4, pz >> 4)
            err = None
        except Exception as e:
            m, rng, err = None, None, e
        self.after(0, lambda: self._loaded(m, rng, err))

    def _loaded(self, m, rng, err):
        if err:
            self.status_var.set(f"Couldn't open the world: {err}")
            return
        self.map = m
        self.min_y, self.max_y = m.min_y, m.max_y
        for s in (self.height_scale, self.depth_scale):
            s.configure(from_=m.min_y, to=m.max_y)
        if self.box is None:
            self.height.set(m.max_y)
        self._sync_sliders()
        self.status_var.set("")
        self.redraw()

    # -- drawing ---------------------------------------------------------------------------------
    @property
    def zoom(self):
        return ZOOMS[self.zoom_i]

    def _view(self):
        w, h = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        bw, bh = int(np.ceil(w / self.zoom)) + 2, int(np.ceil(h / self.zoom)) + 2
        x0 = int(np.floor(self.cx - w / (2 * self.zoom)))
        z0 = int(np.floor(self.cz - h / (2 * self.zoom)))
        return w, h, x0, z0, bw, bh

    def to_world(self, px, py):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        return (int(np.floor(self.cx + (px - w / 2) / self.zoom)),
                int(np.floor(self.cz + (py - h / 2) / self.zoom)))

    def _box3(self):
        if not self.box:
            return None
        minx, minz, maxx, maxz = self.box
        return (minx, self.depth.get(), minz, maxx, self.height.get(), maxz)

    def redraw(self):
        if self.map is None:
            return
        self._gen += 1
        gen = self._gen
        w, h, x0, z0, bw, bh = self._view()
        top, box, zoom, cx, cz = self.height.get(), self._box3(), self.zoom, self.cx, self.cz

        def work():
            started = time.monotonic()
            last = [started]

            def prog(f, partial):
                now = time.monotonic()
                # only when new chunks take a while: show what's drawn so far a few times a second
                if gen == self._gen and now - last[0] > 0.3:
                    last[0] = now
                    snap = partial.copy()
                    self.after(0, lambda: self._show(gen, snap, x0, z0, zoom, cx, cz, w, h,
                                                     f"Drawing map... {int(f * 100)}%"))
            with self._lock:
                if gen != self._gen:
                    return  # a newer redraw is queued
                try:
                    img = self.map.render(x0, z0, bw, bh, top, box, progress=prog,
                                          stop=lambda: gen != self._gen)
                    if img is None:
                        return  # the view moved on; what was drawn stays cached
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    msg = f"Map error: {e}"
                    self.after(0, lambda: self.status_var.set(msg))
                    return
            self.after(0, lambda: self._show(gen, img, x0, z0, zoom, cx, cz, w, h))
        threading.Thread(target=work, daemon=True).start()

    def _show(self, gen, img, x0, z0, zoom, cx, cz, w, h, note=None):
        if gen != self._gen:
            return
        from PIL import Image, ImageTk
        im = Image.fromarray(img)
        im = im.resize((max(1, int(round(im.width * zoom))), max(1, int(round(im.height * zoom)))),
                       Image.NEAREST if zoom >= 1 else Image.BOX)
        # offset so world (cx, cz) lands at the canvas centre
        ox = int(round(w / 2 - (cx - x0) * zoom))
        oy = int(round(h / 2 - (cz - z0) * zoom))
        self._photo = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)
        px, _, pz = self.player
        sx, sy = w / 2 + (px + 0.5 - cx) * zoom, h / 2 + (pz + 0.5 - cz) * zoom
        self.canvas.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, outline="#ff4040", width=2)
        self.status_var.set(note or f"Zoom {zoom:g}x")

    # -- mouse -----------------------------------------------------------------------------------
    def _press(self, e, select=False):
        self.canvas.focus_set()
        select = select or bool(e.state & 0x4)  # Ctrl+left selects, like Mineways on a Mac
        if select:
            x, z = self.to_world(e.x, e.y)
            self._drag = ("select", x, z)
            self.box = (x, z, x, z)
        else:
            self._drag = ("pan", e.x, e.y, self.cx, self.cz)

    def _motion(self, e):
        if not self._drag:
            return
        if self._drag[0] == "pan":
            _, sx, sy, cx, cz = self._drag
            new_cx, new_cz = cx - (e.x - sx) / self.zoom, cz - (e.y - sy) / self.zoom
            self.canvas.move("all", (self.cx - new_cx) * self.zoom, (self.cz - new_cz) * self.zoom)
            self.cx, self.cz = new_cx, new_cz
        else:
            _, x0, z0 = self._drag
            x, z = self.to_world(e.x, e.y)
            self.box = (min(x0, x), min(z0, z), max(x0, x), max(z0, z))
            self._describe()
        self.redraw()

    def _release(self, e):
        if self._drag and self._drag[0] == "select":
            self._fit_depth()
        self._drag = None

    def _wheel(self, e):
        before = self.to_world(e.x, e.y)
        self.zoom_i = max(0, min(len(ZOOMS) - 1, self.zoom_i + (1 if e.delta > 0 else -1)))
        # keep the block under the mouse in place
        after = self.to_world(e.x, e.y)
        self.cx += before[0] - after[0]
        self.cz += before[1] - after[1]
        self.redraw()

    def _hover(self, e):
        if self.map is None:
            return
        x, z = self.to_world(e.x, e.y)
        try:
            y, st = self.map.block_at(x, z, self.height.get())
        except Exception:
            y, st = None, None
        where = f"{x}, {y if y is not None else '-'}, {z}"
        self.status_var.set(f"{where}   {st.name if st else 'empty'}   (zoom {self.zoom:g}x)")

    # -- height & depth --------------------------------------------------------------------------
    def _slid(self, which, v):
        v = int(float(v))
        var = self.height if which == "height" else self.depth
        if var.get() == v:
            return
        var.set(v)
        if self.depth.get() > self.height.get():  # keep depth <= height
            (self.height if which == "depth" else self.depth).set(v)
        self._sync_sliders()
        self._describe()
        self.redraw()

    def _sync_sliders(self):
        self.height_scale.set(self.height.get())
        self.depth_scale.set(self.depth.get())
        self.height_lbl.configure(text=str(self.height.get()))
        self.depth_lbl.configure(text=str(self.depth.get()))

    def _nudge_depth(self, d):
        self.depth.set(max(self.min_y, min(self.height.get(), self.depth.get() + d)))
        self._sync_sliders()
        self._describe()
        self.redraw()

    def _fit_depth(self, force=False):
        """Mineways' depth auto-correction: set the depth to the lowest block visible in the box."""
        if not self.box or self.map is None or not (force or self._autofix.get()):
            self._describe()
            return
        minx, minz, maxx, maxz = self.box
        low = self.map.min_selection_height(minx, minz, maxx, maxz, self.height.get())
        if low != self.depth.get():
            self.depth.set(max(self.min_y, min(self.height.get(), low)))
            self._sync_sliders()
            self.redraw()
        self._describe(note=f"Depth set to {self.depth.get()}, the lowest block visible in the selection.")

    def _describe(self, note=""):
        if not self.box:
            return
        minx, minz, maxx, maxz = self.box
        y0, y1 = self.depth.get(), self.height.get()
        self.sel_var.set(f"Selection {minx}, {y0}, {minz} to {maxx}, {y1}, {maxz}  "
                         f"({maxx - minx + 1} x {y1 - y0 + 1} x {maxz - minz + 1} blocks)   {note}")

    def _shade_changed(self):
        if self.map:
            self.map.set_elevation_shading(self._shade.get())
            self.redraw()

    def _to_player(self):
        self.cx, self.cz = self.player[0] + 0.5, self.player[2] + 0.5
        self.redraw()

    def _use(self):
        if not self.box:
            self.sel_var.set("Select an area first: right-drag (or Ctrl+drag) on the map.")
            return
        minx, minz, maxx, maxz = self.box
        top = self.height.get()
        with self._lock:  # don't export the empty sky above the build
            highest = self.map.max_selection_height(minx, minz, maxx, maxz, top) if self.map else None
        if highest is not None:
            top = max(self.depth.get(), highest)
        self.on_use((minx, self.depth.get(), minz), (maxx, top, maxz))
        self.destroy()
