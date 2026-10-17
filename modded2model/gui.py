"""Window for Minecraft-Modded2Model: python -m modded2model.gui

Left: what to export (a schematic and which part of it, a world region, or an Aeronautics ship).
Right: a live preview of it, rendered in the background; schematics with Aeronautics sub-levels
show every sub-level numbered and coloured. Below: options by topic, then Export and the log.
"""
import base64
import contextlib
import io
import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import __version__, instance, nbt, sable, theme
from .__main__ import main as cli_main

SETTINGS = os.path.join(os.path.expanduser("~"), ".modded2model.json")
DEFAULT_INSTANCE = os.path.join(os.environ.get("APPDATA", ""), "PrismLauncher", "instances", "AeronauticsModpack")
MAIN_PART = "Main build (with merged sub-levels)"


# moving contraptions: what the window shows -> the --contraptions value
CONTRAPTION_ARGS = {"As in the world": "world", "Snapped to the grid": "snap", "As built": "assembled",
                    "Leave out": "none"}
CONTRAPTION_MODES = tuple(CONTRAPTION_ARGS)
OLD_SETTINGS = os.path.join(os.path.expanduser("~"), ".modways.json")  # from before the rename


def load_settings():
    for path in (SETTINGS, OLD_SETTINGS):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return {}


def save_settings(data):
    try:
        with open(SETTINGS, "w") as f:
            json.dump(data, f, indent=1)
    except OSError:
        pass


class _QueueWriter(io.TextIOBase):
    def __init__(self, q):
        self.q = q

    def write(self, s):
        self.q.put(s)
        return len(s)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Minecraft-Modded2Model v{__version__}")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1440, int(sw * 0.85))}x{min(940, int(sh * 0.88))}")
        self.minsize(1150, 680)
        self.settings = load_settings()
        self.log_q = queue.Queue()
        self.done_q = queue.Queue()
        self.busy = False
        self.schematics = []
        self._vox = None  # shared Voxelizer for previews (loaded once per instance)
        self._vox_instance = None
        self._preview_token = 0
        self._preview_img = None
        theme.apply(self)

        self._build_header()
        self._build_bottom()  # packed from the bottom up, so the footer, log and options keep their room
        self._build_options()
        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=(12, 0))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=3)
        self._build_sources(left)
        self._build_preview(right)

        self.after(100, self.drain_queues)
        if os.path.isdir(self.instance_var.get()):
            self.load_instance()

    # --- layout ---------------------------------------------------------------------------
    @staticmethod
    def _card(parent, step, title, hint=None):
        """A panel with a numbered heading; returns (outer frame, body frame)."""
        outer = ttk.Frame(parent, style="Card.TFrame", padding=(14, 12))
        head = ttk.Frame(outer, style="Card.TFrame")
        head.pack(fill="x", pady=(0, 8))
        if step:
            ttk.Label(head, text=step, style="Step.TLabel").pack(side="left", padx=(0, 8))
        ttk.Label(head, text=title, style="Head.TLabel").pack(side="left")
        if hint:
            ttk.Label(head, text=hint, style="Hint.TLabel").pack(side="left", padx=(10, 0))
        body = ttk.Frame(outer, style="Card.TFrame")
        body.pack(fill="both", expand=True)
        return outer, body

    def _build_header(self):
        top = ttk.Frame(self, style="Header.TFrame", padding=(18, 12))
        top.pack(fill="x")
        brand = ttk.Frame(top, style="Header.TFrame")
        brand.pack(side="left")
        ttk.Label(brand, text="Minecraft-Modded2Model", style="Title.TLabel").pack(side="left")
        ttk.Label(brand, text="  modded builds  →  3D prints and Blender",
                  style="Subtitle.TLabel").pack(side="left", pady=(8, 0))
        inst = ttk.Frame(top, style="Header.TFrame")
        inst.pack(side="right", fill="x", expand=True, padx=(40, 0))
        ttk.Button(inst, text="Load", command=self.load_instance).pack(side="right", padx=(6, 0))
        ttk.Button(inst, text="Browse…", command=self.pick_instance).pack(side="right", padx=(6, 0))
        self.instance_var = tk.StringVar(value=self.settings.get("instance", DEFAULT_INSTANCE))
        ttk.Entry(inst, textvariable=self.instance_var).pack(side="right", fill="x", expand=True)
        ttk.Label(inst, text="Prism instance", style="Hint.TLabel").pack(side="right", padx=(0, 8))
        ttk.Frame(self, style="Rule.TFrame", height=1).pack(fill="x")

    def _build_sources(self, parent):
        card, body = self._card(parent, "1", "Choose a build")
        card.pack(fill="both", expand=True, padx=(0, 6))
        self.tabs = ttk.Notebook(body)
        self.tabs.pack(fill="both", expand=True)
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self.refresh_preview())

        # Schematic
        sch = ttk.Frame(self.tabs, style="Card.TFrame", padding=(2, 10, 2, 2))
        self.tabs.add(sch, text="Schematic")
        r = ttk.Frame(sch, style="Card.TFrame")
        r.pack(fill="x")
        ttk.Label(r, text="⌕", font=("Segoe UI Symbol", 12), foreground=theme.MUTED).pack(side="left", padx=(0, 6))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *a: self.fill_schematics())
        ttk.Entry(r, textvariable=self.search_var).pack(side="left", fill="x", expand=True)
        # the part picker is packed from the bottom first, so the list shrinks instead of hiding it
        part = ttk.Frame(sch, style="Card.TFrame")
        part.pack(side="bottom", fill="x")
        r = ttk.Frame(part, style="Card.TFrame")
        r.pack(fill="x")
        ttk.Label(r, text="Part to export").pack(side="left", padx=(0, 8))
        self.part_var = tk.StringVar(value=MAIN_PART)
        self.part_box = ttk.Combobox(r, textvariable=self.part_var, values=[MAIN_PART], state="readonly")
        self.part_box.pack(side="left", fill="x", expand=True)
        self.part_box.bind("<<ComboboxSelected>>", lambda e: self.refresh_preview())
        ttk.Label(part, text="Aeronautics sub-levels are listed here; the preview shows which is which.",
                  style="Hint.TLabel").pack(anchor="w", pady=(4, 0))
        lf = ttk.Frame(sch, style="Card.TFrame")
        lf.pack(fill="both", expand=True, pady=8)
        self.schem_list = theme.listbox(lf, exportselection=False, height=3)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.schem_list.yview)
        self.schem_list.configure(yscrollcommand=sb.set)
        self.schem_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.schem_list.bind("<<ListboxSelect>>", lambda e: self.schematic_selected())

        # World region
        reg = ttk.Frame(self.tabs, style="Card.TFrame", padding=(2, 10, 2, 2))
        self.tabs.add(reg, text="World region")
        grid = ttk.Frame(reg, style="Card.TFrame")
        grid.pack(fill="x")
        ttk.Label(grid, text="World").grid(row=0, column=0, sticky="w", pady=2)
        self.world_var = tk.StringVar()
        self.world_box = ttk.Combobox(grid, textvariable=self.world_var, state="readonly")
        self.world_box.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(8, 0))
        ttk.Label(grid, text="Dimension").grid(row=1, column=0, sticky="w", pady=3)
        self.dim_var = tk.StringVar(value="overworld")
        ttk.Combobox(grid, textvariable=self.dim_var, values=["overworld", "nether", "end"]).grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=(8, 0))
        self.corners = []
        for i, label in enumerate(("From x y z", "To x y z")):
            ttk.Label(grid, text=label).grid(row=2 + i, column=0, sticky="w", pady=3)
            vars_ = [tk.StringVar() for _ in range(3)]
            for j, v in enumerate(vars_):
                ttk.Entry(grid, textvariable=v, width=9).grid(row=2 + i, column=1 + j, padx=(8, 0), sticky="w")
            self.corners.append(vars_)
        grid.columnconfigure(3, weight=1)
        # one row of buttons, so the card fits World, Dimension, From and To above it
        r = ttk.Frame(reg, style="Card.TFrame")
        r.pack(fill="x", pady=(8, 4))
        ttk.Button(r, text="Pick on map…", command=self.pick_on_map).pack(side="left")
        ttk.Button(r, text="Use area around player", command=self.around_player).pack(side="left", padx=(6, 0))
        ttk.Label(r, text="radius").pack(side="left", padx=(8, 4))
        self.radius_var = tk.StringVar(value="16")
        ttk.Entry(r, textvariable=self.radius_var, width=5).pack(side="left")
        ttk.Button(r, text="Preview", command=self.refresh_preview).pack(side="right")
        ttk.Label(reg, text="Pick on map: drag a box on a top-down map of the world. In game, F3 shows block "
                            "coordinates.", style="Hint.TLabel", wraplength=520).pack(anchor="w")

        # Aeronautics ship
        shp = ttk.Frame(self.tabs, style="Card.TFrame", padding=(2, 10, 2, 2))
        self.tabs.add(shp, text="Aeronautics ship")
        r = ttk.Frame(shp, style="Card.TFrame")
        r.pack(fill="x")
        ttk.Label(r, text="World").pack(side="left")
        self.ship_world_var = tk.StringVar()
        self.ship_world_box = ttk.Combobox(r, textvariable=self.ship_world_var, state="readonly")
        self.ship_world_box.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r, text="Find ships", command=self.find_ships).pack(side="left")
        cols = ("id", "size", "blocks")
        self.ship_tree = ttk.Treeview(shp, columns=cols, show="headings", selectmode="browse", height=5)
        for c, w in zip(cols, (150, 110, 70)):
            self.ship_tree.heading(c, text=c.title())
            self.ship_tree.column(c, width=w, anchor="w")
        self.ship_tree.pack(fill="both", expand=True, pady=8)
        self.ship_tree.bind("<<TreeviewSelect>>", lambda e: self.refresh_preview())

    def _build_preview(self, parent):
        card, body = self._card(parent, None, "Preview")
        card.pack(fill="both", expand=True, padx=(6, 0))
        stage_ = tk.Frame(body, background=theme.PREVIEW)
        stage_.pack(fill="both", expand=True)
        self.preview_label = ttk.Label(stage_, anchor="center", style="Preview.TLabel",
                                       text="Pick a build on the left to see it here.")
        self.preview_label.pack(fill="both", expand=True, padx=4, pady=4)
        bar = ttk.Frame(body, style="Card.TFrame")
        bar.pack(fill="x", pady=(8, 4))
        self.preview_status = ttk.Label(bar, text="", style="Hint.TLabel")
        self.preview_status.pack(side="left")
        self.preview_pct = ttk.Label(bar, text="", style="Hint.TLabel", width=5, anchor="e")
        self.preview_pct.pack(side="right")
        self.preview_bar = ttk.Progressbar(body, mode="determinate", maximum=100)
        self.preview_bar.pack(fill="x")
        self.preview_label.bind("<Configure>", lambda e: self._fit_preview())
        self.preview_label.bind("<Button-1>", self.open_big_preview)
        self.preview_label.configure(cursor="hand2")

    def _build_options(self):
        s = self.settings
        self.opts = {}
        self._deps = []  # (checkbox var, widgets enabled only when it's on)

        def num(frame, key, default, width=6):
            v = tk.StringVar(value=str(s.get(key, default)))
            self.opts[key] = v
            return ttk.Entry(frame, textvariable=v, width=width, justify="center")

        def flag(frame, key, label, default):
            v = tk.BooleanVar(value=s.get(key, default))
            cb = ttk.Checkbutton(frame, text=label, variable=v, command=self.update_option_states)
            return v, cb

        card = ttk.Frame(self, style="Card.TFrame", padding=(14, 12))
        card.pack(side="bottom", fill="x", padx=14, pady=(12, 0))
        head = ttk.Frame(card, style="Card.TFrame")
        head.pack(fill="x")
        ttk.Label(head, text="2", style="Step.TLabel").pack(side="left", padx=(0, 8))
        ttk.Label(head, text="Format and options", style="Head.TLabel").pack(side="left", padx=(0, 18))

        # format is a segmented switch; the option tabs below follow it
        seg = ttk.Frame(head, style="Card.TFrame")
        seg.pack(side="left")
        self.fmt_var = tk.StringVar(value=s.get("format", "stl"))
        for val, label in (("stl", "STL  ·  3D print"), ("3mf", "3MF  ·  3D print"),
                           ("glb", "GLB  ·  Blender, textured")):
            ttk.Radiobutton(seg, text=label, value=val, variable=self.fmt_var, style="Segment.Toolbutton",
                            command=self.update_option_states).pack(side="left")
        ttk.Label(head, text="voxels per block, 0 = auto", style="Hint.TLabel").pack(side="right")
        num(head, "res", "0", 5).pack(side="right", padx=6)
        ttk.Label(head, text="Detail").pack(side="right")

        self.opt_tabs = ttk.Notebook(card)
        self.opt_tabs.pack(fill="x", pady=(10, 0))

        def page():
            return ttk.Frame(self.opt_tabs, style="Card.TFrame", padding=(4, 10, 4, 2))

        bl = page()
        self.merge_var, self.merge_cb = flag(bl, "merge", "Connect meshes, cull touching faces", True)
        self.merge_cb.grid(row=0, column=0, sticky="w")
        self.glass_var, self.glass_cb = flag(bl, "glass_cull", "One-sided glass (backface culling)", False)
        self.glass_cb.grid(row=1, column=0, sticky="w")

        cl = page()
        self.backup_var, cb = flag(cl, "remove_backups", "Remove backup blocks", True)
        cb.grid(row=1, column=0, sticky="w")
        self.vs_var, cb = flag(cl, "replace_vs", "Replace Valkyrien Skies blocks", True)
        cb.grid(row=2, column=0, sticky="w")
        r = ttk.Frame(cl, style="Card.TFrame")
        r.grid(row=3, column=0, sticky="w", pady=4)
        ttk.Label(r, text="Blocks with no model").pack(side="left")
        self.missing_var = tk.StringVar(value=s.get("missing", "cube"))
        ttk.Combobox(r, textvariable=self.missing_var, values=("cube", "skip"), width=6,
                     state="readonly").pack(side="left", padx=8)

        # 3D printing: model size and print options, grouped into columns
        pr = page()
        self.print_frame, self.blender_frame = pr, bl
        self.opt_tabs.add(pr, text="3D printing")
        self.opt_tabs.add(bl, text="Blender")
        self.opt_tabs.add(cl, text="Clean-up")
        pr.columnconfigure(0, weight=2, uniform="col")
        pr.columnconfigure(1, weight=3, uniform="col")
        rows = {}

        def line(col, title=None):
            if title:
                g = rows.get(col, 0)
                ttk.Label(pr, text=title, style="Section.TLabel").grid(
                    row=g, column=col, sticky="w", pady=(0 if g == 0 else 8, 2), padx=(0, 24))
                rows[col] = g + 1
            g = rows.get(col, 0)
            f = ttk.Frame(pr, style="Card.TFrame")
            f.grid(row=g, column=col, sticky="w", padx=(0, 24))
            rows[col] = g + 1
            return f

        f = line(0, "Size and orientation")
        self.zup_var, cb = flag(f, "z_up", "Make Z the up direction instead of Y", True)
        cb.pack(side="left")
        f = line(0)
        self.size_var = tk.StringVar(value=s.get("size_mode", "block"))
        ttk.Radiobutton(f, text="Make the model", value="height", variable=self.size_var,
                        command=self.update_option_states).pack(side="left")
        self.height_entry = num(f, "height_cm", "5")
        self.height_entry.pack(side="left", padx=6)
        ttk.Label(f, text="cm in height").pack(side="left")
        f = line(0)
        ttk.Radiobutton(f, text="Make each block", value="block", variable=self.size_var,
                        command=self.update_option_states).pack(side="left")
        self.block_entry = num(f, "block_mm", "5")
        self.block_entry.pack(side="left", padx=6)
        ttk.Label(f, text="mm").pack(side="left")
        f = line(0)
        ttk.Label(f, text="Model's units").pack(side="left", pady=3)
        self.units_var = tk.StringVar(value=s.get("units", "Millimeters"))
        ttk.Combobox(f, textvariable=self.units_var, values=("Millimeters", "Centimeters", "Meters", "Inches"),
                     width=12, state="readonly").pack(side="left", padx=8)

        f = line(0, "Detail")
        self.lesser_var, cb = flag(f, "export_lesser", "Export lesser, detailed blocks", True)
        cb.pack(side="left")
        self.fatten_var, fat = flag(f, "fatten", "Fatten them (safer printing)", False)
        fat.pack(side="left", padx=(16, 0))
        self._deps.append((self.lesser_var, [fat]))
        f = line(0)
        ttk.Label(f, text="Min thickness").pack(side="left")
        num(f, "min_thick", "0", 5).pack(side="left", padx=6)
        ttk.Label(f, text="mm (0 = exact)", style="Hint.TLabel").pack(side="left")
        self.voxel_var, cb = flag(f, "voxel", "Blocky voxel export", False)
        cb.pack(side="left", padx=(16, 0))

        f = line(1, "Make it printable")
        self.fill_var, cb = flag(f, "fill_bubbles", "Fill air bubbles", True)
        cb.pack(side="left")
        self.seal_var, seal = flag(f, "seal_entrances", "Seal off entrances", False)
        seal.pack(side="left", padx=(16, 0))
        self.tunnel_var, tun = flag(f, "seal_tunnels", "Fill isolated tunnels in the base", False)
        tun.pack(side="left", padx=(16, 0))
        self._deps.append((self.fill_var, [seal, tun]))

        f = line(1)
        self.connect_var, cb = flag(f, "connect_parts", "Connect parts sharing an edge", True)
        cb.pack(side="left")
        self.corner_var, corner = flag(f, "connect_corners", "Connect corner tips", True)
        corner.pack(side="left", padx=(16, 0))
        self.weld_var, weld = flag(f, "weld_all", "Weld all shared edges", False)
        weld.pack(side="left", padx=(16, 0))
        self._deps.append((self.connect_var, [corner, weld]))

        f = line(1)
        self.float_var, cb = flag(f, "delete_floating", "Delete floating trees and parts smaller than", True)
        cb.pack(side="left")
        fe = num(f, "floater_count", "16", 5)
        fe.pack(side="left", padx=6)
        ttk.Label(f, text="blocks").pack(side="left")
        self._deps.append((self.float_var, [fe]))

        f = line(1)
        self.hollowb_var, cb = flag(f, "hollow_bottom", "Hollow out the bottom, walls", True)
        cb.pack(side="left")
        he = num(f, "hollow_mm", "2", 5)
        he.pack(side="left", padx=6)
        ttk.Label(f, text="mm thick").pack(side="left")
        self.super_var, sup = flag(f, "superhollow", "Superhollow", True)
        sup.pack(side="left", padx=(16, 0))
        self._deps.append((self.hollowb_var, [he, sup]))

        f = line(1)
        self.melt_var, cb = flag(f, "melt_snow", "Melt snow blocks (stronger with fill, seal and hollow)", False)
        cb.pack(side="left")

        f = line(1)
        ttk.Label(f, text="Non-solid blocks inside (cogwheels, gearboxes, shafts…)").pack(side="left", pady=3)
        old = "Remove (hollow)" if s.get("hollow_interior") else "Keep"
        self.interior_var = tk.StringVar(value=s.get("interior", old))
        ttk.Combobox(f, textvariable=self.interior_var, values=("Keep", "Remove (hollow)", "Make solid blocks"),
                     width=18, state="readonly").pack(side="left", padx=8)
        self.update_option_states()

    def _set_state(self, widget, on):
        try:
            widget.state(["!disabled"] if on else ["disabled"])
        except (AttributeError, tk.TclError):
            widget.configure(state="normal" if on else "disabled")
        for child in widget.winfo_children():
            self._set_state(child, on)

    def update_option_states(self):
        glb = self.fmt_var.get() == "glb"
        # only the chosen format's tab is shown (plus Clean-up), and it comes up when the format changes
        mine, other = (self.blender_frame, self.print_frame) if glb else (self.print_frame, self.blender_frame)
        self.opt_tabs.hide(other)
        if self.opt_tabs.tab(mine, "state") == "hidden" or getattr(self, "_shown_fmt", None) != glb:
            self.opt_tabs.insert(0, mine)
            self.opt_tabs.select(mine)
        self._shown_fmt = glb
        if glb:
            return
        for var, widgets in self._deps:
            for w in widgets:
                self._set_state(w, var.get())
        by_height = self.size_var.get() == "height"
        self._set_state(self.height_entry, by_height)
        self._set_state(self.block_entry, not by_height)

    def _build_bottom(self):
        lf = tk.Frame(self, background=theme.LOG)
        lf.pack(side="bottom", fill="x")
        self.log = tk.Text(lf, height=3, state="disabled", wrap="word", relief="flat", borderwidth=0,
                           background=theme.LOG, foreground=theme.MUTED, font=theme.MONO,
                           insertbackground=theme.TEXT, selectbackground=theme.ACCENT, padx=18, pady=8,
                           highlightthickness=0)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.log.tag_configure("done", foreground="#7fcf8f")
        self.log.tag_configure("error", foreground="#ef7b6c")
        self.log.tag_configure("head", foreground=theme.ACCENT)

        bottom = ttk.Frame(self, padding=(14, 10))
        bottom.pack(side="bottom", fill="x")
        self._footer = bottom
        self.export_btn = ttk.Button(bottom, text="Export  →", style="Accent.TButton", command=self.export)
        self.export_btn.pack(side="right")
        self.progress = ttk.Progressbar(bottom, mode="determinate", length=240)
        self.progress.pack(side="right", padx=14)
        self.status = ttk.Label(bottom, text="Ready", style="Status.TLabel")
        self.status.pack(side="left")

    # --- background work ------------------------------------------------------------------
    def write_log(self, text):
        tag = ("error" if text.startswith("Error") else "done" if text.startswith("Done") else
               "head" if text.strip().startswith(("Exporting", "Loaded", "Found")) else None)
        self.log.configure(state="normal")
        self.log.insert("end", text, tag or ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def drain_queues(self):
        try:
            while True:
                self.write_log(self.log_q.get_nowait())
        except queue.Empty:
            pass
        try:
            while True:
                kind, *rest = self.done_q.get_nowait()
                if kind == "job":
                    self.finish(*rest)
                elif kind == "job_progress":
                    frac, text = rest
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=frac * 100)
                    self.status.configure(text=f"{text}… {frac:.0%}")
                elif kind == "preview_progress":
                    token, frac, text = rest
                    if token == self._preview_token:
                        self.preview_bar.configure(value=frac * 100)
                        self.preview_pct.configure(text=f"{frac:.0%}")
                        self.preview_status.configure(text=f"{text}…")
                else:
                    self.show_preview(*rest)
        except queue.Empty:
            pass
        self.after(100, self.drain_queues)

    def run_bg(self, work, done=None, label="Working…"):
        if self.busy:
            return
        self.busy = True
        self.export_btn.state(["disabled"])
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(12)
        self.status.configure(text=label)

        def target():
            from .progress import listen
            result, error = None, None
            with contextlib.redirect_stdout(_QueueWriter(self.log_q)), contextlib.redirect_stderr(_QueueWriter(self.log_q)),                     listen(lambda f, t: self.done_q.put(("job_progress", f, t))):
                try:
                    result = work()
                except SystemExit as e:
                    if e.code:
                        error = "stopped (see the message above)"
                except Exception as e:  # show any failure in the log instead of dying silently
                    error = e
            self.done_q.put(("job", result, error, done))  # Tk may only be touched from the main thread

        threading.Thread(target=target, daemon=True).start()

    def finish(self, result, error, done):
        self.busy = False
        self.export_btn.state(["!disabled"])
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.status.configure(text="Ready")
        if error:
            self.write_log(f"Error: {error}\n")
        elif done:
            done(result)

    # --- preview --------------------------------------------------------------------------
    def voxelizer(self, inst):
        if self._vox is None or self._vox_instance != inst:
            from .assets import Assets
            from .voxelize import Voxelizer
            self._vox = Voxelizer(Assets(instance.asset_paths(inst)), 16)
            self._vox_instance = inst
        return self._vox

    def refresh_preview(self):
        """Render what Export would export, in the background; only the newest request is shown."""
        self._preview_token += 1
        token = self._preview_token
        job = self.preview_job()
        if job is None:
            return
        title, make = job
        self.preview_status.configure(text=f"Loading {title}…")
        self.preview_bar.configure(value=0)
        self.preview_pct.configure(text="0%")
        size = self._preview_render_size()
        inst = self.instance_var.get()  # read Tk state here, not in the thread

        def target():
            from .progress import listen, stage
            try:
                with contextlib.redirect_stdout(io.StringIO()),                         listen(lambda f, t: self.done_q.put(("preview_progress", token, f, t))):
                    first = self._vox is None or self._vox_instance != inst
                    with stage(0, 0.35 if first else 0, "Loading mods"):
                        vox = self.voxelizer(inst)
                    with stage(0.35 if first else 0, 1, "Building model"):
                        img = make(vox, size)
                self.done_q.put(("preview", token, title, img, None))
            except Exception as e:
                self.done_q.put(("preview", token, title, None, e))

        threading.Thread(target=target, daemon=True).start()

    def open_big_preview(self, event=None):
        """Open the current selection in its own large window, rendered at high resolution."""
        job = self.preview_job()
        if job is None:
            return
        title, make = job
        win = tk.Toplevel(self)
        win.title(f"Preview: {title}")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        side = int(min(sw * 0.9, sh * 0.85))
        win.geometry(f"{side}x{side}+{(sw - side) // 2}+{int(sh * 0.03)}")
        win.configure(background="#222226")
        label = tk.Label(win, text=f"Rendering {title}…", background="#222226", foreground="#bbbbbb")
        label.pack(fill="both", expand=True)
        size = min(side, 1200)
        inst = self.instance_var.get()
        result = queue.Queue()

        def work():
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result.put((make(self.voxelizer(inst), size), None))
            except Exception as e:
                result.put((None, e))

        def fit(img):
            w, h = max(50, label.winfo_width() - 8), max(50, label.winfo_height() - 8)
            scale = min(w / img.width, h / img.height)
            shown = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
            buf = io.BytesIO()
            shown.save(buf, "PNG")
            photo = tk.PhotoImage(data=base64.b64encode(buf.getvalue()))
            label.image = photo  # before configure, which can re-enter fit()
            label.configure(image=photo, text="")

        def poll():
            if not win.winfo_exists():
                return
            try:
                img, error = result.get_nowait()
            except queue.Empty:
                win.after(150, poll)
                return
            if error is not None:
                label.configure(text=f"Couldn't preview {title}: {error}")
                return
            fit(img)
            label.bind("<Configure>", lambda e: fit(img))

        threading.Thread(target=work, daemon=True).start()
        poll()

    def _preview_render_size(self):
        """Pixel size to render previews at: the shown area (pictures are square)."""
        w, h = self.preview_label.winfo_width(), self.preview_label.winfo_height()
        return int(max(360, min(1000, h, w)))

    def preview_job(self):
        """(title, function(vox, size) -> PIL image) for the current selection, or None."""
        from . import contraptions, subpreview
        from .cleanup import clean as clean_only
        from .schematic import load_structure
        tab = self.tabs.index(self.tabs.select())
        inst = self.instance_var.get()
        var = getattr(self, "contraption_var", None)
        mode = CONTRAPTION_ARGS.get(var.get(), "world") if var is not None else "world"

        def with_moving(structure, vox, where):
            """Clean up like the export, then add moving contraptions (where -> (list, mapping))."""
            out = clean_only(structure, vox)[0]
            if mode != "none":
                found, to_local = where()
                contraptions.attach(out, found, to_local, mode, exact=True)
            return out

        def clean(structure, vox):  # the preview's clean-up, with contraptions (schematic pieces)
            return (with_moving(structure, vox, lambda: contraptions.for_schematic(path, structure)),)
        if tab == 0:
            name = self.selected_schematic()
            if not name:
                return None
            path = instance.find_schematic(inst, name)
            part = self.part_var.get()
            if part.startswith("Sub-level "):
                n = int(part.split()[1].rstrip(":"))
                return (f"{name}, sub-level {n}", lambda vox, size: subpreview.structure_picture(
                    clean(load_structure(path, n), vox)[0], vox, size, f"{name}: sub-level {n}"))

            def make(vox, size):
                img, n = subpreview.picture(path, vox, size, mode)
                if img is None:
                    img = subpreview.structure_picture(clean(load_structure(path), vox)[0], vox, size, name)
                return img
            return name, make
        if tab == 1:
            try:
                c1 = [int(v.get()) for v in self.corners[0]]
                c2 = [int(v.get()) for v in self.corners[1]]
            except ValueError:
                return None
            from .world import World
            world, dim = instance.find_world(inst, self.world_var.get()), self.dim_var.get()
            return "the region", lambda vox, size: subpreview.structure_picture(
                with_moving(World(world, dim).extract(c1, c2), vox,
                            lambda: contraptions.for_region(world, c1, c2, dim)), vox, size, "region")
        sel = self.ship_tree.selection()
        if not sel:
            return None
        ship_id = str(self.ship_tree.item(sel[0])["values"][0])
        world = instance.find_world(inst, self.ship_world_var.get())

        def make_ship(vox, size):
            ship = next(s for s in sable.ships(world) if s.id == ship_id)
            st = ship.structure()
            return subpreview.structure_picture(with_moving(st, vox, lambda: contraptions.for_ship(world, ship, st)),
                                                vox, size, f"ship {ship_id}")
        return f"ship {ship_id}", make_ship

    def show_preview(self, token, title, img, error):
        if token != self._preview_token:
            return  # an older render finished after a newer request
        if error is not None:
            self.preview_status.configure(text=f"Couldn't preview {title}: {error}")
            self.preview_bar.configure(value=0)
            self.preview_pct.configure(text="")
            return
        self._preview_img = img
        self.preview_status.configure(text=title)
        self.preview_bar.configure(value=100)
        self.preview_pct.configure(text="")
        self._fit_preview()

    def _fit_preview(self):
        img = self._preview_img
        if img is None:
            return
        w = max(50, self.preview_label.winfo_width() - 8)
        h = max(50, self.preview_label.winfo_height() - 8)
        key = (id(img), w, h)
        if getattr(self, "_fit_key", None) == key:
            return  # already showing this picture at this size
        self._fit_key = key
        scale = min(w / img.width, h / img.height)
        shown = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        buf = io.BytesIO()
        shown.save(buf, "PNG")
        photo = tk.PhotoImage(data=base64.b64encode(buf.getvalue()))
        # keep the reference before configure: configure can re-enter this (via <Configure>), and an
        # image Tk still shows must not be garbage-collected
        self.preview_label.image = photo
        self.preview_label.configure(image=photo)

    # --- actions --------------------------------------------------------------------------
    def pick_instance(self):
        d = filedialog.askdirectory(initialdir=self.instance_var.get() or os.path.expanduser("~"))
        if d:
            self.instance_var.set(d)
            self.load_instance()

    def load_instance(self):
        inst = self.instance_var.get()
        try:
            self.schematics = instance.list_schematics(inst)
            worlds = instance.list_worlds(inst)
            version = instance.minecraft_version(inst)
        except Exception as e:
            messagebox.showerror("Minecraft-Modded2Model", f"That doesn't look like a Prism instance:\n{e}")
            return
        self.fill_schematics()
        for box, var in ((self.world_box, self.world_var), (self.ship_world_box, self.ship_world_var)):
            box["values"] = worlds
            if worlds and var.get() not in worlds:
                var.set(worlds[0])
        self.settings["instance"] = inst
        save_settings(self.settings)
        self.write_log(f"Loaded instance: Minecraft {version}, {len(self.schematics)} schematics, "
                       f"{len(worlds)} worlds\n")
        self.build_block_map(inst)

    def build_block_map(self, inst):
        """In the background, build the map colour of every block in the instance (all mods),
        so world maps draw instantly. Skipped when the saved table is already up to date."""
        building = getattr(self, "_block_map_building", set())
        self._block_map_building = building
        key = os.path.normcase(os.path.abspath(inst))
        if key in building:
            return
        building.add(key)
        log = self.log_q

        def work():
            from . import blocktable
            shown = [-1]

            def progress(frac, text):
                step = int(frac * 4)  # a line at 25%, 50%, 75% and when done
                if frac >= 1 or step > shown[0]:
                    shown[0] = step
                    log.put(text + "\n")
            try:
                blocktable.build(inst, instance.asset_paths(inst), progress)
            except Exception as e:
                log.put(f"Couldn't build the block map (world maps will work it out as they go): {e}\n")
            finally:
                building.discard(key)

        threading.Thread(target=work, daemon=True).start()

    def fill_schematics(self):
        q = self.search_var.get().lower()
        self.schem_list.delete(0, "end")
        for s in self.schematics:
            if q in s.lower():
                self.schem_list.insert("end", s)

    def selected_schematic(self):
        sel = self.schem_list.curselection()
        return self.schem_list.get(sel[0]) if sel else None

    def schematic_selected(self):
        """List the selected schematic's Aeronautics sub-levels as parts to export, then preview."""
        name = self.selected_schematic()
        values = [MAIN_PART]
        if name:
            try:
                from .schematic import load_structure
                s = load_structure(instance.find_schematic(self.instance_var.get(), name))
                for i, sl in enumerate(s.sub_levels):
                    note = "merged into main" if i in s.merged else "rotated, export separately"
                    values.append(f"Sub-level {i}: {sl.size[0]}x{sl.size[1]}x{sl.size[2]}, "
                                  f"{len(sl.blocks)} blocks ({note})")
            except Exception:
                pass
        self.part_box["values"] = values
        self.part_var.set(MAIN_PART)
        self.refresh_preview()

    def pick_on_map(self):
        from .mapwindow import MapWindow
        inst = self.instance_var.get()
        try:
            world = instance.find_world(inst, self.world_var.get())
        except Exception as e:
            messagebox.showerror("Minecraft-Modded2Model", f"Pick a world first:\n{e}")
            return
        try:  # start from the coordinates already typed in, if any
            box = tuple(tuple(int(v.get()) for v in c) for c in self.corners)
        except ValueError:
            box = None

        def use(lo, hi):
            for vars_, p in zip(self.corners, (lo, hi)):
                for var, v in zip(vars_, p):
                    var.set(str(v))
            self.refresh_preview()

        MapWindow(self, world, self.dim_var.get(), instance.asset_paths(inst), use, box)

    def around_player(self):
        try:
            world = instance.find_world(self.instance_var.get(), self.world_var.get())
            pos = nbt.load(os.path.join(world, "level.dat"))["Data"]["Player"]["Pos"]
            r = int(self.radius_var.get())
        except Exception as e:
            messagebox.showerror("Minecraft-Modded2Model", f"Couldn't read the player position:\n{e}")
            return
        x, y, z = (int(v) for v in pos)
        for var, v in zip(self.corners[0], (x - r, y - r // 2, z - r)):
            var.set(str(v))
        for var, v in zip(self.corners[1], (x + r, y + r, z + r)):
            var.set(str(v))
        self.refresh_preview()

    def find_ships(self):
        inst, name = self.instance_var.get(), self.ship_world_var.get()

        def work():
            world = instance.find_world(inst, name)
            rows = []
            for ship in sable.ships(world):
                s = ship.structure()
                if s.blocks:
                    rows.append((ship.id, f"{s.size[0]}x{s.size[1]}x{s.size[2]}", len(s.blocks)))
            return sorted(rows, key=lambda r: -r[2])

        def done(rows):
            self.ship_tree.delete(*self.ship_tree.get_children())
            for r in rows:
                self.ship_tree.insert("", "end", values=r)
            self.write_log(f"Found {len(rows)} ships in {name}\n")

        self.run_bg(work, done, "Finding ships…")

    def export(self):
        inst = self.instance_var.get()
        args = ["--instance", inst]
        tab = self.tabs.index(self.tabs.select())
        if tab == 0:
            name = self.selected_schematic()
            if not name:
                return messagebox.showinfo("Minecraft-Modded2Model", "Pick a schematic first.")
            args.insert(0, name)
            if self.part_var.get().startswith("Sub-level "):
                n = self.part_var.get().split()[1].rstrip(":")
                args += ["--sub-level", n]
                name += f"_sub{n}"
        elif tab == 1:
            try:
                c1 = [str(int(v.get())) for v in self.corners[0]]
                c2 = [str(int(v.get())) for v in self.corners[1]]
            except ValueError:
                return messagebox.showinfo("Minecraft-Modded2Model", "Fill in both corners with whole numbers.")
            args += ["-w", self.world_var.get(), "--dimension", self.dim_var.get(), "--from", *c1, "--to", *c2]
            name = self.world_var.get().replace(" ", "_") + "_region"
        else:
            sel = self.ship_tree.selection()
            if not sel:
                return messagebox.showinfo("Minecraft-Modded2Model", "Click 'Find ships' and pick one first.")
            ship_id = self.ship_tree.item(sel[0])["values"][0]
            args += ["-w", self.ship_world_var.get(), "--ship", str(ship_id)]
            name = "ship_" + str(ship_id).replace(":", "_").replace(".", "_")

        ext = self.fmt_var.get()
        out = filedialog.asksaveasfilename(
            defaultextension="." + ext, initialfile=f"{name}.{ext}",
            initialdir=self.settings.get("out_dir", os.path.expanduser("~")),
            filetypes=[("3D model", f"*.{ext}")])
        if not out:
            return

        o = {k: v.get() for k, v in self.opts.items()}
        glb = ext == "glb"
        args += ["-o", out, "--res", o["res"], "--missing", self.missing_var.get()]
        contraption_var = getattr(self, "contraption_var", None)  # the classic window's setting
        if contraption_var is not None:
            args += ["--contraptions", CONTRAPTION_ARGS[contraption_var.get()]]
            self.settings["contraptions"] = contraption_var.get()
        if not self.backup_var.get():
            args.append("--keep-backup-blocks")
        if not self.vs_var.get():
            args.append("--keep-vs-blocks")
        if glb:
            args += [flag for var, flag in ((self.merge_var, "--merge"), (self.glass_var, "--glass-backface-cull"))
                     if var.get()]
        else:
            args += ["--interior", {"Keep": "keep", "Remove (hollow)": "hollow",
                                    "Make solid blocks": "solid"}[self.interior_var.get()]]
            units = {"Millimeters": "mm", "Centimeters": "cm", "Meters": "m", "Inches": "in"}[self.units_var.get()]
            args += ["--units", units, "--min-thickness-mm", o["min_thick"]]
            if self.size_var.get() == "height":
                args += ["--height-cm", o["height_cm"]]
            else:
                args += ["--block-mm", o["block_mm"]]
            if not self.zup_var.get():
                args.append("--y-up")
            if self.fill_var.get():
                args.append("--fill-bubbles")
                args += ["--seal-entrances"] * self.seal_var.get() + ["--seal-tunnels"] * self.tunnel_var.get()
            if self.connect_var.get():
                args.append("--connect-parts")
                args += ["--connect-corners"] * self.corner_var.get() + ["--weld-all-edges"] * self.weld_var.get()
            if self.float_var.get():
                args += ["--delete-floating", o["floater_count"]]
            if self.hollowb_var.get():
                args += ["--hollow-bottom-mm", o["hollow_mm"]] + ["--superhollow"] * self.super_var.get()
            if self.melt_var.get():
                args.append("--melt-snow")
            if not self.lesser_var.get():
                args.append("--no-lesser")
            elif self.fatten_var.get():
                args.append("--fatten")
            if self.voxel_var.get():
                args.append("--voxel")

        flags = {"merge": self.merge_var, "glass_cull": self.glass_var, 
                 "remove_backups": self.backup_var, "replace_vs": self.vs_var, "z_up": self.zup_var,
                 "fill_bubbles": self.fill_var, "seal_entrances": self.seal_var, "seal_tunnels": self.tunnel_var,
                 "connect_parts": self.connect_var, "connect_corners": self.corner_var, "weld_all": self.weld_var,
                 "delete_floating": self.float_var, "hollow_bottom": self.hollowb_var,
                 "superhollow": self.super_var, "melt_snow": self.melt_var, "export_lesser": self.lesser_var,
                 "fatten": self.fatten_var, "voxel": self.voxel_var}
        self.settings.update(o, out_dir=os.path.dirname(out), format=ext, missing=self.missing_var.get(),
                             units=self.units_var.get(), size_mode=self.size_var.get(),
                             interior=self.interior_var.get(),
                             **{k: v.get() for k, v in flags.items()})
        save_settings(self.settings)
        self.write_log(f"\nExporting {name}…\n")
        self.run_bg(lambda: cli_main(args), lambda _: self.write_log(f"Done: {out}\n"), f"Exporting {name}…")


def main():
    try:  # sharp text on scaled Windows displays instead of a blurry stretched window
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()


if __name__ == "__main__":
    main()
