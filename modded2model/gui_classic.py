"""The classic (plain, light) window: python -m modded2model.gui_classic

Same features and saved settings as the main window (gui.py); only the look and layout differ.
"""
import os
import queue
import tkinter as tk
from tkinter import ttk

from . import __version__, gui
from .gui import DEFAULT_INSTANCE, MAIN_PART, load_settings


class ClassicApp(gui.App):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title(f"Minecraft-Modded2Model v{__version__}")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1500, int(sw * 0.9))}x{min(1000, int(sh * 0.88))}+{int(sw * 0.03)}+{int(sh * 0.02)}")
        self.minsize(1100, 700)
        self.settings = load_settings()
        self.log_q = queue.Queue()
        self.done_q = queue.Queue()
        self.busy = False
        self.schematics = []
        self._vox = None  # shared Voxelizer for previews (loaded once per instance)
        self._vox_instance = None
        self._preview_token = 0
        self._preview_img = None

        style = ttk.Style(self)
        style.configure("Big.TButton", padding=(16, 6))
        style.configure("Hint.TLabel", foreground="gray")

        self._build_instance_row()
        # Export and the log run along the bottom, under both panes
        self._build_bottom()
        # left: what to export and the options; right: a square preview
        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=(4, 0))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=0)
        body.add(right, weight=1)
        self._left, self._right = left, right
        self._build_options()  # packed from the bottom up, so the build list takes what's left
        self._build_sources(left)
        preview = ttk.LabelFrame(right, text="Preview")
        preview.pack(fill="both", expand=True)
        self._build_preview(preview)
        self._sash_set = False
        body.bind("<Configure>", lambda e: self._square_preview(body))

        self.after(100, self.drain_queues)
        if os.path.isdir(self.instance_var.get()):
            self.load_instance()

    # --- layout ---------------------------------------------------------------------------
    def _square_preview(self, body):
        """Size the panes so the preview is square (its height decides its width)."""
        if self._sash_set:
            return  # only once, so the user can still drag the divider
        self._sash_set = True
        self.after(300, lambda: self._fit_sash(body))

    def _fit_sash(self, body):
        self.update_idletasks()
        w = body.winfo_width()
        side = self._stage.winfo_height()  # the preview's height decides its width
        body.sashpos(0, max(self._left.winfo_reqwidth(), w - side - 24))

    def _build_instance_row(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=(8, 2))
        ttk.Label(top, text="Prism instance:").pack(side="left")
        self.instance_var = tk.StringVar(value=self.settings.get("instance", DEFAULT_INSTANCE))
        ttk.Entry(top, textvariable=self.instance_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(top, text="Browse…", command=self.pick_instance).pack(side="left")
        ttk.Button(top, text="Load", command=self.load_instance).pack(side="left", padx=(4, 0))

    def _build_sources(self, parent):
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(fill="both", expand=True)
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self.refresh_preview())

        # Schematic
        sch = ttk.Frame(self.tabs, padding=4)
        self.tabs.add(sch, text="Schematic")
        r = ttk.Frame(sch)
        r.pack(fill="x")
        ttk.Label(r, text="Search:").pack(side="left")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *a: self.fill_schematics())
        ttk.Entry(r, textvariable=self.search_var).pack(side="left", fill="x", expand=True, padx=4)
        lf = ttk.Frame(sch)
        lf.pack(fill="both", expand=True, pady=4)
        self.schem_list = tk.Listbox(lf, exportselection=False, activestyle="none", height=4)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.schem_list.yview)
        self.schem_list.configure(yscrollcommand=sb.set)
        self.schem_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.schem_list.bind("<<ListboxSelect>>", lambda e: self.schematic_selected())
        ttk.Label(sch, text="Part to export:").pack(anchor="w")
        self.part_var = tk.StringVar(value=MAIN_PART)
        self.part_box = ttk.Combobox(sch, textvariable=self.part_var, values=[MAIN_PART], state="readonly")
        self.part_box.pack(fill="x")
        self.part_box.bind("<<ComboboxSelected>>", lambda e: self.refresh_preview())
        ttk.Label(sch, text="Aeronautics sub-levels are listed here; the preview shows which is which.",
                  style="Hint.TLabel", wraplength=420).pack(anchor="w", pady=(2, 0))

        # World region
        reg = ttk.Frame(self.tabs, padding=4)
        self.tabs.add(reg, text="World region")
        grid = ttk.Frame(reg)
        grid.pack(fill="x")
        ttk.Label(grid, text="World:").grid(row=0, column=0, sticky="w", pady=2)
        self.world_var = tk.StringVar()
        self.world_box = ttk.Combobox(grid, textvariable=self.world_var, state="readonly")
        self.world_box.grid(row=0, column=1, columnspan=3, sticky="ew", padx=4)
        ttk.Label(grid, text="Dimension:").grid(row=1, column=0, sticky="w", pady=2)
        self.dim_var = tk.StringVar(value="overworld")
        ttk.Combobox(grid, textvariable=self.dim_var, values=["overworld", "nether", "end"]).grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=4)
        self.corners = []
        for i, label in enumerate(("From x y z:", "To x y z:")):
            ttk.Label(grid, text=label).grid(row=2 + i, column=0, sticky="w", pady=2)
            vars_ = [tk.StringVar() for _ in range(3)]
            for j, v in enumerate(vars_):
                ttk.Entry(grid, textvariable=v, width=9).grid(row=2 + i, column=1 + j, padx=2, sticky="w")
            self.corners.append(vars_)
        grid.columnconfigure(3, weight=1)
        r = ttk.Frame(reg)
        r.pack(fill="x", pady=(6, 0))
        ttk.Button(r, text="Pick on map…", command=self.pick_on_map).pack(side="left")
        ttk.Label(r, text="drag a box on a top-down map of the world", style="Hint.TLabel").pack(
            side="left", padx=8)
        r = ttk.Frame(reg)
        r.pack(fill="x", pady=6)
        ttk.Button(r, text="Use area around player", command=self.around_player).pack(side="left")
        ttk.Label(r, text="radius").pack(side="left", padx=(8, 2))
        self.radius_var = tk.StringVar(value="16")
        ttk.Entry(r, textvariable=self.radius_var, width=5).pack(side="left")
        ttk.Button(r, text="Preview", command=self.refresh_preview).pack(side="right")
        ttk.Label(reg, text="Tip: press F3 in game to read block coordinates.", style="Hint.TLabel").pack(anchor="w")

        # Aeronautics ship
        shp = ttk.Frame(self.tabs, padding=4)
        self.tabs.add(shp, text="Aeronautics ship")
        r = ttk.Frame(shp)
        r.pack(fill="x")
        ttk.Label(r, text="World:").pack(side="left")
        self.ship_world_var = tk.StringVar()
        self.ship_world_box = ttk.Combobox(r, textvariable=self.ship_world_var, state="readonly")
        self.ship_world_box.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(r, text="Find ships", command=self.find_ships).pack(side="left")
        cols = ("id", "size", "blocks")
        self.ship_tree = ttk.Treeview(shp, columns=cols, show="headings", selectmode="browse", height=5)
        for c, w in zip(cols, (150, 110, 70)):
            self.ship_tree.heading(c, text=c.title())
            self.ship_tree.column(c, width=w, anchor="w")
        self.ship_tree.pack(fill="both", expand=True, pady=4)
        self.ship_tree.bind("<<TreeviewSelect>>", lambda e: self.refresh_preview())

    def _build_preview(self, parent):
        stage_ = ttk.Frame(parent)
        stage_.pack(fill="both", expand=True, padx=4, pady=4)
        self._stage = stage_
        self.preview_label = tk.Label(stage_, anchor="center", background="#222226", foreground="#bbbbbb",
                                      text="Pick something on the left to see it here.", cursor="hand2")

        def keep_square(e):
            side = max(50, min(e.width, e.height))
            self.preview_label.place(x=(e.width - side) // 2, y=(e.height - side) // 2, width=side, height=side)
        stage_.bind("<Configure>", keep_square)
        bar = ttk.Frame(parent)
        bar.pack(fill="x", padx=6, pady=(0, 6))
        self.preview_status = ttk.Label(bar, text="Click the preview to open it large.", style="Hint.TLabel")
        self.preview_status.pack(side="left")
        self.preview_pct = ttk.Label(bar, text="", style="Hint.TLabel", width=5, anchor="e")
        self.preview_pct.pack(side="right")
        self.preview_bar = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self.preview_bar.pack(fill="x", padx=6, pady=(0, 6))
        self.preview_label.bind("<Configure>", lambda e: self._fit_preview())
        self.preview_label.bind("<Button-1>", self.open_big_preview)

    def _build_options(self):
        s = self.settings
        self.opts = {}
        self._deps = []  # (checkbox var, widgets enabled only when it's on)

        def num(frame, key, default, width=6):
            v = tk.StringVar(value=str(s.get(key, default)))
            self.opts[key] = v
            return ttk.Entry(frame, textvariable=v, width=width)

        def flag(frame, key, label, default):
            v = tk.BooleanVar(value=s.get(key, default))
            cb = ttk.Checkbutton(frame, text=label, variable=v, command=self.update_option_states)
            return v, cb

        box = ttk.Frame(self._left)
        box.pack(side="bottom", fill="x", pady=(6, 0))
        out = ttk.Frame(box)
        out.pack(fill="x")
        ttk.Label(out, text="Format:").pack(side="left")
        self.fmt_var = tk.StringVar(value=s.get("format", "stl"))
        for val, label in (("stl", "STL (3D print)"), ("3mf", "3MF (3D print)"), ("glb", "GLB (Blender, textured)")):
            ttk.Radiobutton(out, text=label, value=val, variable=self.fmt_var,
                            command=self.update_option_states).pack(side="left", padx=(6, 0))
        out = ttk.Frame(box)
        out.pack(fill="x", pady=(2, 0))
        ttk.Label(out, text="Detail").pack(side="left")
        num(out, "res", "0", 5).pack(side="left", padx=4)
        ttk.Label(out, text="voxels/block, 0 = auto", style="Hint.TLabel").pack(side="left")

        self.opt_tabs = ttk.Notebook(box)
        self.opt_tabs.pack(fill="x", pady=4)

        bl = ttk.Frame(self.opt_tabs, padding=8)
        self.merge_var, self.merge_cb = flag(bl, "merge", "Connect meshes, cull touching faces", True)
        self.merge_cb.grid(row=0, column=0, sticky="w")
        self.glass_var, self.glass_cb = flag(bl, "glass_cull", "One-sided glass (backface culling)", False)
        self.glass_cb.grid(row=1, column=0, sticky="w")

        cl = ttk.Frame(self.opt_tabs, padding=8)
        self.backup_var, cb = flag(cl, "remove_backups", "Remove backup blocks", True)
        cb.grid(row=1, column=0, sticky="w")
        self.vs_var, cb = flag(cl, "replace_vs", "Replace Valkyrien Skies blocks", True)
        cb.grid(row=2, column=0, sticky="w")
        r = ttk.Frame(cl)
        r.grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(r, text="Blocks with no model:").pack(side="left")
        self.missing_var = tk.StringVar(value=s.get("missing", "cube"))
        ttk.Combobox(r, textvariable=self.missing_var, values=("cube", "skip"), width=6,
                     state="readonly").pack(side="left", padx=4)
        r = ttk.Frame(cl)
        r.grid(row=4, column=0, sticky="w", pady=2)
        ttk.Label(r, text="Moving contraptions (bearings, propellers, windmills):").pack(side="left")
        self.contraption_var = tk.StringVar(value=s.get("contraptions", gui.CONTRAPTION_MODES[0]))
        ttk.Combobox(r, textvariable=self.contraption_var, values=gui.CONTRAPTION_MODES, width=22,
                     state="readonly").pack(side="left", padx=4)

        pr = ttk.Frame(self.opt_tabs, padding=8)
        self.print_frame, self.blender_frame = pr, bl
        self.opt_tabs.add(pr, text="3D printing")
        self.opt_tabs.add(bl, text="Blender")
        self.opt_tabs.add(cl, text="Clean-up")
        g = 0

        def line():
            nonlocal g
            f = ttk.Frame(pr)
            f.grid(row=g, column=0, sticky="w", pady=1)
            g += 1
            return f

        f = line()
        self.zup_var, cb = flag(f, "z_up", "Make Z the up direction instead of Y", True)
        cb.pack(side="left")

        f = line()
        self.size_var = tk.StringVar(value=s.get("size_mode", "block"))
        ttk.Radiobutton(f, text="Make the model", value="height", variable=self.size_var,
                        command=self.update_option_states).pack(side="left")
        self.height_entry = num(f, "height_cm", "5")
        self.height_entry.pack(side="left", padx=4)
        ttk.Label(f, text="cm in height").pack(side="left")

        f = line()
        ttk.Radiobutton(f, text="Make each block", value="block", variable=self.size_var,
                        command=self.update_option_states).pack(side="left")
        self.block_entry = num(f, "block_mm", "5")
        self.block_entry.pack(side="left", padx=4)
        ttk.Label(f, text="mm").pack(side="left")
        ttk.Label(f, text="Units:").pack(side="left", padx=(16, 4))
        self.units_var = tk.StringVar(value=s.get("units", "Millimeters"))
        ttk.Combobox(f, textvariable=self.units_var, values=("Millimeters", "Centimeters", "Meters", "Inches"),
                     width=12, state="readonly").pack(side="left")

        f = line()
        self.fill_var, cb = flag(f, "fill_bubbles", "Fill air bubbles:", True)
        cb.pack(side="left")
        self.seal_var, seal = flag(f, "seal_entrances", "Seal off entrances", False)
        seal.pack(side="left", padx=(12, 0))
        self.tunnel_var, tun = flag(f, "seal_tunnels", "Fill tunnels in the base", False)
        tun.pack(side="left", padx=(12, 0))
        self._deps.append((self.fill_var, [seal, tun]))

        f = line()
        self.connect_var, cb = flag(f, "connect_parts", "Connect parts sharing an edge:", True)
        cb.pack(side="left")
        self.corner_var, corner = flag(f, "connect_corners", "Corner tips", True)
        corner.pack(side="left", padx=(12, 0))
        self.weld_var, weld = flag(f, "weld_all", "Weld all edges", False)
        weld.pack(side="left", padx=(12, 0))
        self._deps.append((self.connect_var, [corner, weld]))

        f = line()
        self.float_var, cb = flag(f, "delete_floating", "Delete floating trees and parts smaller than", True)
        cb.pack(side="left")
        fe = num(f, "floater_count", "16", 5)
        fe.pack(side="left", padx=4)
        ttk.Label(f, text="blocks").pack(side="left")
        self._deps.append((self.float_var, [fe]))

        f = line()
        self.hollowb_var, cb = flag(f, "hollow_bottom", "Hollow out the bottom, wall", True)
        cb.pack(side="left")
        he = num(f, "hollow_mm", "2", 5)
        he.pack(side="left", padx=4)
        ttk.Label(f, text="mm thick").pack(side="left")
        self.super_var, sup = flag(f, "superhollow", "Superhollow", True)
        sup.pack(side="left", padx=(12, 0))
        self._deps.append((self.hollowb_var, [he, sup]))

        f = line()
        self.melt_var, cb = flag(f, "melt_snow", "Melt snow blocks (useful with fill, seal and hollow)", False)
        cb.pack(side="left")

        f = line()
        ttk.Label(f, text="Non-solid blocks inside (cogs, gearboxes...):").pack(side="left")
        old = "Remove (hollow)" if s.get("hollow_interior") else "Keep"
        self.interior_var = tk.StringVar(value=s.get("interior", old))
        ttk.Combobox(f, textvariable=self.interior_var, values=("Keep", "Remove (hollow)", "Make solid blocks"),
                     width=17, state="readonly").pack(side="left", padx=4)

        f = line()
        self.lesser_var, cb = flag(f, "export_lesser", "Export lesser, detailed blocks:", True)
        cb.pack(side="left")
        self.fatten_var, fat = flag(f, "fatten", "Fatten them (safer printing)", False)
        fat.pack(side="left", padx=(12, 0))
        self._deps.append((self.lesser_var, [fat]))

        f = line()
        ttk.Label(f, text="Min thickness").pack(side="left")
        num(f, "min_thick", "0", 5).pack(side="left", padx=4)
        ttk.Label(f, text="mm (0 = exact)", style="Hint.TLabel").pack(side="left")
        self.voxel_var, cb = flag(f, "voxel", "Blocky voxel export", False)
        cb.pack(side="left", padx=(12, 0))
        self.update_option_states()

    def _build_bottom(self):
        lf = ttk.Frame(self)
        lf.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
        self.log = tk.Text(lf, height=4, width=40, state="disabled", wrap="word", relief="flat", background="#f4f4f4")
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.log.tag_configure("done", foreground="#2e7d32")
        self.log.tag_configure("error", foreground="#c62828")
        self.log.tag_configure("head", foreground="#8a5a00")
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=8, pady=4)
        self.export_btn = ttk.Button(bottom, text="Export…", style="Big.TButton", command=self.export)
        self.export_btn.pack(side="right")
        self.progress = ttk.Progressbar(bottom, mode="determinate", length=240)
        self.progress.pack(side="right", padx=8)
        self.status = ttk.Label(bottom, text="", style="Hint.TLabel")
        self.status.pack(side="left")


def main():
    try:  # sharp text on scaled Windows displays instead of a blurry stretched window
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    ClassicApp().mainloop()


if __name__ == "__main__":
    main()
