"""Command line: python -m modded2model truck1 --instance <Prism instance folder>"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

from . import __version__, instance, mesh, printprep, sable
from .assets import Assets
from .schematic import load_structure
from .exact import ExactModels, assemble
from .progress import stage
from .voxelize import Voxelizer, rasterize_mesh
from .world import World


def main(argv=None):
    p = argparse.ArgumentParser(prog="modded2model", description="Export Minecraft structures (modded too) as printable models.")
    p.add_argument("--version", action="version", version=f"Minecraft-Modded2Model v{__version__}")
    p.add_argument("schematic", nargs="?",
                   help="structure file (.nbt), or with --instance just its name in the schematics folder")
    p.add_argument("-i", "--instance", help="Prism/MultiMC instance folder: finds mods, resource packs, "
                   "the vanilla jar and schematics automatically")
    p.add_argument("--list", action="store_true", help="list the instance's schematics and worlds, and exit")
    p.add_argument("-w", "--world", help="world save: a name in the instance's saves folder, or a path")
    p.add_argument("--dimension", default="overworld", help="overworld, nether, end, or a modded id like mod:sky")
    p.add_argument("--from", dest="corner1", nargs=3, type=int, metavar=("X", "Y", "Z"), help="region corner")
    p.add_argument("--to", dest="corner2", nargs=3, type=int, metavar=("X", "Y", "Z"), help="opposite region corner")
    p.add_argument("--list-ships", action="store_true", help="list Create Aeronautics (Sable) ships in --world and exit")
    p.add_argument("--ship", help="export a Sable ship by id from --list-ships")
    p.add_argument("--contraptions", choices=("world", "snap", "assembled", "none"), default="world",
                   help="moving Create contraptions (bearings, propellers, windmills): 'world' = at their "
                        "saved angle (exact in GLB, nearest quarter turn in prints), 'snap' = nearest quarter "
                        "turn, 'assembled' = as built (angle 0), 'none' = leave them out")
    p.add_argument("--sub-level", type=int, metavar="N",
                   help="for schematics holding Aeronautics ships: export ship N (0-based; default: the "
                        "normal blocks, or the largest ship if there are none)")
    p.add_argument("--preview-sub-levels", metavar="PNG",
                   help="write a picture of the schematic's Aeronautics sub-levels (numbered and coloured, "
                        "placed where they were saved) and exit")
    p.add_argument("-a", "--assets", action="append", default=[],
                   help="resource pack, mod jar, mods folder or vanilla client jar; repeat, highest priority first")
    p.add_argument("-o", "--output", help="output .stl or .3mf for printing, or .glb (textured) for Blender "
                   "(default: <schematic name>.stl)")
    p.add_argument("--block-mm", type=float, default=5.0, help="size of one block in millimetres (default 5)")
    p.add_argument("--res", type=int, default=0,
                   help="voxels per block edge: 16 = full model detail; default picks the most that fits --max-cells")
    p.add_argument("--max-cells", type=int, default=600, help="cap on grid cells along the longest side (default 600)")
    p.add_argument("--min-thickness-mm", type=float, default=0,
                   help="thicken parts thinner than this for printing (0 = off: keep the exact model detail; "
                        "flat elements always get 1 px)")
    p.add_argument("--hollow-mm", type=float, default=0, help="hollow solid regions leaving this wall thickness (0 = off)")
    p.add_argument("--no-fill", action="store_true", help="keep sealed internal rooms empty")
    p.add_argument("--cull-interior", action="store_true",
                   help="remove faces hidden inside, using the interior fill: outside air floods in "
                        "(through glass, grates and open doors too), every space it can't reach counts as "
                        "filled, and faces looking into filled space are dropped. Blender: per face; print: "
                        "a part is dropped only when none of its faces show")
    p.add_argument("--solidify", type=float, default=0, metavar="BLOCKS",
                   help="fill the whole interior (fill air bubbles + seal off entrances): doorways and "
                        "windows up to this many blocks wide are sealed while filling (0 = off)")
    p.add_argument("--min-piece-mm3", type=float, default=1.0, help="drop loose pieces smaller than this volume")
    p.add_argument("--largest-only", action="store_true", help="keep only the largest connected piece")
    mw = p.add_argument_group("3D print options")
    mw.add_argument("--fill-bubbles", action="store_true", help="Fill air bubbles: fill enclosed air")
    mw.add_argument("--seal-entrances", action="store_true",
                    help="Seal off entrances: one-block doorways and windows don't let outside air in")
    mw.add_argument("--seal-tunnels", action="store_true",
                    help="Fill in isolated tunnels in model's base: air only gets in from the top of the box")
    mw.add_argument("--connect-parts", action="store_true",
                    help="Connect parts sharing an edge: weld blocks touching only along an edge")
    mw.add_argument("--connect-corners", action="store_true", help="Connect corner tips (with --connect-parts)")
    mw.add_argument("--weld-all-edges", action="store_true", help="Weld all shared edges, also within one part")
    mw.add_argument("--delete-floating", type=int, default=0, metavar="N",
                    help="Delete floating objects: trees and parts smaller than N blocks (0 = off)")
    mw.add_argument("--hollow-bottom-mm", type=float, default=0, metavar="MM",
                    help="Hollow out bottom of model, making the wall this many mm thick (0 = off)")
    mw.add_argument("--superhollow", action="store_true", help="Superhollow (with --hollow-bottom-mm)")
    mw.add_argument("--melt-snow", action="store_true", help="Melt snow blocks after sealing and filling")
    mw.add_argument("--no-lesser", action="store_true",
                    help="leave out lesser, detailed blocks (torches, levers, rails, plants...)")
    mw.add_argument("--fatten", action="store_true", help="Fatten lesser blocks: thin parts at least 2 px")
    mw.add_argument("--y-up", action="store_true", help="keep Minecraft's Y up (default: Z up for printers)")
    mw.add_argument("--height-cm", type=float, default=0,
                    help="make the model this many cm tall (instead of --block-mm)")
    mw.add_argument("--units", choices=("mm", "cm", "m", "in"), default="mm", help="model's units in the file")
    p.add_argument("--merge", action="store_true",
                   help="Blender export: connect the meshes (weld shared vertices) and cull faces hidden "
                        "against neighbouring blocks or touching parts")
    p.add_argument("--missing", choices=("cube", "skip"), default="cube",
                   help="blocks with no model (unknown, or drawn only by code): full cube (default) or leave out")
    p.add_argument("--interior", choices=("keep", "hollow", "solid"), default="keep",
                   help="blocks inside the model that aren't plain solid blocks (cogwheels, gearboxes, shafts, "
                        "machines, furniture): keep them, remove them (hollow), or replace each with a solid "
                        "block (solid). Inside = not touching outside air once every block is a wall and "
                        "doorways are sealed (the interior fill)")
    p.add_argument("--hollow-interior", action="store_true",
                   help="remove blocks inside the model that aren't plain solid blocks (cogwheels, gearboxes, "
                        "shafts, machines, furniture), leaving the interior hollow. Inside = not touching outside "
                        "air when every block is a wall and doorways/windows are sealed (the interior fill)")
    p.add_argument("--keep-backup-blocks", action="store_true",
                   help="keep Drive-by-Wire backup blocks (removed by default)")
    p.add_argument("--keep-vs-blocks", action="store_true",
                   help="keep Valkyrien Skies blocks as they are (by default helms become stairs, balloons "
                        "Aeronautics envelopes and floaters brown wool)")
    p.add_argument("--glass-backface-cull", action="store_true",
                   help="Blender export: glass materials are single-sided (backface culling), so you see "
                        "through panes to what's behind instead of the pane's far side")
    p.add_argument("--voxel", action="store_true",
                   help="blocky export from the voxel grid (default: each block's exact model geometry)")
    p.add_argument("--overrides", help="JSON file: block name -> 'full' | 'skip' | {'boxes': [[x1,y1,z1,x2,y2,z2], ...]}")
    a = p.parse_args(argv)

    if a.instance:
        if a.list:
            print("Schematics:\n  " + "\n  ".join(instance.list_schematics(a.instance)))
            print("Worlds:\n  " + "\n  ".join(instance.list_worlds(a.instance)))
            return 0
        if a.world:
            a.world = instance.find_world(a.instance, a.world)
        if a.schematic:
            a.schematic = instance.find_schematic(a.instance, a.schematic)
        a.assets += instance.asset_paths(a.instance)
    if a.list_ships:
        if not a.world:
            p.error("--list-ships needs --world")
        for ship in sable.ships(a.world):
            s = ship.structure()
            x, y, z = ship.position
            print(f"{ship.id:16} {s.size[0]}x{s.size[1]}x{s.size[2]} blocks, {len(s.blocks):5} non-air, "
                  f"at {x:.0f} {y:.0f} {z:.0f}")
        return 0
    if not a.assets:
        p.error("give --instance or -a asset sources")

    if a.preview_sub_levels:
        if not a.schematic:
            p.error("--preview-sub-levels needs a schematic")
        from . import subpreview
        n = subpreview.render(a.schematic, Voxelizer(Assets(a.assets), 16), a.preview_sub_levels)
        print(f"Wrote {a.preview_sub_levels}: {n} sub-level(s)" if n else "This schematic has no sub-levels.")
        return 0

    t0 = time.time()
    moving = None
    if a.ship:
        if not a.world:
            p.error("--ship needs --world")
        match = [s for s in sable.ships(a.world) if s.id == a.ship]
        if not match:
            p.error(f"no ship {a.ship}; see --list-ships")
        structure = match[0].structure()
        default_name = "ship_" + a.ship.replace(":", "_").replace(".", "_")
        moving = _ship_contraptions(a, match[0], structure)
    elif a.world:
        if not (a.corner1 and a.corner2):
            p.error("exporting from a world needs --from X Y Z and --to X Y Z (or --ship)")
        structure = World(a.world, a.dimension).extract(a.corner1, a.corner2)
        default_name = os.path.basename(os.path.normpath(a.world)).replace(" ", "_") + "_region"
        moving = _region_contraptions(a)
    elif a.schematic:
        structure = load_structure(a.schematic, a.sub_level)
        moving = _schematic_contraptions(a, structure)
        if structure.sub_levels:
            print("Aeronautics ships in this schematic (use --sub-level N to pick one):")
            for i, sl in enumerate(structure.sub_levels):
                mark = ("  <- exporting" if sl is structure else "  <- merged onto the grid" if i in structure.merged
                        else "  <- rotated, left out (use --sub-level)" if i in structure.skipped else "")
                print(f"  {i}: {sl.size[0]}x{sl.size[1]}x{sl.size[2]} blocks, {len(sl.blocks)} non-air{mark}")
        default_name = os.path.splitext(os.path.basename(a.schematic))[0]
    else:
        p.error("give a schematic, or --world with --from/--to or --ship")
    if not structure.blocks:
        p.error("nothing but air there")
    if not a.output:
        a.output = default_name + ".stl"

    longest = max(structure.size)
    # Model detail is on a 1/16-block grid, so only resolutions that divide 16 keep
    # parts at their true size (others round 2 px layers and 4 px shafts unevenly).
    res = a.res or next(r for r in (16, 8, 4, 2, 1) if r == 1 or r * longest <= a.max_cells)
    cell_mm = a.block_mm / res
    print(f"Structure {structure.size[0]}x{structure.size[1]}x{structure.size[2]} blocks, "
          f"{len(structure.blocks)} non-air; {res} voxels per block ({cell_mm:.3f} mm each)")

    overrides = json.load(open(a.overrides)) if a.overrides else None
    with stage(0.0, 0.25, "Loading mods"):
        vox = Voxelizer(Assets(a.assets), res, overrides)
    from .cleanup import clean
    structure, notes = clean(structure, vox, not a.keep_backup_blocks, not a.keep_vs_blocks, a.missing,
                             a.hollow_interior, a.interior)
    for line in notes:
        print(line)
    if moving and a.contraptions != "none":
        from .contraptions import attach
        found, to_local = moving
        for line in attach(structure, found, to_local, a.contraptions, exact=a.output.lower().endswith(".glb")):
            print(line)
    if not structure.blocks:
        p.error("nothing left to export")
    filler = None
    if not a.output.lower().endswith(".glb"):
        structure, filler, cell_mm = print_prep(a, structure, vox, res, cell_mm)
        if not structure.blocks:
            p.error("nothing left to export")
    with stage(0.25, 0.45, "Reading block shapes"):
        grid = vox.voxelize(structure)

    if a.output.lower().endswith(".glb"):
        from .blender import TexturedBuilder, write_glb
        builder = TexturedBuilder(vox, structure)
        with stage(0.45, 0.85, "Building model"):
            faces = builder.build(merge=a.merge)
        if a.cull_interior:
            from .visibility import FILL_TEXTURE, Interior
            interior = Interior(vox, structure, max(1, a.solidify))
            n = faces.cull_invisible(interior.expo)
            fill = interior.fill_faces()
            faces.block = None
            for quad in fill:  # the fill closes sealed rooms; close them where they can be seen
                faces.add_quad(FILL_TEXTURE, None, quad, [(0, 0), (16, 0), (16, 16), (0, 16)], cullable=False)
            print(f"Removed {n} faces hidden inside (interior fill: "
                  f"{int(interior.filled_air.sum())} enclosed air blocks filled, {len(fill)} fill faces where they show)")
        tris, images = write_glb(a.output, faces, vox.assets, os.path.splitext(os.path.basename(a.output))[0],
                                 weld=a.merge, glass_single_sided=a.glass_backface_cull)
        extra = f", merged (culled {builder.culled} hidden faces, welded vertices)" if a.merge else ""
        print(f"Wrote {a.output} for Blender: {tris} triangles, {images} textures, "
              f"{len(faces.groups)} materials, 1 block = 1 m{extra}, in {time.time() - t0:.1f}s")
        report(vox)
        return 0
    if not a.voxel:
        export_exact(a, structure, vox, grid, res, cell_mm, t0, filler)
        report(vox)
        return 0

    grid = with_filler(grid, filler, vox, res, a.fill_bubbles)
    if a.min_thickness_mm > 0:
        grid = printprep.thicken(grid, math.ceil(a.min_thickness_mm / cell_mm))
    if a.hollow_mm > 0:
        grid = printprep.hollow(grid, max(1, math.ceil(a.hollow_mm / cell_mm)))
    grid = printprep.fix_nonmanifold(grid)

    verts, tris = mesh.surface(grid)
    verts, tris = mesh.add_meshes(verts, tris, vox.smooth_meshes)  # round parts such as tires
    verts = mesh.to_print_space(verts, cell_mm * UNITS[a.units], not a.y_up)
    write_print(a.output, verts, tris)
    size = verts.max(0) if len(verts) else np.zeros(3)
    print(f"Wrote {a.output}: {len(tris)} triangles, {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} {a.units}, "
          f"{grid.sum() * cell_mm ** 3 / 1000:.1f} cm3 in {time.time() - t0:.1f}s")
    report(vox)


UNITS = {"mm": 1.0, "cm": 0.1, "m": 0.001, "in": 1 / 25.4}


def write_print(path, verts, tris):
    if path.lower().endswith(".3mf"):
        mesh.write_3mf(path, verts, tris)
    else:
        mesh.write_stl(path, verts, tris)


def _region_contraptions(a):
    """Moving contraptions anchored inside a world region: (list, save -> grid mapping)."""
    from . import contraptions
    return None if a.contraptions == "none" else contraptions.for_region(a.world, a.corner1, a.corner2, a.dimension)


def _ship_contraptions(a, ship, structure):
    """Moving contraptions mounted on an Aeronautics ship: (list, save -> ship grid mapping)."""
    from . import contraptions
    return None if a.contraptions == "none" else contraptions.for_ship(a.world, ship, structure)


def _schematic_contraptions(a, structure):
    """Moving contraptions saved in a schematic (or on its sub-levels): (list, mapping)."""
    from . import contraptions
    return None if a.contraptions == "none" else contraptions.for_schematic(a.schematic, structure)


def print_prep(a, structure, vox, res, cell_mm):
    """Print steps on whole blocks: lesser blocks, model height, then fill / connect /
    floating / hollow / melt (modded2model.printsteps). Returns (structure of the blocks that stay,
    filler cells (padded block grid) or None, cell_mm)."""
    from . import printsteps
    if a.solidify > 0:  # older option: fill with sealed entrances
        a.fill_bubbles = a.seal_entrances = True
    if a.largest_only:
        a.delete_floating = max(a.delete_floating, 10 ** 9)
    blocks = structure.blocks
    if a.no_lesser:
        cache = {}
        kept = [b for b in blocks if not cache.setdefault(b[3], printsteps.is_lesser(vox, b[3]))]
        if len(kept) < len(blocks):
            print(f"Left out {len(blocks) - len(kept)} lesser block(s)")
        blocks = kept
    if a.height_cm > 0 and blocks:
        ys = [b[1] for b in blocks]
        a.block_mm = a.height_cm * 10 / (max(ys) - min(ys) + 1)
        cell_mm = a.block_mm / res
        print(f"Model {a.height_cm} cm tall: {a.block_mm:.2f} mm per block")
    structure = _subset(structure, blocks)
    opts = printsteps.Options(a.fill_bubbles, a.seal_entrances, a.seal_tunnels, a.connect_parts,
                            a.connect_corners and a.connect_parts, a.weld_all_edges, a.delete_floating,
                            printsteps.hollow_blocks_for(a.hollow_bottom_mm, a.block_mm), a.superhollow,
                            a.melt_snow)
    if not any((opts.fill_bubbles, opts.connect_parts, opts.delete_floating, opts.hollow_blocks, opts.melt_snow)):
        return structure, None, cell_mm
    with stage(0.25, 0.3, "Print steps"):
        cells, _ = printsteps.process(structure, vox, opts)
    keep = [b for b in structure.blocks if cells[b[0] + 1, b[1] + 1, b[2] + 1] == printsteps.BLOCK
            or not vox.block_shape(b[3]).any()]
    gone = len(structure.blocks) - len(keep)
    if gone:
        print(f"Print steps removed {gone} block(s)")
    filler = cells == printsteps.FILLER
    return _subset(structure, keep), (filler if filler.any() else None), cell_mm


def _subset(structure, blocks):
    from .schematic import Structure
    keep = {b[:3] for b in blocks}
    return Structure(structure.size, blocks, {k: v for k, v in structure.block_data.items() if k in keep},
                     structure.sub_levels, structure.merged, structure.skipped)


def with_filler(grid, filler, vox, res, pockets):
    """Add Filler blocks to the voxel grid as whole cells; with pockets (fill air
    bubbles) also fill air inside blocks that outside air can't reach."""
    out = grid.copy()
    off = np.asarray(vox.grid_offset)
    if filler is not None:
        for bx, by, bz in np.argwhere(filler[1:-1, 1:-1, 1:-1]):
            x0, y0, z0 = off + np.array([bx, by, bz]) * res
            out[x0:x0 + res, y0:y0 + res, z0:z0 + res] = True
    if pockets:
        g = np.pad(out, 1)
        out |= ~printprep._outside(~g)[1:-1, 1:-1, 1:-1]
    return out


def export_exact(a, structure, vox, grid, res, cell_mm, t0, filler=None):
    """Write each block's exact model geometry; Filler blocks become boxes."""
    joined = grid.copy()  # tires count as solid when deciding what's connected
    for tris in vox.smooth_meshes:
        rasterize_mesh(joined, tris)
    filled = with_filler(joined, filler, vox, res, a.fill_bubbles)
    if a.hollow_mm > 0:
        print("Note: --hollow-mm only applies with --voxel; use --hollow-bottom-mm  instead.")

    unit_mm = a.block_mm / 16
    # flat (zero-thickness) model elements always get 1 px so they print; more only on request
    exact = ExactModels(vox, max(1.0, a.min_thickness_mm / unit_mm))
    if a.fatten:  # "fatten lesser blocks": thin parts of small blocks at least 2 px
        from . import printsteps
        exact = _Fattened(exact, ExactModels(vox, max(2.0, a.min_thickness_mm / unit_mm)), vox, printsteps.is_lesser)
    expo = None
    if a.cull_interior:
        from .visibility import exposure
        expo = exposure(vox, structure, max(1, a.solidify))
    with stage(0.45, 0.95, "Building print model"):
        tris = assemble(structure, exact, joined, filled, None, 16 / res, expo)
    verts = mesh.to_print_space(tris.reshape(-1, 3), unit_mm * UNITS[a.units], not a.y_up)
    faces = np.arange(len(verts)).reshape(-1, 3)
    write_print(a.output, verts, faces)
    size = verts.max(0) if len(verts) else np.zeros(3)
    print(f"Wrote {a.output}: {len(faces)} triangles, {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} {a.units}, "
          f"about {filled.sum() * cell_mm ** 3 / 1000:.1f} cm3 in {time.time() - t0:.1f}s")


class _Fattened:
    """ExactModels that uses the fattened models for lesser blocks."""

    def __init__(self, normal, fat, vox, is_lesser):
        self._normal, self._fat, self._vox, self._is_lesser = normal, fat, vox, is_lesser
        self._lesser = {}

    def block_tris(self, state):
        if state not in self._lesser:
            self._lesser[state] = self._is_lesser(self._vox, state)
        return (self._fat if self._lesser[state] else self._normal).block_tris(state)

    def __getattr__(self, name):
        return getattr(self._normal, name)


def report(vox):
    if vox.unknown:
        print("No blockstate found (drawn as full cubes; add the mod jar or an override):", ", ".join(sorted(vox.unknown)))
    if vox.no_geometry:
        print("Models with no geometry, likely drawn by code (full cubes; consider an override):",
              ", ".join(sorted(vox.no_geometry)))


if __name__ == "__main__":
    sys.exit(main())
