# Minecraft-Modded2Model v1.0.2

Export Minecraft builds, modded blocks included, as watertight 3D-printable models.
Block shapes come from the game's and mods' own model files,
so Create, Create Aeronautics and other mods work without per-block code.

## Setup

```
python -m pip install --user numpy scipy
```

## Window

Double-click `Minecraft-Modded2Model.bat` (or run `python -m modded2model.gui_classic`). For the
dark-themed window with the same options, use `Minecraft-Modded2Model (dark).bat` (`python -m modded2model.gui`). Pick the instance folder, then on the
left choose what to export: a **Schematic** (search box, plus "Part to export" for Aeronautics
sub-levels), a **World region** (type two corners, or use the area around your player), or an
**Aeronautics ship** (Find ships, then click one). The **Preview** on the right renders it in the
background; schematics with sub-levels show each one numbered and coloured where it was saved.
The preview is square; click it to open a large, high-resolution view. Under the list, pick the
format (STL, 3MF or GLB); the option tabs follow it: **3D printing** for STL and
3MF, **Blender** for GLB, and **Clean-up** for both. Press **Export**. The window remembers your
instance and options in `~/.modded2model.json`.

## Blender export

Give the output a `.glb` name (or pick "GLB · Blender, textured" in the window):

```
python -m modded2model mm_pickup_truck_1 --instance <instance> -o truck.glb
```

Add `--merge` (window: "Blender: connect meshes and cull hidden faces") to drop faces nobody can
see (faces pressed against a solid neighbouring block, and back-to-back faces where two blocks or
parts touch; glass, grates and doors don't hide anything) and weld shared vertices so surfaces
come in as connected meshes.

In Blender: File > Import > glTF 2.0. Every face keeps its texture and UVs from the mod jars,
OBJ models (tires, valve handles) keep theirs, copycats show the block they copy, and
cut-out textures (glass, grates) are alpha-masked. 1 block = 1 metre, centred and standing on
the floor. Textures use nearest filtering so the pixel art stays sharp.

## Checking models

- `python tools_audit.py <instance>` checks every blockstate in the modpack for geometry the
  export might miss (parts drawn by block entities, empty models) and lists the ones in your builds.
- `python tools_render.py <instance> <block id> key=value ...` renders one block's exported shape.

## Usage

Save a build with Create's Schematic and Quill, then point Minecraft-Modded2Model at your Prism instance. It finds the
mods, enabled resource packs, the matching vanilla jar and the schematics folder by itself:

```
python -m modded2model --instance %APPDATA%/PrismLauncher/instances/AeronauticsModpack --list
python -m modded2model truck1 --instance %APPDATA%/PrismLauncher/instances/AeronauticsModpack
```

### From a world save

Export a box of blocks (corners in world coordinates, as shown by F3):

```
python -m modded2model --instance <instance> -w "Create Testing" --from 905 70 -1185 --to 935 90 -1160
```

Add `--dimension nether`, `end` or a modded id such as `mod:sky` for other dimensions.

Create Aeronautics ships are Sable "sub-levels" stored in `<world>/sublevels/`, not in the normal world.
Export one in its own unrotated layout, even if it was mid-flight when you saved:

```
python -m modded2model --instance <instance> -w "Create Testing" --list-ships
python -m modded2model --instance <instance> -w "Create Testing" --ship r.1.-3.0:12
```

The Sable format was worked out from your saves, not from documentation, so a Sable update may change it.

Or give every source by hand:

```
python -m modded2model my_ship.nbt -a mods/create-aeronautics.jar -a mods/create.jar -a versions/1.21.1/1.21.1.jar -o my_ship.stl --block-mm 5
```

List `-a` sources highest priority first: resource packs, then mod jars, then the vanilla client jar.
An `-a` can be a whole mods folder; jars bundled inside other jars (e.g. Create Aeronautics) are read too.
For the Prism instance:

```
python -m modded2model %APPDATA%/PrismLauncher/instances/AeronauticsModpack/minecraft/schematics/truck1.nbt -a %APPDATA%/PrismLauncher/instances/AeronauticsModpack/minecraft/mods -a %APPDATA%/PrismLauncher/libraries/com/mojang/minecraft/1.21.1/minecraft-1.21.1-client.jar -o truck1.stl
```

Copycats+ and Create copycat blocks, and Create fluid pipes, get built-in shape rules
(`modded2model/codeshapes.py`). Some are best guesses and marked "approx" there.

By default every block is written with its exact model geometry: each Blockbench element is its own
closed box at its true size and rotation, OBJ models (tires, valve handles) are kept as they are, and
nothing is rounded to a grid. Slicers merge the overlapping shells. `--voxel` gives the older blocky
export, which is the only mode that supports `--hollow-mm`.

### 3D printing options

The window's "3D printing" tab follows the classic 3D print dialog (without
material, cost and size-for-material settings). The block operations (`modded2model/printsteps.py`) run
on whole blocks, in this order:

| Window | Command line | What it does |
|---|---|---|
| Make Z the up direction | (default; `--y-up` to turn off) | printers expect Z up |
| Make the model N cm in height / Make each block N mm | `--height-cm N` / `--block-mm N` | model size |
| Model's units | `--units mm\|cm\|m\|in` | units written in the file |
| Fill air bubbles | `--fill-bubbles` | fills air that outside air can't reach |
| Seal off entrances | `--seal-entrances` | one-block doorways/windows count as closed while filling |
| Fill in isolated tunnels in model's base | `--seal-tunnels` | air only enters from the top of the box |
| Connect parts sharing an edge | `--connect-parts` | adds a block where parts touch only along an edge |
| Connect corner tips | `--connect-corners` | adds a block where parts touch only at a corner |
| Weld all shared edges | `--weld-all-edges` | also welds edges within one part |
| Delete floating objects smaller than N | `--delete-floating N` | removes small loose parts and trees |
| Hollow out bottom, wall N mm, Superhollow | `--hollow-bottom-mm N`, `--superhollow` | hollows from the bottom up |
| Melt snow blocks | `--melt-snow` | removes snow blocks after the steps above |
| Export lesser, detailed blocks / Fatten | (default; `--no-lesser`), `--fatten` | keeps small detail blocks (handles, levers, shafts...); fatten makes their thin parts at least 2 px |

Extras: `--min-thickness-mm` (thicken thin parts everywhere), `--voxel` (blocky export).
`--solidify 1` still works and means fill air bubbles + seal off entrances.

`--interior keep|hollow|solid` (window: 3D printing tab, "Non-solid blocks inside") handles every
block inside the model that isn't a plain solid block (cogwheels, gearboxes, shafts, engines, tanks,
furniture): keep it, remove it (leaving the inside hollow), or replace it with a solid block.
`--hollow-interior` is the same as `--interior hollow`. Inside means not touching outside air once
every block is a wall and doorways are sealed.

Before exporting, Drive-by-Wire backup blocks are removed (`--keep-backup-blocks` to keep them) and
Valkyrien Skies blocks become look-alikes: helms become stairs, balloons Aeronautics envelopes and
floaters brown wool (`--keep-vs-blocks`). `--missing skip` leaves out blocks with no model instead of
drawing full cubes. `--glass-backface-cull` makes glass single-sided in Blender.

Torches, ladders, bars, redstone link antennas and other cut-out "card" parts are rebuilt from their
texture's opaque pixels as solid pixel boxes, so they print and render as solid objects.

Offroad wheel mounts get their tire (read from the block's stored data and the tire's OBJ model),
axle and hub, so wheels print attached to the vehicle.

Other options: `--res` (voxels per block, 16 = full detail), `--max-cells`. Output `.stl` or `.3mf`.

Blocks drawn by code instead of model files are reported at the end and drawn as full cubes.
Give them shapes with `--overrides overrides.json`:

```json
{"create:belt": {"boxes": [[0, 0, 0, 16, 4, 16]]}, "create:fluid_tank": "full", "minecraft:light": "skip"}
```

## Layout

- `modded2model/nbt.py`: NBT reader/writer
- `modded2model/schematic.py`: structure (.nbt) loader
- `modded2model/assets.py`: blockstate and model lookup across jars and packs
- `modded2model/voxelize.py`: models to a voxel grid (handles blockstate and angled element rotations)
- `modded2model/printprep.py`: fill cavities, thicken, hollow, remove debris, fix non-manifold contacts
- `modded2model/mesh.py`: watertight surface extraction, STL and 3MF writers
- `tests/test_pipeline.py`: end-to-end test (pass a vanilla client jar path to include vanilla blocks)
- `tests/test_pickup_truck.py`: reference model test on `mm_pickup_truck_1` from the Prism instance
- `modded2model/attachments.py`: parts drawn from block data outside the block (Offroad wheels)

## Aeronautics sub-levels in schematics

Schematics can hold Aeronautics ships ("sub-levels") next to the normal blocks. To see which is
which, use the window's "Preview sub-levels" button on the Schematic tab, or:

```
python -m modded2model oct_submarine --instance <instance> --preview-sub-levels subs.png
```

The picture shows the build with every sub-level placed where it was saved, each in its own colour,
and a numbered thumbnail per sub-level. Pick one in the window's "Export" box, or pass
`--sub-level N`. Unrotated sub-levels (things held by a sticker or piston) are merged into the main
export automatically.

## Moving contraptions (bearings, propellers, windmills)

While a bearing is turning, the blocks it carries are saved inside a Create contraption entity, not
as blocks. They are read back from schematics, world regions and Aeronautics ships, and placed on
their bearing (`modded2model/contraptions.py`). `--contraptions` (window: Clean-up tab) chooses how:

- `world` (default): at the angle they were saved at. GLB exports turn the meshes to that exact
  angle; prints snap to the nearest quarter turn, since a print is built on the block grid.
- `snap`: the nearest quarter turn, so every block stays on the grid.
- `assembled`: as built (angle 0).
- `none`: leave them out.

## Known issues (not fixed yet)

- Rotated Aeronautics sub-levels in schematics (e.g. oct_submarine sub-level 1) aren't merged
  with the main blocks; export them with --sub-level N.

## Credits

The 3D print block operations (`modded2model/printsteps.py`) are ported from ObjFileManip.cpp of Eric
Haines' open-source Mineways.
