# Changelog

## v1.0.7
Printed cars can roll, schematics can export every sub-level at once, and a full check of every export mode.

- Snap-on rolling wheels for 3D printing (`--snap-wheels`):
  - Wheels print separately as smooth round wheels with a bore, and snap onto split-head peg axles on the car.
  - Where the car is in a wheel's way, the wheel is trimmed and made smaller, or kept full size and moved out (`--snap-full-size`).
- "Everything" part (`--sub-level all`): the main build and every Aeronautics sub-level placed where it was saved, attached or not. GLB keeps exact angles; prints turn tilted ships to the nearest quarter turn.
- 3MF files record the unit chosen with `--units` (they always said millimetres).
- Water, lava and bubble columns are left out of exports. A schematic of only ships and water exports its ship.
- Beds, chests, signs and banners get stand-in shapes instead of full cubes.
- Create fluid pipe junctions with five connections draw properly; open sliding doors are empty.
- Contraption blocks turned onto their side (stairs) keep their shape instead of becoming a cube.
- Create Big Cannons: connected cannon textures and sliding breechblocks; old mount states resolve.
- Smaller fixes, and more tests.

## v1.0.6
- Create Deep Seas multi-block propellers and Create Propulsion multi-block thrusters, in every export mode including blocky print mode.
- Create: High Seas rudder and Create Propulsion copycat wings copy their block.
- Aeroworks control desks and stands: modules and copycat skins.

## v1.0.5
- Single-file Windows exe with its own icon; given arguments it runs the command line.
- Python no longer stays running after the window closes.

## v1.0.4
Blender rigging:
- Moving contraptions (bearings, propellers, windmills) as separate objects on their pivots.
- Create kinetic parts (shafts, cogs, gearboxes, chain drives...) turning at Create's speed ratios, including Create Connected parallel and six-way gearboxes.
- Separate toggles for contraptions and kinetic parts; Kinetics and Spin animations.
- Auto-rig option: one skinned mesh with an armature, a bone per moving part.

## v1.0.3
- Fluid pipes follow Create 6: junction casings, rims, and straight-pipe rules for glass pipes, smart pipes and valves.
- Ship names from Sable's display name or the ship's nameplates, shown in lists and used as the default file name.
- Nameplate text drawn in Minecraft's own font (GLB and previews).
- The World region tab renders its preview only when Preview is pressed.

## v1.0.2
- Valkyrien Skies Clockwork propeller blades drawn the way Clockwork does.
- Blocks from mods not in the chosen instance borrow their models from your other Prism/MultiMC instances.

## v1.0.1
- Moving contraptions on bearings, propeller bearings and windmills are exported on their bearing.
- World map picker, with a per-instance block table and cached map tiles.
- Faster exports.
