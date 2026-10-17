"""Audit every block model in an instance for geometry Minecraft-Modded2Model may be missing.

python tools_audit.py <instance> [-o report.csv]

For every blockstate file in every jar it checks:
  missing_parts  the item model has more elements than the placed block model (the block
                 entity probably draws the rest, like the portable engine's exhaust and hatch)
  unused_parts   model files in the block's own folder that the blockstate never uses
  no_shape       blockstate resolves to nothing / only empty models (exported as a cube)
Blocks that Minecraft-Modded2Model already handles with a rule, alias or extra-parts entry are marked.
Blocks used in the instance's schematics or Sable ships are listed first.
"""
import argparse
import csv
import re
from collections import Counter

from modded2model import codeshapes, instance, sable
from modded2model.assets import Assets, _split_id
from modded2model.schematic import load_structure


def model_refs(bs):
    out = []
    for v in (bs or {}).get("variants", {}).values():
        for e in (v if isinstance(v, list) else [v]):
            out.append(e.get("model"))
    for part in (bs or {}).get("multipart", []):
        a = part.get("apply")
        for e in (a if isinstance(a, list) else [a]):
            out.append(e.get("model"))
    return [m for m in out if m]


def count_elements(assets, model_id, seen=None):
    m = assets.model(model_id)
    if m is None:
        return 0
    return len(m["elements"]) + (1 if m.get("mesh") is not None else 0)


def used_blocks(inst):
    used = Counter()
    for name in instance.list_schematics(inst):
        try:
            for *_, st in load_structure(instance.find_schematic(inst, name)).blocks:
                used[st.name] += 1
        except Exception:
            pass
    for world in instance.list_worlds(inst):
        try:
            for ship in sable.ships(instance.find_world(inst, world)):
                for *_, st in ship.structure().blocks:
                    used[st.name] += 1
        except Exception:
            pass
    return used


def main():
    p = argparse.ArgumentParser()
    p.add_argument("instance")
    p.add_argument("-o", "--output", default="model_audit.csv")
    a = p.parse_args()
    assets = Assets(instance.asset_paths(a.instance))
    used = used_blocks(a.instance)

    # Every blockstate file, first source wins (same priority as export).
    blockstates, all_models = {}, set()
    for src in assets.sources:
        names = src.names if src.zip is not None else []
        for n in names:
            m = re.match(r"assets/([^/]+)/blockstates/(.+)\.json$", n)
            if m:
                blockstates.setdefault(f"{m.group(1)}:{m.group(2)}", None)
            m = re.match(r"assets/([^/]+)/models/(block/.+)\.json$", n)
            if m:
                all_models.add(f"{m.group(1)}:{m.group(2)}")

    rows = []
    for name in sorted(blockstates):
        handled = ("rule" if name in codeshapes.RULES else "alias" if name in codeshapes.ALIASES else
                   "jar_models" if name in codeshapes.JAR_MODELS else
                   "extra_parts" if codeshapes.extra_models(name) else "")
        try:
            bs = assets.blockstate(name)
        except ValueError:
            rows.append((name, used.get(name, 0), "bad_json", "", handled))
            continue
        refs = model_refs(bs)
        block_els = max((count_elements(assets, r) for r in refs), default=0)
        issues = []
        if not refs or block_els == 0:
            issues.append("no_shape")
        ns, path = _split_id(name)
        item_id = f"{ns}:item/{path}"
        item = assets.model(item_id)
        item_els = len(item["elements"]) + (1 if item.get("mesh") is not None else 0) if item else 0
        if item_els > block_els and block_els > 0:
            issues.append(f"missing_parts(item {item_els} > block {block_els})")
        # Part models living in a folder named after the block (or its base model's folder).
        folders = {f"{ns}:block/{path}/"}
        for r in refs:
            rns, rpath = _split_id(r)
            folders.add(f"{rns}:{rpath.rsplit('/', 1)[0]}/")
            data = assets._json(f"assets/{rns}/models/{rpath}.json") or {}
            if "parent" in data:
                pns, ppath = _split_id(data["parent"])
                folders.add(f"{pns}:{ppath.rsplit('/', 1)[0]}/")
        folders = {f for f in folders if f.count("/") >= 2}  # only real sub-folders, not block/
        referenced = set(refs) | set(codeshapes.extra_models(name))
        unused = sorted(m for m in all_models
                        if any(m.startswith(f) for f in folders) and m not in referenced
                        and not m.endswith(("/item", "/block")) and count_elements(assets, m) > 0)
        if unused:
            issues.append(f"unused_parts({len(unused)}: {', '.join(u.split('/')[-1] for u in unused[:6])})")
        if issues:
            rows.append((name, used.get(name, 0), "; ".join(issues), item_els, handled))

    rows.sort(key=lambda r: (-r[1], r[0]))
    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block", "uses in your builds", "issues", "item elements", "handled by"])
        w.writerows(rows)
    in_builds = [r for r in rows if r[1] > 0]
    print(f"{len(blockstates)} blockstates checked, {len(rows)} with possible missing geometry, "
          f"{len(in_builds)} of them used in your builds. Full list: {a.output}")
    for r in in_builds:
        print(f"  {r[0]:50} x{r[1]:<5} {r[2]}  {('[handled: ' + r[4] + ']') if r[4] else ''}")


if __name__ == "__main__":
    main()
