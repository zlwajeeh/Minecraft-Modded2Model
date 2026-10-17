"""Resolve block states to model geometry using the game's and mods' own asset files.

Sources are searched in order (first match wins), like resource packs:
pass resource packs first, then mod jars, then the vanilla client jar.
"""
import glob
import hashlib
import io
import json
import os
import zipfile


def _split_id(ident, default_ns="minecraft"):
    if ":" in ident:
        ns, path = ident.split(":", 1)
    else:
        ns, path = default_ns, ident
    return ns, path


INDEX_DIR = os.path.join(os.path.expanduser("~"), ".modded2model", "assetindex")
BORROW_INDEX = os.path.join(os.path.expanduser("~"), ".modded2model", "borrowindex.json")


def _instances_root(paths):
    """The launcher's instances folder, found from an instance's mods folder (.../instances/X/
    [.]minecraft/mods), or None."""
    for p in paths:
        parts = os.path.normpath(os.path.abspath(p)).split(os.sep)
        low = [x.lower() for x in parts]
        if "instances" in low:
            i = len(low) - 1 - low[::-1].index("instances")
            return os.sep.join(parts[:i + 1])
    return None


def _jars_with_namespace(root, ns):
    """Mod jars in any instance under `root` with assets for namespace `ns`, newest first. Which
    namespaces each jar has is kept in an index (re-read only for jars that changed)."""
    try:
        with open(BORROW_INDEX, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        index = {}
    found, changed = [], False
    for jar in glob.glob(os.path.join(root, "*", "*minecraft", "mods", "*.jar")):
        try:
            st = os.stat(jar)
        except OSError:
            continue
        sig = f"{st.st_size}:{int(st.st_mtime)}"
        entry = index.get(jar)
        if not entry or entry.get("sig") != sig:
            try:
                with zipfile.ZipFile(jar) as z:
                    names = {n.split("/")[1] for n in z.namelist() if n.startswith("assets/") and n.count("/") >= 2}
            except (OSError, zipfile.BadZipFile):
                names = set()
            entry = index[jar] = {"sig": sig, "ns": sorted(names)}
            changed = True
        if ns in entry["ns"]:
            found.append((st.st_mtime, jar))
    if changed:
        try:
            os.makedirs(os.path.dirname(BORROW_INDEX), exist_ok=True)
            with open(BORROW_INDEX + ".tmp", "w", encoding="utf-8") as f:
                json.dump(index, f)
            os.replace(BORROW_INDEX + ".tmp", BORROW_INDEX)
        except OSError:
            pass
    return [jar for _, jar in sorted(found, reverse=True)]


class _Borrowed:
    """Stand-in source for Assets._json: reads `rel` from another instance's jar if it can."""
    is_dir, skip_models = False, False

    def __init__(self, assets):
        self.assets = assets

    def read(self, rel):
        return self.assets._borrow(rel)


class AssetSource:
    """One resource pack folder, or a jar/zip (possibly bundled inside another jar).

    Jars are opened only when a file is actually read from them; which asset files each one
    holds comes from the instance's index (see Assets), so most jars are never opened at all."""

    def __init__(self, path, names=None, parent=None, member=None):
        self.path = path
        self.is_dir = names is None and parent is None and os.path.isdir(path)
        self._zip = None
        self._parent, self._member = parent, member  # a jar bundled inside `parent`
        if names is None and not self.is_dir:
            names = [n for n in self._open().namelist() if n.startswith(("assets/", "META-INF/jarjar/"))]
        self.names = set(names or ())

    def _open(self):
        if self._zip is None:
            if self._parent is not None:  # read the bundled jar out of its parent jar (once)
                self._zip = zipfile.ZipFile(io.BytesIO(self._parent._open().read(self._member)))
            else:
                self._zip = zipfile.ZipFile(self.path)
        return self._zip

    def nested(self):
        """Jars bundled inside this jar (NeoForge jar-in-jar), e.g. Create Aeronautics' bundled mods."""
        if self.is_dir:
            return []
        return [AssetSource(f"{self.path}!{n}", names=(), parent=self, member=n)
                for n in sorted(self.names) if n.startswith("META-INF/jarjar/") and n.endswith(".jar")]

    def read(self, rel):
        if self.is_dir:
            full = os.path.join(self.path, rel)
            if os.path.isfile(full):
                with open(full, "rb") as f:
                    return f.read()
            return None
        if rel in self.names:
            return self._open().read(rel)
        return None


def _index_path(files):
    """The asset index for this exact set of jars (any jar added, removed or changed makes a new one)."""
    sig = []
    for f in files:
        try:
            st = os.stat(f)
            sig.append(f"{os.path.abspath(f)}|{st.st_size}|{int(st.st_mtime)}")
        except OSError:
            sig.append(f)
    return os.path.join(INDEX_DIR, hashlib.sha1(chr(10).join(sig).encode()).hexdigest()[:20] + ".json")


class Assets:
    def __init__(self, paths):
        """paths: resource pack folders/zips, jars, or folders of jars (e.g. a mods folder)."""
        from .progress import report
        self.sources = []
        files = []
        for p in paths:
            if os.path.isdir(p) and not os.path.isdir(os.path.join(p, "assets")):
                files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith((".jar", ".zip"))]
            else:
                files.append(p)
        index = _index_path(files)
        known = {}
        try:
            with open(index, encoding="utf-8") as f:
                known = json.load(f)
        except (OSError, ValueError):
            pass
        for i, f in enumerate(files):
            report(i / max(1, len(files)), f"Loading mods ({i + 1}/{len(files)})")
            self._add(AssetSource(f, known.get(f)) if f in known or os.path.isdir(f) else AssetSource(f), known)
        if not known:  # first time with this set of jars: remember which files each one holds
            try:
                os.makedirs(INDEX_DIR, exist_ok=True)
                tmp = index + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({s.path: sorted(s.names) for s in self.sources if not s.is_dir}, f)
                os.replace(tmp, index)
            except OSError:
                pass
        self._cache = {}
        self._namespaces = None  # mods (asset namespaces) this instance has
        self._borrow_root = _instances_root(paths)  # where the other instances are, if anywhere
        self._borrowed = {}  # namespace -> [AssetSource] taken from other instances
        self.borrowed_from = {}  # namespace -> jar path, for the export report

    # Mods that replace vanilla block models only as a rendering trick (Sodium builds stairs
    # from a full-size element with some faces left out). Their models look the same in game
    # but aren't the real shape, so they are skipped for model files.
    RENDER_ONLY_MODEL_MODS = ("sodium",)

    def _add(self, source, known=None):
        if any(m in os.path.basename(source.path.split("!")[-1]).lower() for m in self.RENDER_ONLY_MODEL_MODS):
            source.skip_models = True
        self.sources.append(source)
        for inner in source.nested():
            if known and inner.path in known:  # bundled jar: names from the index, opened only if read
                inner.names = set(known[inner.path])
            else:
                inner.names = {n for n in inner._open().namelist() if n.startswith(("assets/", "META-INF/jarjar/"))}
            self._add(inner, known)

    def _own_namespaces(self):
        if self._namespaces is None:
            ns = set()
            for s in self.sources:
                if s.is_dir:
                    d = os.path.join(s.path, "assets")
                    if os.path.isdir(d):
                        ns.update(os.listdir(d))
                else:
                    ns.update(n.split("/")[1] for n in s.names if n.startswith("assets/") and n.count("/") >= 2)
            self._namespaces = ns
        return self._namespaces

    def _borrow(self, rel):
        """For a mod this instance lacks: the jar from another Prism/MultiMC instance that holds `rel`
        (newest first), added as a last-resort source. Blocks from mods that aren't installed here
        (old schematics, other packs) then still get their real models."""
        parts = rel.split("/")
        if self._borrow_root is None or len(parts) < 3 or parts[0] != "assets":
            return None
        ns = parts[1]
        if ns in self._own_namespaces():
            return None
        if ns not in self._borrowed:
            self._borrowed[ns] = [AssetSource(path) for path in _jars_with_namespace(self._borrow_root, ns)]
        for s in self._borrowed[ns]:
            if rel in s.names:
                if s not in self.sources:
                    self.sources.append(s)
                    self.borrowed_from[ns] = s.path
                return s.read(rel)
        return None

    def _json(self, rel):
        if rel in self._cache:
            return self._cache[rel]
        data = None
        for s in self.sources + [_Borrowed(self)]:
            if "/models/" in rel and getattr(s, "skip_models", False):
                continue
            raw = s.read(rel)
            if raw is not None:
                try:
                    data = json.loads(raw.decode("utf-8-sig"))
                except ValueError:  # a mod shipped a broken or empty JSON file: treat as missing
                    data = None
                break
        self._cache[rel] = data
        return data

    def blockstate(self, block_name):
        ns, path = _split_id(block_name)
        return self._json(f"assets/{ns}/blockstates/{path}.json")

    def model(self, model_id):
        """Return the fully inherited model: {'elements': [...], 'textures': {...}} or None."""
        key = ("model", model_id)
        if key in self._cache:
            return self._cache[key]
        ns, path = _split_id(model_id)
        data = self._json(f"assets/{ns}/models/{path}.json")
        result = None
        if data is not None:
            parent = self.model(data["parent"]) if "parent" in data else None
            elements = data.get("elements")
            if elements is None and parent:
                elements = parent["raw_elements"]
            textures = dict(parent["textures"]) if parent else {}
            textures.update(data.get("textures", {}))
            mesh = parent.get("mesh") if parent else None
            obj = parent.get("obj") if parent else None
            if str(data.get("loader", "")).endswith(":obj") and "model" in data:
                mesh = self.obj_mesh(data["model"])
                obj = data["model"]
            from .billboard import solidify_elements
            raw = elements or []
            result = {"elements": solidify_elements(raw, textures, self, self._cache.setdefault("alpha", {})),
                      "raw_elements": raw, "textures": textures, "mesh": mesh, "obj": obj, "id": model_id}
        self._cache[key] = result
        return result

    def raw(self, rel):
        for s in self.sources:
            data = s.read(rel)
            if data is not None:
                return data
        return self._borrow(rel)

    def obj_mesh(self, ident):
        """Triangles (T, 3, 3) in 0..16 block units from a Wavefront .obj (NeoForge/Forge obj loader)."""
        key = ("obj", ident)
        if key in self._cache:
            return self._cache[key]
        ns, path = _split_id(ident)
        raw = self.raw(f"assets/{ns}/{path}")
        tris = None
        if raw is not None:
            verts, faces = [], []
            for line in raw.decode("utf-8", "replace").splitlines():
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "v":
                    verts.append([float(v) for v in parts[1:4]])
                elif parts[0] == "f":
                    idx = [int(p.split("/")[0]) for p in parts[1:]]
                    idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                    for k in range(1, len(idx) - 1):  # fan-triangulate polygons
                        faces.append((idx[0], idx[k], idx[k + 1]))
            if faces:
                import numpy as np
                # The loaders treat OBJ units as blocks, with the origin at the block's corner.
                tris = np.array(verts)[np.array(faces)] * 16.0
        self._cache[key] = tris
        return tris

    def item_extra_elements(self, block_name, force=False):
        """Elements a block's item model has that its placed model doesn't.

        Many mods draw moving parts (propeller blades, shafts, handles, dials) with a block
        entity, so the placed model is only the casing; the item model shows the assembled
        block. The extra elements are those parts at rest, authored in the same frame.
        Skipped for multipart blockstates (fences, pipes: the item shows a sample shape)
        and vanilla blocks.
        """
        key = ("item_extra", block_name, force)
        if key in self._cache:
            return self._cache[key]
        ns, path = _split_id(block_name)
        extras = []
        bs = self.blockstate(block_name)
        if ns != "minecraft" and bs and "variants" in bs and "multipart" not in bs:
            item = self.model(f"{ns}:item/{path}")
            if item and item["elements"]:
                def k(e):
                    return (tuple(round(v, 3) for v in e["from"]), tuple(round(v, 3) for v in e["to"]),
                            json.dumps(e.get("rotation"), sort_keys=True))
                placed, base = set(), None
                for v in bs["variants"].values():
                    for e in (v if isinstance(v, list) else [v]):
                        m = self.model(e.get("model", ""))
                        if m:
                            keys = {k(el) for el in m["elements"]}
                            placed |= keys
                            base = keys if base is None else base
                item_keys = {k(el) for el in item["elements"]}
                # Only trust the item when it is the placed model plus parts, in the same frame.
                # Items that redraw the whole block turned another way (iron handles, winches,
                # cranks) would otherwise add a second, rotated copy.
                if base and (force or base <= item_keys):
                    # Carry the item's textures along so a textured export can resolve "#refs".
                    extras = [dict(el, __textures=item["textures"]) for el in item["elements"]
                              if k(el) not in placed]
        self._cache[key] = extras
        return extras

    def obj_textured(self, ident):
        """Like obj_mesh, plus per-corner UVs (T, 3, 2) and each triangle's material name.

        Materials come from the .mtl file's map_Kd, which in mod OBJs is usually a model
        texture reference such as "#tire_0". Returns (tris, uvs, materials) or None.
        """
        key = ("objtex", ident)
        if key in self._cache:
            return self._cache[key]
        ns, path = _split_id(ident)
        raw = self.raw(f"assets/{ns}/{path}")
        result = None
        if raw is not None:
            import numpy as np
            verts, uvs, tris, tuv, mats = [], [], [], [], []
            mtl_maps, current = {}, None
            folder = path.rsplit("/", 1)[0]
            for line in raw.decode("utf-8", "replace").splitlines():
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "mtllib":
                    mtl = self.raw(f"assets/{ns}/{folder}/{parts[1]}")
                    name = None
                    for ml in (mtl or b"").decode("utf-8", "replace").splitlines():
                        mp = ml.split()
                        if mp and mp[0] == "newmtl":
                            name = mp[1]
                        elif mp and mp[0] == "map_Kd" and name:
                            mtl_maps[name] = mp[1]
                elif parts[0] == "usemtl":
                    current = parts[1]
                elif parts[0] == "v":
                    verts.append([float(v) for v in parts[1:4]])
                elif parts[0] == "vt":
                    uvs.append([float(v) for v in parts[1:3]])
                elif parts[0] == "f":
                    corners = []
                    for p in parts[1:]:
                        bits = p.split("/")
                        vi = int(bits[0])
                        ti = int(bits[1]) if len(bits) > 1 and bits[1] else 0
                        corners.append((vi - 1 if vi > 0 else len(verts) + vi,
                                        ti - 1 if ti > 0 else (len(uvs) + ti if ti else -1)))
                    for k in range(1, len(corners) - 1):
                        tri = (corners[0], corners[k], corners[k + 1])
                        tris.append([verts[c[0]] for c in tri])
                        tuv.append([uvs[c[1]] if c[1] >= 0 else [0, 0] for c in tri])
                        mats.append(mtl_maps.get(current, current))
            if tris:
                result = (np.array(tris) * 16.0, np.array(tuv), mats)
        self._cache[key] = result
        return result

    def raw_texture(self, texture_id):
        ns, path = _split_id(texture_id)
        return self.raw(f"assets/{ns}/textures/{path}.png")

    def texture_png(self, texture_id):
        """PNG bytes of a texture like 'create:block/axis' (first frame if animated), or None."""
        key = ("png", texture_id)
        if key in self._cache:
            return self._cache[key]
        ns, path = _split_id(texture_id)
        raw = self.raw(f"assets/{ns}/textures/{path}.png")
        result = None
        if raw is not None:
            import io as _io
            from PIL import Image
            img = Image.open(_io.BytesIO(raw)).convert("RGBA")
            w, h = img.size
            if h > w and h % w == 0:  # animated strip: keep the first frame
                img = img.crop((0, 0, w, w))
            alpha = img.getchannel("A").getextrema()[0] < 255
            out = _io.BytesIO()
            img.save(out, "PNG")
            result = (out.getvalue(), alpha)
        self._cache[key] = result
        return result

    def resolve(self, state):
        """Return a list of (model_dict, x_rot, y_rot) for a BlockState, or None if unknown."""
        bs = self.blockstate(state.name)
        if bs is None:
            return None
        props = state.properties
        picks = []
        if "variants" in bs:
            for key, value in bs["variants"].items():
                if _variant_matches(key, props):
                    picks.append(value)
                    break
        for part in bs.get("multipart", []):
            if "when" not in part or _when_matches(part["when"], props):
                picks.append(part["apply"])
        out = []
        for p in picks:
            if isinstance(p, list):  # weighted random choice: take the first
                p = p[0]
            m = self.model(p["model"])
            if m is not None:
                out.append((m, p.get("x", 0), p.get("y", 0)))
        return out


def _variant_matches(key, props):
    if key in ("", "normal"):
        return True
    for cond in key.split(","):
        k, _, v = cond.partition("=")
        if props.get(k) != v:
            return False
    return True


def _when_matches(when, props):
    if "OR" in when:
        return any(_when_matches(w, props) for w in when["OR"])
    if "AND" in when:
        return all(_when_matches(w, props) for w in when["AND"])
    for k, v in when.items():
        if str(props.get(k)) not in str(v).split("|"):
            return False
    return True

