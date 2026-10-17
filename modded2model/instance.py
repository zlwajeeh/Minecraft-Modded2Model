"""Find assets and schematics in a Prism Launcher / MultiMC instance folder."""
import json
import os


def _game_dir(instance):
    for name in ("minecraft", ".minecraft"):
        d = os.path.join(instance, name)
        if os.path.isdir(d):
            return d
    return instance


def minecraft_version(instance):
    with open(os.path.join(instance, "mmc-pack.json")) as f:
        for c in json.load(f)["components"]:
            if c["uid"] == "net.minecraft":
                return c["version"]
    raise ValueError("no Minecraft version in mmc-pack.json")


def vanilla_jar(instance, version):
    """Prism keeps client jars in <launcher>/libraries/com/mojang/minecraft/<v>/."""
    launcher = os.path.dirname(os.path.dirname(os.path.abspath(instance)))
    candidates = [
        os.path.join(launcher, "libraries", "com", "mojang", "minecraft", version, f"minecraft-{version}-client.jar"),
        os.path.join(os.environ.get("APPDATA", ""), ".minecraft", "versions", version, f"{version}.jar"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Minecraft {version} client jar not found; launch the instance once, or pass it with -a")


def resource_packs(game_dir):
    """Enabled resource packs from options.txt, highest priority first."""
    opts = os.path.join(game_dir, "options.txt")
    if not os.path.isfile(opts):
        return []
    with open(opts, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("resourcePacks:"):
                packs = json.loads(line.split(":", 1)[1])
                break
        else:
            return []
    out = []
    for p in reversed(packs):  # options.txt lists lowest priority first
        if p.startswith("file/"):
            path = os.path.join(game_dir, "resourcepacks", p[5:])
            if os.path.exists(path):
                out.append(path)
    return out


def asset_paths(instance):
    game = _game_dir(instance)
    paths = resource_packs(game)
    paths.append(os.path.join(game, "mods"))
    paths.append(vanilla_jar(instance, minecraft_version(instance)))
    return paths


def find_schematic(instance, name):
    if os.path.isfile(name):
        return name
    d = os.path.join(_game_dir(instance), "schematics")
    for candidate in (name, name + ".nbt"):
        p = os.path.join(d, candidate)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"no schematic '{name}' in {d}")


def list_schematics(instance):
    d = os.path.join(_game_dir(instance), "schematics")
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".nbt"))


def find_world(instance, name):
    if os.path.isfile(os.path.join(name, "level.dat")):
        return name
    p = os.path.join(_game_dir(instance), "saves", name)
    if os.path.isfile(os.path.join(p, "level.dat")):
        return p
    raise FileNotFoundError(f"no world '{name}' in {os.path.dirname(p)}")


def list_worlds(instance):
    d = os.path.join(_game_dir(instance), "saves")
    return sorted(w for w in os.listdir(d) if os.path.isfile(os.path.join(d, w, "level.dat")))
