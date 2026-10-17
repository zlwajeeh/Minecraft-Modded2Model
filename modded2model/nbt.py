"""Minimal reader/writer for Minecraft's NBT format (big-endian, Java Edition)."""
import gzip
import struct

END, BYTE, SHORT, INT, LONG, FLOAT, DOUBLE, BYTE_ARRAY, STRING, LIST, COMPOUND, INT_ARRAY, LONG_ARRAY = range(13)


class _Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, fmt):
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return vals[0] if len(vals) == 1 else vals

    def string(self):
        n = self.take(">H")
        s = self.data[self.pos:self.pos + n].decode("utf-8", "replace")
        self.pos += n
        return s

    def payload(self, tag):
        if tag == BYTE: return self.take(">b")
        if tag == SHORT: return self.take(">h")
        if tag == INT: return self.take(">i")
        if tag == LONG: return self.take(">q")
        if tag == FLOAT: return self.take(">f")
        if tag == DOUBLE: return self.take(">d")
        if tag == STRING: return self.string()
        if tag in (BYTE_ARRAY, INT_ARRAY, LONG_ARRAY):
            n = self.take(">i")
            fmt = {BYTE_ARRAY: "b", INT_ARRAY: "i", LONG_ARRAY: "q"}[tag]
            vals = struct.unpack_from(f">{n}{fmt}", self.data, self.pos)
            self.pos += n * struct.calcsize(fmt)
            return list(vals)
        if tag == LIST:
            sub = self.take(">b")
            n = self.take(">i")
            return [self.payload(sub) for _ in range(n)]
        if tag == COMPOUND:
            out = {}
            while True:
                t = self.take(">b")
                if t == END:
                    return out
                name = self.string()  # must be read before the payload
                out[name] = self.payload(t)
        raise ValueError(f"unknown NBT tag {tag}")


def loads(data):
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    r = _Reader(data)
    tag = r.take(">b")
    r.string()  # root name
    return r.payload(tag)


def load(path):
    with open(path, "rb") as f:
        return loads(f.read())


# --- fast chunk reader: only the block palettes, everything else skipped ---

_FIXED = {BYTE: 1, SHORT: 2, INT: 4, LONG: 8, FLOAT: 4, DOUBLE: 8}
_ARRAY = {BYTE_ARRAY: 1, INT_ARRAY: 4, LONG_ARRAY: 8}
_U16, _I32 = struct.Struct(">H"), struct.Struct(">i")
_PALETTES = {}  # raw palette bytes -> parsed palette (shared, read-only)


def _skip(d, pos, tag):
    """Position just past a payload, without building it."""
    if tag in _FIXED:
        return pos + _FIXED[tag]
    if tag == STRING:
        return pos + 2 + _U16.unpack_from(d, pos)[0]
    if tag in _ARRAY:
        return pos + 4 + _I32.unpack_from(d, pos)[0] * _ARRAY[tag]
    if tag == LIST:
        sub, n = d[pos], _I32.unpack_from(d, pos + 1)[0]
        pos += 5
        if sub in _FIXED:
            return pos + n * _FIXED[sub]
        for _ in range(n):
            pos = _skip(d, pos, sub)
        return pos
    if tag == COMPOUND:
        while True:
            t = d[pos]
            pos += 1
            if t == END:
                return pos
            pos = _skip(d, pos + 2 + _U16.unpack_from(d, pos)[0], t)
    raise ValueError(f"unknown NBT tag {tag}")


def _name(d, pos):
    n = _U16.unpack_from(d, pos)[0]
    return d[pos + 2:pos + 2 + n].decode("utf-8", "replace"), pos + 2 + n


def _palette(d, pos):
    """A block palette (list of {Name, Properties{str: str}}) read directly; None if it holds
    anything other than those string tags."""
    if d[pos] != COMPOUND:
        return None
    n = _I32.unpack_from(d, pos + 1)[0]
    pos += 5
    out = []
    for _ in range(n):
        e = {}
        while True:
            t = d[pos]
            pos += 1
            if t == END:
                break
            k, pos = _name(d, pos)
            if t == STRING:
                e[k], pos = _name(d, pos)
            elif t == COMPOUND:
                props = {}
                while True:
                    pt = d[pos]
                    pos += 1
                    if pt == END:
                        break
                    if pt != STRING:
                        return None
                    pk, pos = _name(d, pos)
                    props[pk], pos = _name(d, pos)
                e[k] = props
            else:
                return None
        out.append(e)
    return out


def palette(key):
    """Parse a palette returned by chunk_sections (cached: neighbouring chunks share palettes)."""
    pal = _PALETTES.get(key)
    if pal is None:
        tag, body = key[0], key[1:]
        pal = _palette(body, 0) if tag == LIST else None
        if pal is None:  # unusual layout: use the general reader
            pal = _Reader(body).payload(tag)
        if len(_PALETTES) > 20000:
            _PALETTES.clear()
        _PALETTES[key] = pal
    return pal


def chunk_sections(data):
    """From raw (decompressed) chunk NBT, read only each section's Y and block_states, without
    parsing them: [(y, palette key, packed data as int64 array or None)]. The palette key is the
    palette's raw bytes, for palette(); entities, lighting, biomes and the rest are skipped."""
    import numpy as np
    d = data
    if d[0] != COMPOUND:
        raise ValueError("chunk NBT must start with a compound")
    pos = 3 + _U16.unpack_from(d, 1)[0]  # root tag + name
    out = []
    while True:
        t = d[pos]
        pos += 1
        if t == END:
            return out
        key, pos = _name(d, pos)
        if key != "sections" or t != LIST:
            pos = _skip(d, pos, t)
            continue
        n = _I32.unpack_from(d, pos + 1)[0]
        pos += 5
        for _ in range(n):  # each section is a compound
            y, pal, packed = None, None, None
            while True:
                st = d[pos]
                pos += 1
                if st == END:
                    break
                k, pos = _name(d, pos)
                if k == "Y" and st == BYTE:
                    y = struct.unpack_from(">b", d, pos)[0]
                    pos += 1
                elif k == "block_states" and st == COMPOUND:
                    while True:
                        bt = d[pos]
                        pos += 1
                        if bt == END:
                            break
                        bk, pos = _name(d, pos)
                        end = _skip(d, pos, bt)
                        if bk == "data" and bt == LONG_ARRAY:
                            cnt = _I32.unpack_from(d, pos)[0]
                            packed = np.frombuffer(d, ">i8", cnt, pos + 4).astype(np.int64)
                        elif bk == "palette":
                            pal = bytes([bt]) + bytes(d[pos:end])
                        pos = end
                else:
                    pos = _skip(d, pos, st)
            if y is not None and pal is not None:
                out.append((y, pal, packed))


# --- writer (used for tests and for producing structure files) ---

def _w_payload(out, value, tag):
    if tag == BYTE: out += struct.pack(">b", value)
    elif tag == INT: out += struct.pack(">i", value)
    elif tag == STRING:
        b = value.encode("utf-8"); out += struct.pack(">H", len(b)) + b
    elif tag == LIST:
        sub = _guess(value[0]) if value else END
        out += struct.pack(">bi", sub, len(value))
        for v in value: _w_payload(out, v, sub)
    elif tag == COMPOUND:
        for k, v in value.items():
            t = _guess(v)
            out += struct.pack(">b", t); _w_payload(out, k, STRING); _w_payload(out, v, t)
        out += b"\x00"
    else:
        raise ValueError(tag)


def _guess(v):
    if isinstance(v, bool): return BYTE
    if isinstance(v, int): return INT
    if isinstance(v, str): return STRING
    if isinstance(v, list): return LIST
    if isinstance(v, dict): return COMPOUND
    raise TypeError(type(v))


def dumps(root):
    out = bytearray(struct.pack(">b", COMPOUND))
    _w_payload(out, "", STRING)
    _w_payload(out, root, COMPOUND)
    return gzip.compress(bytes(out))
