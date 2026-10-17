"""Draw the app icon (an isometric block with a cog on its side) as packaging/icon.ico."""
import math
import os

from PIL import Image, ImageDraw

S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
c, r = S / 2, S * 0.46
top = [(c, c - r), (c + r * 0.866, c - r / 2), (c, c), (c - r * 0.866, c - r / 2)]
left = [(c - r * 0.866, c - r / 2), (c, c), (c, c + r), (c - r * 0.866, c + r / 2)]
right = [(c, c), (c + r * 0.866, c - r / 2), (c + r * 0.866, c + r / 2), (c, c + r)]
d.polygon(top, fill=(176, 186, 180), outline=(60, 64, 62), width=4)        # andesite casing
d.polygon(left, fill=(132, 140, 136), outline=(60, 64, 62), width=4)
d.polygon(right, fill=(98, 104, 101), outline=(60, 64, 62), width=4)


gx, gy = c + r * 0.433, c + r * 0.25  # centre of the right face
teeth = []
for k in range(16):
    a = k * math.pi / 8
    rad = 0.33 if k % 2 == 0 else 0.25
    for da in (-0.13, 0.13):
        x, y = math.cos(a + da) * rad, math.sin(a + da) * rad
        teeth.append((gx + r * 0.866 * x, gy + r * (y - 0.5 * x)))
d.polygon(teeth, fill=(222, 170, 70), outline=(110, 74, 20), width=3)      # brass cog
hub = [(gx + r * 0.866 * math.cos(a) * 0.1, gy + r * (math.sin(a) * 0.1 - 0.5 * math.cos(a) * 0.1))
       for a in [i * math.pi / 12 for i in range(24)]]
d.polygon(hub, fill=(90, 62, 30))
out = os.path.join(os.path.dirname(__file__), "icon.ico")
img.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.save(out.replace(".ico", ".png"))
print("wrote", out)
