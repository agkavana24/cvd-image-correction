from cvd.engine import *

img = load_image("test.png")
kind = "deuteranopia"

save_image(simulate(img, kind), "before_as_seen.png")

for a in [0.5, 1.0, 1.5]:
    fixed = correct(img, kind, alpha=a)
    save_image(fixed, f"fixed_{a}.png")
    save_image(simulate(fixed, kind), f"fixed_{a}_as_seen.png")
    print("saved alpha", a)