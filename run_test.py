from cvd.engine import *

img = load_image("test.png")
kind = "deuteranopia"

fixed = correct(img, kind, alpha=1.0)
fixed = emphasize_figure(fixed, img)

save_image(fixed, "final_corrected.png")
save_image(simulate(fixed, kind), "final_after_as_seen.png")
print("done")