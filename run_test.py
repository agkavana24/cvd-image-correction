from cvd.engine import *

img = load_image("test.png")
lin = srgb_to_linear(img)
back = linear_to_srgb(lin)
save_image(back, "roundtrip.png")
print("max difference:", abs(img - back).max())