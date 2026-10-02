from cvd.engine import *

for kind in ["protanopia", "deuteranopia", "tritanopia"]:
       rank, s, null_dir, check = null_space_info(kind)
       print(kind)
       print("  rank:", rank)
       print("  singular values:", s)
       print("  null direction:", null_dir)
       print("  matrix x null direction (should be ~0):", check)
       print("")