import sys
from matts_misc.sheared_wcs_wl_sims.run_metacal import run_metacal

if len(sys.argv) > 2:
    wcs_g1 = float(sys.argv[2])
else:
    wcs_g1 = 0.0

if len(sys.argv) > 3:
    wcs_g2 = float(sys.argv[3])
else:
    wcs_g2 = wcs_g1

run_metacal(n_sims=int(sys.argv[1]), wcs_g1=wcs_g1, wcs_g2=wcs_g2)
