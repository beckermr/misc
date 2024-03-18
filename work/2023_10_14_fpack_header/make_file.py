import subprocess
import numpy as np
import fitsio

hdr = {
    "FZQVALUE": 4,
    "FZTILE": "(10240,1)",
    "FZALGOR": "RICE_1",
    "FZQMETHD": "SUBTRACTIVE_DITHER_2",
    "ZDITHER0": 25,
}

rng = np.random.default_rng(42)
d = np.ones((100, 100), dtype=np.float64)
d += rng.normal(size=d.shape, scale=0.02)

with fitsio.FITS('data.fits', 'rw', clobber=True) as f:
    f.write(d, header=hdr)
    f.write(d+2, header=hdr)

subprocess.run(
    "rm -f data.fits.fz && fpack -qz42 4 data.fits",
    shell=True,
    check=True,
)

for ext in [0, 1, 2, 3]:
    try:
        hdr = fitsio.read_header('data.fits.fz', ext=ext)
        print("\nEXT=%d\n" % ext, hdr)
    except Exception:
        pass
