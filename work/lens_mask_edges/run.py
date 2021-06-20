#!/usr/bin.env python
from mdet_tools import meas_m

if __name__ == "__main__":
    n_stars = 20
    m, merr, c, cerr, _, _ = meas_m(
        n_stars=n_stars, mask_width=4, seed=20, n_jobs=5_000_000
    )

    print("\nstar density: %s [num per arcmin^2]" % (n_stars / (235*0.20/60)**2))
    mstr = "m +/- merr: %0.6f +/- %0.6f [10^(-3), 3sigma]" % (m/1e-3, 3*merr/1e-3)
    print(mstr, flush=True)

    cstr = "c +/- cerr: %0.6f +/- %0.6f [10^(-5), 3sigma]" % (c/1e-3, 3*cerr/1e-3)
    print(cstr, flush=True)
