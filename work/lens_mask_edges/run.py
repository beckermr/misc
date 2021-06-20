#!/usr/bin.env python
from mdet_tools import meas_m

if __name__ == "__main__":
    m, merr, c, cerr = meas_m(n_stars=20, mask_width=4, seed=20, n_jobs=100000)
