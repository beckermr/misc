import sys
import pickle
import numpy as np

from matts_misc.piff_wl_sims.run_metacal import run_metacal
from config import jacobian_dict, gauss_psf, n_sims


stamp_size = 33
psf_stamp_size = 53
rng = np.random.RandomState(seed=int(sys.argv[1]))

res = run_metacal(
    n_sims, stamp_size, psf_stamp_size, rng,
    jacobian_dict, gauss_psf)


def _mask(cat):
    msk = (
        (cat['mcal_s2n'] > 10) &
        (cat['mcal_T_ratio'] > 0.5))
    return msk


msk = _mask(res['noshear'])
e1 = np.mean(res['noshear']['mcal_g'][msk, 0])
e2 = np.mean(res['noshear']['mcal_g'][msk, 1])

msk = _mask(res['1p'])
e1p = np.mean(res['1p']['mcal_g'][msk, 0])

msk = _mask(res['1m'])
e1m = np.mean(res['1m']['mcal_g'][msk, 0])

msk = _mask(res['2p'])
e2p = np.mean(res['2p']['mcal_g'][msk, 1])

msk = _mask(res['2m'])
e2m = np.mean(res['2m']['mcal_g'][msk, 1])

with open(sys.argv[2], 'wb') as fp:
    pickle.dump((e1p, e1m, e1, e2p, e2m, e2), fp)
