import numpy as np
import tqdm
import glob
import pickle
import joblib
import itertools


def estimate_m_and_c(results):
    """Estimate m and c from paired lensing simulations.

    Parameters
    ----------
    results : list of iterables
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a `g1` shear in the 1-component and
        0 true shear in the 2-component.

    Returns
    -------
    m : float
        Estimate of the multiplicative bias.
    merr : float
        Estimat of the 1-sigma standard error in `m`.
    c : float
        Estimate of the additive bias.
    cerr : float
        Estimate of the 1-sigma standard error in `c`.
    """

    g_true = 0.02
    step = 0.01

    def _get_stuff(rr):
        _a = np.vstack(rr)
        g1p = _a[:, 0]
        g1m = _a[:, 1]
        g1 = _a[:, 2]
        g2p = _a[:, 3]
        g2m = _a[:, 4]
        g2 = _a[:, 5]

        return (
            g1, (g1p - g1m) / 2 / step * g_true,
            g2, (g2p - g2m) / 2 / step)

    g1, R11, g2, R22 = _get_stuff(results)

    msk = (
        np.isfinite(g1) &
        np.isfinite(R11) &
        np.isfinite(g2) &
        np.isfinite(R22))
    g1 = g1[msk]
    R11 = R11[msk]
    g2 = g2[msk]
    R22 = R22[msk]

    x1 = R11
    y1 = g1
    x2 = R22
    y2 = g2

    rng = np.random.RandomState(seed=100)
    mvals = []
    cvals = []
    for _ in tqdm.trange(500, leave=False):
        ind = rng.choice(len(y1), replace=True, size=len(y1))
        mvals.append(np.mean(y1[ind]) / np.mean(x1[ind]) - 1)
        cvals.append(np.mean(y2[ind]) / np.mean(x2[ind]))

    return (
        np.mean(y1) / np.mean(x1) - 1, np.std(mvals),
        np.mean(y2) / np.mean(x2), np.std(cvals))


def _func(fname):
    try:
        with open(fname, 'rb') as fp:
            data = pickle.load(fp)
        return data
    except Exception:
        return []


tmpdir = 'outputs'

files = glob.glob('%s/out*.pkl' % tmpdir)
print('found %d outputs' % len(files))
io = [joblib.delayed(_func)(fname) for fname in files]
outputs = joblib.Parallel(
    verbose=10,
    n_jobs=-1,
    pre_dispatch='2*n_jobs',
    max_nbytes=None)(io)

s2n = 10
pres = [i for i in itertools.chain.from_iterable(outputs)]
m, msd, c, csd = estimate_m_and_c(pres)

print('s2n:', s2n)
print("""\
# of sims: {n_sims}
m       : {m:f} +/- {msd:f}
c       : {c:f} +/- {csd:f}""".format(
    n_sims=len(pres),
    m=m,
    msd=msd,
    c=c,
    csd=csd), flush=True)
