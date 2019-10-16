import numpy as np
import glob
import pickle
import joblib
import itertools
import numba


@numba.njit
def _jack_est(g1, R11, g2, R22):
    g1bar = np.mean(g1)
    R11bar = np.mean(R11)
    g2bar = np.mean(g2)
    R22bar = np.mean(R22)
    n = g1.shape[0]
    fac = n / (n-1)
    m_samps = np.zeros_like(g1)
    c_samps = np.zeros_like(g1)

    for i in range(n):
        _g1 = fac * (g1bar - g1[i]/n)
        _R11 = fac * (R11bar - R11[i]/n)
        _g2 = fac * (g2bar - g2[i]/n)
        _R22 = fac * (R22bar - R22[i]/n)
        m_samps[i] = _g1 / _R11 - 1
        c_samps[i] = _g2 / _R22

    m = np.mean(m_samps)
    c = np.mean(c_samps)

    m_err = np.sqrt(np.sum((m - m_samps)**2) / fac)
    c_err = np.sqrt(np.sum((c - c_samps)**2) / fac)

    return m, m_err, c, c_err


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

    return _jack_est(g1, R11, g2, R22)


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

if np.abs(m) < 0.01:
    print("""\
    # of sims: {n_sims}
    m [1e-3]: {m:f} +/- {msd:f}
    c [1e-4]: {c:f} +/- {csd:f}""".format(
        n_sims=len(pres),
        m=m/1e-3,
        msd=msd/1e-3,
        c=c/1e-4,
        csd=csd/1e-4), flush=True)
else:
    print("""\
    # of sims: {n_sims}
    m       : {m:f} +/- {msd:f}
    c       : {c:f} +/- {csd:f}""".format(
        n_sims=len(pres),
        m=m,
        msd=msd,
        c=c,
        csd=csd), flush=True)
