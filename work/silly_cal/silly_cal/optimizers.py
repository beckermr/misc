import numpy as np
import tqdm


def run_amsgrad(
    fun, x, alpha, nsteps, beta1=0.9, beta2=0.99,
    alpha_decay_pow=0.1,
    ostate=None, eps=1e-6, atol=1e-6, rtol=1e-6, ntol=100
):
    """Run AMSgrad on a loss function. See https://arxiv.org/pdf/1904.09237.pdf

    Parameters
    ----------
    fun : callable
        Should return the value and the gradient as 2-tuple.
    x : array-like
        The initial guess.
    alpha : float
        The learning rate. A good default is 1e-3.
    nsteps : int
        The number of steps to take.
    beta1 : float, optional
        Parameter of AMSgrad. Leave it at the default of 0.9.
    beta2 : float, optional
        Parameter of AMSgrad. Leave it at the default of 0.99.
    alpha_decay_pow : float, optional
        The decay power of the learning rate (
        i.e. alpha ~ alpha / t**alpha_decay_pow).
    ostate : dict, optional
        An output of a previous optimization. If passed, the optimization
        is resumed.
    eps : float, optional
        A kernel for stability. Leave at the default of 1e-6.
    atol : float, optional
        The absolute tolerance for convergence of the loss.
    rtol : float, optional
        The relative tolerance for convergence of the loss.
    ntol : int, optional
        The loss compared to the previous iteration must not differ by more
        than the absolute or relative tolerances for this many iterations
        in order for the optimizer to stop early.

    Returns
    -------
    ostate : dict
        A dictionary of optimization results:

            x : the final parameter location
            fun : the function value at the optimum
            opt : the optimizer state
            flags : a bit flag field - zero indicates convergence
    """
    if ostate is None:
        ostate = {}
        ostate["opt"] = dict(
            m=0.0,
            v=0.0,
            max_v=0.0,
            t=0,
            fval_prev=None,
        )
    m = ostate["opt"]["m"]
    v = ostate["opt"]["v"]
    max_v = ostate["opt"]["max_v"]
    tstart = ostate["opt"]["t"]
    fval_prev = ostate["opt"]["fval_prev"]
    nok = 0
    ostate["flags"] = 1
    with tqdm.trange(nsteps, ncols=80) as itr:
        for t in itr:
            fval, g = fun(x)
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g*g
            max_v = np.maximum(max_v, v)
            x -= (
                alpha
                / np.power(t + tstart + 1, alpha_decay_pow)
                * m / np.sqrt(max_v + eps)
            )

            if fval_prev is not None:
                if (
                    np.abs(fval - fval_prev)
                    <= (atol + rtol * np.abs(fval_prev))
                ):
                    nok += 1
                    if nok == ntol:
                        ostate["flags"] = 0
                        break
            fval_prev = fval

            itr.set_description("chi2/dof = %0.8e" % fval)

    ostate["x"] = x
    ostate["opt"] = dict(
        m=m,
        v=v,
        max_v=max_v,
        t=t + tstart,
        fval_prev=fval_prev,
    )
    ostate["fun"] = fval
    return ostate


def run_laprop(
    fun, x, alpha, nsteps, mu=0.9, nu=0.99,
    alpha_decay_pow=0.1,
    ostate=None, eps=1e-6, atol=1e-6, rtol=1e-6, ntol=100
):
    """Run LaProp on a loss function. See https://arxiv.org/pdf/2002.04839.pdf.

    Parameters
    ----------
    fun : callable
        Should return the value and the gradient as 2-tuple.
    x : array-like
        The initial guess.
    alpha : float
        The learning rate. A good default is 1e-3.
    nsteps : int
        The number of steps to take.
    mu : float, optional
        Parameter of LaProp. Leave it at the default of 0.9.
    nu : float, optional
        Parameter of LaProp. Leave it at the default of 0.99.
    alpha_decay_pow : float, optional
        The decay power of the learning rate (
        i.e. alpha ~ alpha / t**alpha_decay_pow).
    ostate : dict, optional
        An output of a previous optimization. If passed, the optimization
        is resumed.
    eps : float, optional
        A kernel for stability. Leave at the default of 1e-6.
    atol : float, optional
        The absolute tolerance for convergence of the loss.
    rtol : float, optional
        The relative tolerance for convergence of the loss.
    ntol : int, optional
        The loss compared to the previous iteration must not differ by more
        than the absolute or relative tolerances for this many iterations
        in order for the optimizer to stop early.

    Returns
    -------
    ostate : dict
        A dictionary of optimization results:

            x : the final parameter location
            fun : the function value at the optimum
            opt : the optimizer state
            flags : a bit flag field - zero indicates convergence
    """
    if ostate is None:
        ostate = {}
        ostate["opt"] = dict(
            mt=0.0,
            nt=0.0,
            t=0,
            fval_prev=None,
        )
    mt = ostate["opt"]["mt"]
    nt = ostate["opt"]["nt"]
    tstart = ostate["opt"]["t"]
    fval_prev = ostate["opt"]["fval_prev"]
    nok = 0
    ostate["flags"] = 1
    with tqdm.trange(nsteps, ncols=80) as itr:
        for _t in itr:
            t = _t + tstart
            fval, g = fun(x)
            cn = 1.0 / (1.0 - np.power(nu, t+1))
            cm = 1.0 / (1.0 - np.power(mu, t+1))
            nt = nu * nt + (1.0 - nu) * g*g
            mt = mu * mt + (1.0 - mu) * g / (np.sqrt(nt/cn) + eps)

            x -= (
                alpha
                / np.power(t + 1, alpha_decay_pow)
                * mt / cm
            )

            if fval_prev is not None:
                if (
                    np.abs(fval - fval_prev)
                    <= (atol + rtol * np.abs(fval_prev))
                ):
                    nok += 1
                    if nok == ntol:
                        ostate["flags"] = 0
                        break
            fval_prev = fval

            itr.set_description("chi2/dof = %0.8e" % fval)

    ostate["x"] = x
    ostate["opt"] = dict(
        mt=mt,
        nt=nt,
        t=_t + tstart,
        fval_prev=fval_prev,
    )
    ostate["fun"] = fval
    return ostate
