from __future__ import print_function
import logging

import ngmix
from ngmix.joint_prior import PriorSimpleSep

try:
    from ngmix.joint_prior import PriorBDFSep, PriorBDSep
    NO_BD_MODELS = False
except ImportError:
    NO_BD_MODELS = True

logger = logging.getLogger(__name__)


class FitterBase(dict):
    def __init__(self, conf, nband, rng):
        self.nband = nband
        self.rng = rng
        self.update(conf)
        self._setup()

    def go(self, mbobs_list):
        """abstract method to do measurements"""
        raise NotImplementedError("implement go()")

    def _get_prior(self, conf):
        if 'priors' not in conf:
            return None

        ppars = conf['priors']
        if ppars.get('prior_from_mof', False):
            return None

        # g
        gp = ppars['g']
        assert gp['type'] == "ba"
        g_prior = self._get_prior_generic(gp)

        if 'T' in ppars:
            size_prior = self._get_prior_generic(ppars['T'])
        elif 'hlr' in ppars:
            size_prior = self._get_prior_generic(ppars['hlr'])
        else:
            raise ValueError('need T or hlr in priors')

        flux_prior = self._get_prior_generic(ppars['flux'])

        # center
        cp = ppars['cen']
        assert cp['type'] == 'normal2d'
        cen_prior = self._get_prior_generic(cp)

        if 'bd' in conf['model']:
            assert 'fracdev' in ppars, (
                "set fracdev prior for bdf and bd models")
            assert not NO_BD_MODELS, (
                "Using BD models requires ngmix with the proper priors."
                " Try updating!")

        if conf['model'] == 'bd':
            assert 'logTratio' in ppars, "set logTratio prior for bd model"
            fp = ppars['fracdev']
            logTratiop = ppars['logTratio']

            fracdev_prior = self._get_prior_generic(fp)
            logTratio_prior = self._get_prior_generic(logTratiop)

            prior = PriorBDSep(
                cen_prior,
                g_prior,
                size_prior,
                logTratio_prior,
                fracdev_prior,
                [flux_prior]*self.nband,
            )

        elif conf['model'] == 'bdf':
            fp = ppars['fracdev']

            fracdev_prior = self._get_prior_generic(fp)

            prior = PriorBDFSep(
                cen_prior,
                g_prior,
                size_prior,
                fracdev_prior,
                [flux_prior]*self.nband,
            )

        else:
            prior = PriorSimpleSep(
                cen_prior,
                g_prior,
                size_prior,
                [flux_prior]*self.nband,
            )

        return prior

    def _get_prior_generic(self, ppars):
        """
        get a prior object using the input specification
        """
        ptype = ppars['type']
        bounds = ppars.get('bounds', None)

        if ptype == "flat":
            assert bounds is None, 'bounds not supported for flat'
            prior = ngmix.priors.FlatPrior(*ppars['pars'], rng=self.rng)

        elif ptype == "bounds":
            prior = ngmix.priors.LMBounds(*ppars['pars'], rng=self.rng)

        elif ptype == 'two-sided-erf':
            assert bounds is None, 'bounds not supported for erf'
            prior = ngmix.priors.TwoSidedErf(*ppars['pars'], rng=self.rng)

        elif ptype == 'sinh':
            assert bounds is None, 'bounds not supported for Sinh'
            prior = ngmix.priors.Sinh(
                ppars['mean'], ppars['scale'], rng=self.rng)

        elif ptype == 'normal':
            prior = ngmix.priors.Normal(
                ppars['mean'],
                ppars['sigma'],
                bounds=bounds,
                rng=self.rng,
            )

        elif ptype == 'truncated-normal':
            assert 'do not use truncated normal'
            prior = ngmix.priors.TruncatedGaussian(
                mean=ppars['mean'],
                sigma=ppars['sigma'],
                minval=ppars['minval'],
                maxval=ppars['maxval'],
                rng=self.rng,
            )

        elif ptype == 'log-normal':
            assert bounds is None, 'bounds not yet supported for LogNormal'
            if 'shift' in ppars:
                shift = ppars['shift']
            else:
                shift = None
            prior = ngmix.priors.LogNormal(
                ppars['mean'],
                ppars['sigma'],
                shift=shift,
                rng=self.rng,
            )

        elif ptype == 'normal2d':
            assert bounds is None, 'bounds not yet supported for Normal2D'
            prior = ngmix.priors.CenPrior(
                0.0,
                0.0,
                ppars['sigma'],
                ppars['sigma'],
                rng=self.rng,
            )

        elif ptype == 'ba':
            assert bounds is None, 'bounds not supported for BA'
            prior = ngmix.priors.GPriorBA(ppars['sigma'], rng=self.rng)

        else:
            raise ValueError("bad prior type: '%s'" % ptype)

        return prior
