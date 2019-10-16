import logging
import numpy as np

import ngmix
from ngmix import format_pars
from ngmix.gexceptions import BootPSFFailure
from ngmix.joint_prior import PriorSimpleSep, PriorBDFSep, PriorBDSep


from .util import NoDataError

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


def _fit_all_psfs(mbobs_list, psf_conf):
    """
    fit all psfs in the input observations
    """
    fitter = AllPSFFitter(mbobs_list, psf_conf)
    fitter.go()


def _measure_all_psf_fluxes(mbobs_list):
    """
    fit all psfs in the input observations
    """
    fitter = AllPSFFluxFitter(mbobs_list)
    fitter.go()


class AllPSFFitter(object):
    def __init__(self, mbobs_list, psf_conf):
        self.mbobs_list = mbobs_list
        self.psf_conf = psf_conf

    def go(self):
        for mbobs in self.mbobs_list:
            for obslist in mbobs:
                for obs in obslist:
                    psf_obs = obs.get_psf()
                    _fit_one_psf(psf_obs, self.psf_conf)


def _fit_one_psf(obs, pconf):
    Tguess = 4.0*obs.jacobian.get_scale()**2

    if 'coellip' in pconf['model']:
        ngauss = ngmix.bootstrap.get_coellip_ngauss(pconf['model'])
        runner = ngmix.bootstrap.PSFRunnerCoellip(
            obs,
            Tguess,
            ngauss,
            pconf['lm_pars'],
        )

    elif 'em' in pconf['model']:
        ngauss = ngmix.bootstrap.get_em_ngauss(pconf['model'])
        runner = ngmix.bootstrap.EMRunner(
            obs,
            Tguess,
            ngauss,
            pconf['em_pars'],
        )

    else:
        runner = ngmix.bootstrap.PSFRunner(
            obs,
            pconf['model'],
            Tguess,
            pconf['lm_pars'],
        )

    runner.go(ntry=pconf['ntry'])

    psf_fitter = runner.fitter
    res = psf_fitter.get_result()
    obs.update_meta_data({'fitter': psf_fitter})

    if res['flags'] == 0:
        gmix = psf_fitter.get_gmix()
        obs.set_gmix(gmix)
    else:
        raise BootPSFFailure("failed to fit psfs: %s" % str(res))


class AllPSFFluxFitter(object):
    def __init__(self, mbobs_list):
        self.mbobs_list = mbobs_list

    def go(self):
        for mbobs in self.mbobs_list:
            for band, obslist in enumerate(mbobs):

                if len(obslist) == 0:
                    raise NoDataError('no data in band %d' % band)

                meta = obslist.meta

                res = self._fit_psf_flux(band, obslist)
                meta['psf_flux_flags'] = res['flags']

                for n in ('psf_flux', 'psf_flux_err', 'psf_flux_s2n'):
                    meta[n] = res[n.replace('psf_', '')]

    def _fit_psf_flux(self, band, obslist):
        fitter = ngmix.fitting.TemplateFluxFitter(
            obslist,
            do_psf=True,
        )
        fitter.go()

        res = fitter.get_result()

        if res['flags'] == 0 and res['flux_err'] > 0:
            res['flux_s2n'] = res['flux']/res['flux_err']
        else:
            res['flux_s2n'] = -9999.0
            raise BootPSFFailure(
                "failed to fit psf fluxes for band %d: %s" % (band, str(res)))

        return res


def get_stamp_guesses(list_of_obs, detband, model, rng, prior=None):
    """
    get a guess based on metadata in the obs

    T guess is gotten from detband
    """

    nband = len(list_of_obs[0])

    if model == 'bd':
        npars_per = 7+nband
    elif model == 'bdf':
        npars_per = 6+nband
    else:
        npars_per = 5+nband

    nobj = len(list_of_obs)

    npars_tot = nobj*npars_per
    guess = np.zeros(npars_tot)

    for i, mbo in enumerate(list_of_obs):
        detobslist = mbo[detband]
        detmeta = detobslist.meta

        T = detmeta['Tsky']

        beg = i*npars_per

        # always close guess for center
        rowguess, colguess = prior.cen_prior.sample()
        guess[beg+0] = rowguess
        guess[beg+1] = colguess

        # always arbitrary guess for shape
        guess[beg+2] = rng.uniform(low=-0.05, high=0.05)
        guess[beg+3] = rng.uniform(low=-0.05, high=0.05)

        guess[beg+4] = T*(1.0 + rng.uniform(low=-0.05, high=0.05))

        if 'bd' in model:
            if hasattr(prior.fracdev_prior, 'sigma'):
                # guessing close to mean seems to be important for the
                # pathological cases of an undetected object close to another
                low = prior.fracdev_prior.mean - 0.1*prior.fracdev_prior.sigma
                high = prior.fracdev_prior.mean + 0.1*prior.fracdev_prior.sigma
                while True:
                    fracdev_guess = rng.uniform(low=low, high=high)
                    if 0 < fracdev_guess < 1:
                        break
            else:
                fracdev_guess = prior.fracdev_prior.sample()

        if model == 'bd':
            low = prior.logTratio_prior.mean - 0.1*prior.logTratio_prior.sigma
            high = prior.logTratio_prior.mean + 0.1*prior.logTratio_prior.sigma
            logTratio_guess = rng.uniform(low=low, high=high)

            guess[beg+5] = logTratio_guess
            guess[beg+6] = fracdev_guess
            flux_start = 7

        elif model == 'bdf':
            guess[beg+5] = fracdev_guess
            flux_start = 6

        else:
            flux_start = 5

        for band, obslist in enumerate(mbo):
            obslist = mbo[band]
            band_meta = obslist.meta

            # note we take out scale**2 in DES images when
            # loading from MEDS so this isn't needed
            flux = band_meta['psf_flux']
            low = flux
            high = flux*2.0
            flux_guess = rng.uniform(low=low, high=high)
            guess[beg+flux_start+band] = flux_guess

        # fix fluxes
        fluxes = guess[beg+flux_start:beg+flux_start+nband]
        logic = np.isfinite(fluxes) & (fluxes > 0.0)
        wgood, = np.where(logic == True)  # noqa
        if wgood.size != nband:
            logging.info('fixing bad flux guesses: %s' % format_pars(fluxes))
            if wgood.size == 0:
                fluxes[:] = rng.uniform(low=100, high=200)
            else:
                wbad, = np.where(logic == False)  # noqa
                fac = 1.0 + rng.uniform(low=-0.1, high=0.1, size=wbad.size)
                fluxes[wbad] = fluxes[wgood].mean()*fac
            logging.info('new guesses: %s' % format_pars(fluxes))

        logger.debug('guess[%d]: %s' % (
            i, format_pars(guess[beg:beg+flux_start+band+1])))
    return guess
