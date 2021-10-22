from __future__ import print_function
import logging
import numpy as np
import esutil as eu

from ngmix import ObsList, MultiBandObsList
from ngmix.gexceptions import BootPSFFailure, BootGalFailure
from . import procflags

from ngmix.runners import Runner, PSFRunner
from ngmix.guessers import SimplePSFGuesser, TFluxAndPriorGuesser
from ngmix.fitting import Fitter
from ngmix.metacal import MetacalBootstrapper
from ngmix.gaussmom import GaussMom

from .base_fitter import FitterBase
from .util import Namer

logger = logging.getLogger(__name__)

METACAL_TYPES = ['noshear', '1p', '1m', '2p', '2m']


class MetacalFitter(FitterBase):
    """Run metacal on all a list of observations.

    Parameters
    ----------
    conf : dict
        A configuration dictionary.
    nband : int
        The number of bands.
    rng : np.random.RandomState
        An RNG instance.

    Methods
    -------
    go(mbobs_list_input)
    """
    def __init__(self, conf, nband, rng):
        super().__init__(conf, nband, rng)

    def _setup(self):
        self.metacal_prior = self._get_prior(self['metacal'])
        assert self.metacal_prior is not None
        self['metacal']['symmetrize_weight'] = self['metacal'].get(
            'symmetrize_weight', False)

        # be safe friends!
        if 'types' in self['metacal']:
            assert self['metacal']['types'] == METACAL_TYPES

    @property
    def result(self):
        """Get the result data"""
        if not hasattr(self, '_result'):
            raise RuntimeError('run go() first')

        if self._result is not None:
            return self._result.copy()
        else:
            return None

    def go(self, mbobs_list):
        """Run metcal on a list of MultiBandObsLists.

        Parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object.  If it is a simple MultiBandObsList it will
            be converted to a list
        """
        if not isinstance(mbobs_list, list):
            mbobs_list = [mbobs_list]

        mbobs_list_mcal = mbobs_list
        self.mbobs_list_mcal = mbobs_list_mcal

        if self['metacal']['symmetrize_weight']:
            self._symmetrize_weights(mbobs_list_mcal)

        self._result = self._do_all_metacal(mbobs_list_mcal)

    def _symmetrize_weights(self, mbobs_list):
        def _symmetrize_weight(wt):
            assert wt.shape[0] == wt.shape[1]
            wt_rot = np.rot90(wt)
            wzero = wt_rot == 0.0
            if np.any(wzero):
                wt[wzero] = 0.0

        for mbobs in mbobs_list:
            for obslist in mbobs:
                for obs in obslist:
                    _symmetrize_weight(obs.weight)

    def _do_all_metacal(self, mbobs_list):
        """run metacal on all objects

        NOTE: failed mbobs will have no entries in the final list
        """

        assert len(mbobs_list[0]) == self.nband

        datalist = []
        for i, _mbobs in enumerate(mbobs_list):
            passed_flags, mbobs = self._check_flags(_mbobs)
            if passed_flags:
                try:
                    res = self._do_one_metacal_ngmixv2(mbobs)
                except (BootPSFFailure, BootGalFailure) as err:
                    logger.debug(str(err))
                    res = {'mcal_flags': 1}

                if res['mcal_flags'] != 0:
                    logger.debug("metacal fit failed")
                else:
                    # make sure we send an array
                    fit_data = self._get_metacal_output(res, self.nband, mbobs)
                    self._print_result(fit_data)
                    datalist.append(fit_data)

        if len(datalist) == 0:
            return None

        output = eu.numpy_util.combine_arrlist(datalist)
        return output

    def _do_one_metacal_ngmixv2(self, mbobs):
        gm = GaussMom(1.2).go(mbobs[0][0])
        if gm['flags'] == 0:
            flux_guess = gm['flux']
            Tguess = gm['T']
        else:
            gm = GaussMom(1.2).go(mbobs[0][0].psf)
            if gm['flags'] == 0:
                Tguess = 2 * gm['T']
            else:
                Tguess = 2
            flux_guess = np.sum(mbobs[0][0].image)

        guesser = TFluxAndPriorGuesser(
            rng=self.rng, T=Tguess, flux=flux_guess, prior=self.metacal_prior,
        )
        psf_guesser = SimplePSFGuesser(rng=self.rng, guess_from_moms=True)

        fitter = Fitter(
            model=self['metacal']['model'],
            fit_pars=self['metacal']['max_pars']['pars']['lm_pars'],
            prior=self.metacal_prior
        )
        psf_fitter = Fitter(
            model=self['metacal']['psf']['model'],
            fit_pars=self['metacal']['psf']['lm_pars'],
        )

        psf_runner = PSFRunner(
            fitter=psf_fitter,
            guesser=psf_guesser,
            ntry=self['metacal']['psf']['ntry'],
        )
        runner = Runner(
            fitter=fitter,
            guesser=guesser,
            ntry=self['metacal']['max_pars']['ntry'],
        )

        boot = MetacalBootstrapper(
            runner=runner, psf_runner=psf_runner,
            rng=self.rng, **self['metacal']['metacal_pars'],
        )
        resdict, _ = boot.go(mbobs)
        flags = 0
        for key in resdict:
            flags |= resdict[key]['flags']
        resdict['mcal_flags'] = flags

        psf_T = 0.0
        psf_g = np.zeros(2)
        wsum = 0.0
        for obslist in mbobs:
            for obs in obslist:
                msk = obs.weight > 0
                if np.any(msk) and 'result' in obs.psf.meta:
                    wgt = np.median(obs.weight[msk])
                    psf_T += wgt * obs.psf.meta['result']['T']
                    psf_g += wgt * obs.psf.meta['result']['e']
                    wsum += wgt
        if wsum > 0:
            psf_T /= wsum
            psf_g /= wsum
            resdict['noshear']['Tpsf'] = psf_T
            resdict['noshear']['gpsf'] = psf_g
        else:
            resdict['mcal_flags'] |= procflags.NO_DATA
            resdict['noshear']['Tpsf'] = -9999.0
            resdict['noshear']['gpsf'] = np.array([-9999.0, -9999.0])

        return resdict

    def _check_flags(self, mbobs):
        flags = self['metacal'].get('bmask_flags', None)
        passed_flags = True
        _mbobs = None

        if flags is not None:
            _mbobs = MultiBandObsList()
            _mbobs.update_meta_data(mbobs.meta)
            for obslist in mbobs:
                _obslist = ObsList()
                _obslist.update_meta_data(obslist.meta)

                for obs in obslist:
                    msk = (obs.bmask & flags) != 0
                    if np.any(msk):
                        logger.info("   EDGE HIT")
                    else:
                        _obslist.append(obs)
                        passed_flags = True

                _mbobs.append(_obslist)

            # all bands have to have at least one obs
            for ol in _mbobs:
                if len(ol) == 0:
                    passed_flags = False

        return passed_flags, _mbobs

    def _print_result(self, data):
        logger.debug(
            "    mcal s2n: %g Trat: %g",
            data['mcal_s2n_noshear'][0],
            data['mcal_T_ratio_noshear'][0])

    def _get_metacal_dtype(self, npars, nband):
        dt = [
            ('x', 'f8'),
            ('y', 'f8'),
            ('mcal_flags', 'i8'),
        ]
        for mtype in METACAL_TYPES:
            n = Namer(front='mcal', back=mtype)
            if mtype == 'noshear':
                dt += [
                    (n('psf_g'), 'f8', 2),
                    (n('psf_T'), 'f8'),
                ]

            dt += [
                (n('nfev'), 'i4'),
                (n('s2n'), 'f8'),
                (n('s2n_r'), 'f8'),
                (n('pars'), 'f8', npars),
                (n('pars_cov'), 'f8', (npars, npars)),
                (n('g'), 'f8', 2),
                (n('g_cov'), 'f8', (2, 2)),
                (n('T'), 'f8'),
                (n('T_err'), 'f8'),
                (n('T_ratio'), 'f8'),
                (n('flux'), 'f8', (nband,)),
                (n('flux_cov'), 'f8', (nband, nband)),
                (n('flux_err'), 'f8', (nband,)),
            ]

        return dt

    def _get_metacal_output(self, allres, nband, mbobs):
        # assume one epoch and line up in all
        # bands
        # FIXME? IDK why this was here
        # assert len(mbobs[0]) == 1, 'one epoch only'

        # needed for the code below
        assert METACAL_TYPES[0] == 'noshear'

        if 'T_r' in allres['noshear'] and 'T' not in allres['noshear']:
            do_round = True
        else:
            do_round = False

        npars = len(allres['noshear']['pars'])
        dt = self._get_metacal_dtype(npars, nband)
        data = np.zeros(1, dtype=dt)

        data0 = data[0]
        data0['y'] = mbobs[0][0].meta['orig_row']
        data0['x'] = mbobs[0][0].meta['orig_col']
        data0['mcal_flags'] = 0

        for mtype in METACAL_TYPES:
            n = Namer(front='mcal', back=mtype)

            res = allres[mtype]

            if mtype == 'noshear':
                data0[n('psf_g')] = res['gpsf']
                if do_round:
                    data0[n('psf_T')] = res['psf_T_r']
                else:
                    data0[n('psf_T')] = res['Tpsf']

            for name in res:
                if do_round and name == 'T_r':
                    nn = n('T')
                else:
                    nn = n(name)
                if nn in data.dtype.names:
                    data0[nn] = res[name]

            # this relies on noshear coming first in the metacal
            # types
            data0[n('T_ratio')] = data0[n('T')]/data0['mcal_psf_T_noshear']

        return data
