import numpy as np
import galsim
import ngmix
from ngmix import Observation, DiagonalJacobian, ObsList, MultiBandObsList
import ngmix.metacal
import esutil as eu
import tqdm
import click
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

NO_ATTEMPT = 2**0
IMAGE_FLAGS = 2**1
PSF_FAILURE = 2**2
OBJ_FAILURE = 2**3


def make_mcal_obs(mbobs_list, rng):
    # assert len(mbobs) == 1 and len(mbobs[0]) == 1

    mcal_dicts = [
        ngmix.metacal.get_all_metacal(
            mbobs,
            rng=rng,
            psf='fitgauss',
            types=['noshear', '1p', '1m', '2p', '2m'],
            fix_noise=True,
            step=0.01,
            use_noise_image=True,
        )
        for mbobs in mbobs_list]
    final = {
        k: []
        for k in mcal_dicts[0].keys()
    }
    for mcal_dict in mcal_dicts:
        for k, v in mcal_dict.items():
            final[k].append(v)
    return final


class Namer(object):
    """
    create strings with a specified front prefix
    """
    def __init__(self, front=None, back=None):
        if front == '':
            front = None
        if back == '' or back == 'noshear':
            back = None

        self.front = front
        self.back = back

        if self.front is None and self.back is None:
            self.nomod = True
        else:
            self.nomod = False

    def __call__(self, name):
        n = name
        if not self.nomod:
            if self.front is not None:
                n = '%s_%s' % (self.front, n)
            if self.back is not None:
                n = '%s_%s' % (n, self.back)

        return n


class FitterBase(dict):
    """
    we don't create a new instance of this for each fit, because
    the prior can be set once
    """
    def __init__(self, config, rng):

        self.rng = rng
        self.update(config)

    def go(self, mbobs_list):
        """
        do measurements.
        """
        raise NotImplementedError("implement go()")


class Moments(FitterBase):

    """
    measure simple weighted moments
    """
    def __init__(self, *args, **kw):
        super(Moments, self).__init__(*args, **kw)
        self._set_mompars()

    def go(self, mbobs_list):
        """
        run moments measurements on all objects

        parameters
        ----------
        mbobs_list: list of ngmix.MultiBandObsList
            One for each object to be measured

        returns
        -------
        output: list of numpy arrays with fields
            Results for each object
        """

        datalist = []
        for i, mbobs in enumerate(mbobs_list):

            obs = self._do_coadd_maybe(mbobs)

            pres = self._measure_moments(obs.psf)
            res = self._measure_moments(obs)

            if res['flags'] != 0:
                print("        moments failed: %s" % res['flagstr'])

            if pres['flags'] != 0:
                print('        psf moments failed: %s' % pres['flagstr'])

            fit_data = self._get_output(res, pres)

            if res['flags'] == 0 and pres['flags'] == 0:
                self._print_result(fit_data)

            datalist.append(fit_data)

        if len(datalist) == 0:
            return None
        else:
            return eu.numpy_util.combine_arrlist(datalist)

    def _get_max_psf_size(self, mbobs):
        sizes = []
        for obslist in mbobs:
            for obs in obslist:
                sizes.append(obs.psf.image.shape[0])
        return max(sizes)

    def _maybe_zero_pad_image(self, im, size):
        if im.shape[0] == size:
            return im
        elif im.shape[0] < size:
            diff = size - im.shape[0]
            assert diff % 2 == 0, "Can only pad images with even padding!"
            half_diff = diff // 2
            new_im = np.zeros((size, size), dtype=im.dtype)
            new_im[half_diff:-half_diff, half_diff:-half_diff] = im

            newcen = (size - 1) // 2
            oldcen = (im.shape[0] - 1) // 2
            assert new_im[newcen, newcen] == im[oldcen, oldcen]
            return new_im
        else:
            raise ValueError("cannot pad image to a smaller size!")

    def _do_coadd_maybe(self, mbobs):
        """
        coadd all images and psfs.  Assume perfect registration and
        same wcs
        """

        # note here assuming we can re-use the wcs etc.
        new_obs = mbobs[0][0].copy()

        assert len(mbobs) == 1 and len(mbobs[0]) == 1

        return new_obs

    def _print_result(self, data):
        # mess = "        wmom s2n: %g Trat: %g"
        # print(mess % (data['wmom_s2n'][0], data['wmom_T_ratio'][0]), flush=True)
        pass

    def _measure_moments(self, obs):
        """
        measure weighted moments
        """

        res = self.weight.get_weighted_moments(obs=obs, maxrad=1.e9)

        if res['flags'] != 0:
            return res

        res['numiter'] = 1
        res['g'] = res['e']
        res['g_cov'] = res['e_cov']

        return res

    def _get_dtype(self, model, npars):
        n = Namer(front=model)
        dt = [
            ('flags', 'i4'),

            ('psfrec_flags', 'i4'),  # psfrec is the original psf
            ('psfrec_g', 'f8', 2),
            ('psfrec_T', 'f8'),

            ('psf_g', 'f8', 2),
            ('psf_T', 'f8'),

            (n('flags'), 'i4'),
            (n('s2n'), 'f8'),
            (n('pars'), 'f8', npars),
            (n('g'), 'f8', 2),
            (n('g_cov'), 'f8', (2, 2)),
            (n('T'), 'f8'),
            (n('T_err'), 'f8'),
            (n('T_ratio'), 'f8'),
        ]

        return dt

    def _get_output(self, res, pres):

        npars = 6

        model = 'wmom'
        n = Namer(front=model)

        dt = self._get_dtype(model, npars)
        output = np.zeros(1, dtype=dt)

        output['psfrec_flags'] = NO_ATTEMPT

        output[n('flags')] = res['flags']

        flags = 0
        if pres['flags'] != 0:
            flags |= PSF_FAILURE

        if res['flags'] != 0:
            flags |= OBJ_FAILURE

        if pres['flags'] == 0:
            output['psf_g'] = pres['g']
            output['psf_T'] = pres['T']

        if res['flags'] == 0:
            output[n('s2n')] = res['s2n']
            output[n('pars')] = res['pars']
            output[n('g')] = res['g']
            output[n('g_cov')] = res['g_cov']
            output[n('T')] = res['T']
            output[n('T_err')] = res['T_err']

            if pres['flags'] == 0:
                output[n('T_ratio')] = res['T']/pres['T']

        output['flags'] = flags
        return output

    def _set_mompars(self):
        wpars = self.get('weight', {})

        T = ngmix.moments.fwhm_to_T(wpars.get('fwhm', 1.2))

        # the weight is always centered at 0, 0 or the
        # center of the coordinate system as defined
        # by the jacobian

        weight = ngmix.GMixModel(
            [0.0, 0.0, 0.0, 0.0, T, 1.0],
            'gauss',
        )

        # make the max of the weight 1.0 to get better
        # fluxes

        weight.set_norms()
        norm = weight.get_data()['norm'][0]
        weight.set_flux(1.0/norm)

        self.weight = weight


def make_ngmix_mbobslist(im, psf, wcs, cen1, cen2, noise, rng, nstamp=33):
    nstamp_cen = (nstamp - 1)/2

    psf_im = psf.drawImage(nx=nstamp, ny=nstamp, wcs=wcs)
    jac = DiagonalJacobian(scale=wcs.scale, x=nstamp_cen, y=nstamp_cen)
    psf_obs = Observation(
        image=psf_im.array,
        jacobian=jac,
        weight=np.ones_like(psf_im.array),
    )

    def _make_mbobs(im, cen):
        dx = cen[0] - int(cen[0] + 0.5)
        dy = cen[1] - int(cen[1] + 0.5)
        xlow = int(int(cen[0] + 0.5) - nstamp_cen)
        ylow = int(int(cen[1] + 0.5) - nstamp_cen)
        _im = im[ylow:ylow+nstamp, xlow:xlow+nstamp]
        jac = DiagonalJacobian(
            scale=wcs.scale,
            x=nstamp_cen+dx,
            y=nstamp_cen+dy,
        )

        obs = Observation(
            image=_im,
            jacobian=jac,
            psf=psf_obs,
            weight=np.ones_like(_im),
            noise=rng.normal(size=_im.shape) * noise
        )
        mbobs = MultiBandObsList()
        obsl = ObsList()
        obsl.append(obs)
        mbobs.append(obsl)
        return mbobs

    return [
        _make_mbobs(im, cen1),
        _make_mbobs(im, cen2),
    ]


def make_objs(
    *, rng, blend, shear,
    hlr=1.0, flux=1e4, psf_fhwm=0.9, wcs_scale=0.2, npix=251, noise=1,
):
    if blend:
        dx_pixels = 16
        dy_pixels = 0
    else:
        dx_pixels = 63
        dy_pixels = 63

    wcs = galsim.PixelScale(wcs_scale)
    psf = galsim.Gaussian(fwhm=psf_fhwm)
    dx = dx_pixels * wcs.scale
    dy = dy_pixels * wcs.scale

    dx1 = dx + rng.uniform(low=-0.5, high=0.5) * wcs.scale

    cen = (npix - 1)/2

    cen1 = (cen - dx1/wcs.scale, cen - dy_pixels)
    cen2 = (cen, cen)

    obj1 = galsim.Exponential(
        half_light_radius=hlr
    ).shear(g1=shear, g2=0).shift(dx=-dx1, dy=-dy).withFlux(flux)
    obj2 = galsim.Exponential(
        half_light_radius=hlr
    ).shift(dx=0, dy=0).withFlux(flux)

    im = galsim.Convolve(obj1 + obj2, psf).drawImage(wcs=wcs, nx=npix, ny=npix).array
    im += rng.normal(size=im.shape) * noise

    return im, psf, wcs, cen1, cen2, obj1, obj2, noise


def run_sim(seed, blend, shear):
    rng = np.random.RandomState(seed=seed)

    im, psf, wcs, cen1, cen2, obj1, obj2, noise = make_objs(
        rng=rng, blend=blend, shear=shear
    )

    obsl = make_ngmix_mbobslist(im, psf, wcs, cen1, cen2, noise, rng)

    omcal = make_mcal_obs(obsl, rng)

    fitter = Moments({}, rng)
    res = {}
    for k, v in omcal.items():
        res[k] = fitter.go(v)

    return res


def print_data(g1, g2, R11, R22, total, func=print):
    R = (R11 + R22)/total/2
    func(
        "m1|g2|R1|R2=%0.6g|%0.6g|%0.6g|%0.6g" % (
            (g1/R11/0.02-1)[0],
            (g1/R11)[1],
            R[0],
            R[1],
        )
    )


def _run_job(seed, blend, shear):
    res = run_sim(seed, blend, shear)
    R11 = (res['1p']['wmom_g'][:, 0] - res['1m']['wmom_g'][:, 0]) / 0.02
    R22 = (res['2p']['wmom_g'][:, 1] - res['2m']['wmom_g'][:, 1]) / 0.02
    g1 = res['noshear']['wmom_g'][:, 0]
    g2 = res['noshear']['wmom_g'][:, 1]
    return g1, g2, R11, R22


@click.command()
@click.option('--blend', is_flag=True)
@click.option('--max-workers', default=8, type=int)
@click.option('--shear', type=float, required=True)
@click.option('--n-samples', default=1000, type=int)
def main(blend, shear, max_workers, n_samples):
    rng = np.random.RandomState(seed=3453425342)

    res = run_sim(rng.randint(1, 2**32-1), blend, shear)
    R11 = (res['1p']['wmom_g'][:, 0] - res['1m']['wmom_g'][:, 0]) / 0.02
    R22 = (res['2p']['wmom_g'][:, 1] - res['2m']['wmom_g'][:, 1]) / 0.02
    g1 = res['noshear']['wmom_g'][:, 0]
    g2 = res['noshear']['wmom_g'][:, 1]

    total = n_samples

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futs = [
            exe.submit(_run_job, rng.randint(1, 2**32-1), blend, shear)
            for _ in tqdm.trange(total-1)
        ]

        for i, fut in tqdm.tqdm(enumerate(as_completed(futs)), total=total):
            res = fut.result()
            g1 += res[0]
            g2 += res[1]
            R11 += res[2]
            R22 += res[3]

            if i % (total/100) == 0 and i > 0:
                print_data(g1, g2, R11, R22, i, func=tqdm.tqdm.write)

    print_data(g1, g2, R11, R22, total)

    data = {
        'g1': [_g for _g in g1],
        'g2': [_g for _g in g2],
        'R11': [_g for _g in R11],
        'R22': [_g for _g in R22],
        'n': total,
    }

    with open(f"data_blend{blend}_shear{shear}.json", "w") as fp:
        json.dump(data, fp)


if __name__ == '__main__':
    main()
