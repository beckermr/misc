"""
This is a copy of parts of

    https://github.com/lsst/rubin_scheduler/blob/main/rubin_scheduler/scheduler/utils/footprints.py

at commit afada5e with edits to make it do what we want for new rolling ideas.


Footprints: Take sky area maps and turn them into dynamic `footprint`
objects which understand seasons and time, in order to weight area on sky
appropriately for a given time.
"""

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from rubin_scheduler.utils import _hpid2_ra_dec

from rubin_scheduler.scheduler.utils import (
    set_default_nside,
    Footprints,
    Footprint,
    StepSlopes,
)


def make_rolling_footprints(
    fp_hp=None,
    mjd_start=60218.0,
    sun_ra_start=3.27717639,
    nslice=2,
    scale=0.8,
    nside=32,
    wfd_indx=None,
    order_roll=0,
    n_cycles=None,
    n_constant_start=2,
    n_constant_end=6,
):
    """
    Generate rolling footprints

    Parameters
    ----------
    fp_hp : dict-like
        A dict with filtername keys and HEALpix map values
    mjd_start : `float`
        The starting date of the survey.
    sun_ra_start : `float`
        The RA of the sun at the start of the survey
    nslice : `int`
        How much to slice the sky up. Can be 2, 3, 4, or 6.
    scale : `float`
        The strength of the rolling, value of 1 is full power rolling.
        Zero is no rolling.
    wfd_indx : array of ints
        The indices of the HEALpix map that are to be included in the rolling.
    order_roll : `int`
        Change the order of when bands roll. Default 0.
    n_cycles : `int`
        Number of complete rolling cycles to attempt. If None, defaults to 3
        full cycles for nslice=2, 2 cycles for nslice=3 or 4, and 1 cycle for
        nslice=6.
    n_constant_start : `int`
        The number of constant non-rolling seasons to start with. Anything
        less than 2 will start rolling too early near Y1. Defaults to 2.
    n_constant_end : `int`
        The number of constant seasons to end the survey with. Defaults to 6.

    Returns
    -------
    Footprints object
    """

    nc_default = {2: 3, 3: 2, 4: 2, 6: 1}
    if n_cycles is None:
        n_cycles = nc_default[nslice]

    hp_footprints = fp_hp

    D = 1.0 - scale
    U = nslice - D * (nslice - 1)

    start = [1.0] * n_constant_start
    # After n_cycles, just go to no-rolling for 6 years.
    end = [1.0] * n_constant_end

    rolling = [U] + [D] * (nslice - 1)
    rolling = np.roll(rolling, order_roll).tolist()

    all_slopes = []
    for i in range(nslice):
        _roll = np.roll(rolling, i).tolist() + [1]
        all_slopes.append(
            start + _roll * n_cycles + end
        )
    for i in range(nslice):
        _roll = np.roll(rolling, i).tolist() + [1]
        _roll = [_roll[-1]] + _roll[1:-1] + [_roll[0]]
        all_slopes.append(
            start + _roll * n_cycles + end
        )
    dvals = {
        1: "1",
        D: "D",
        U: "U",
    }
    abc = ["a", "b", "c", "d", "e", "f", "g", "h"]
    slice_names = ["slice %s" % abc[i] for i in range(nslice)]
    for i, s in enumerate(all_slopes):
        if i >= nslice:
            sname = (
                slice_names[i-nslice]
                + " w/ ra - sun_ra in [90, 270]"
            )
        else:
            sname = (
                slice_names[i]
                + " w/ ra - sun_ra in [270, 90]"
            )
        print(sname + ": " + " ".join([dvals[x] for x in s]))

    # if uniform:
    #     # v1
    #     # all_slopes = [
    #     #     start + [U, D, U, D, 1, D, U, 1] + end[:-2],
    #     #     start + [D, U, D, U, 1, U, D, 1] + end[:-2],
    #     #     start + [U, D, 1, D, U, 1, U, D] + end[:-2],
    #     #     start + [D, U, 1, U, D, 1, D, U] + end[:-2],
    #     # ]
    #     # v2
    #     start = [1, 1]
    #     DU1 = [D, U, 1]
    #     UD1 = [U, D, 1]
    #     rDU1 = np.roll(DU1, 1).tolist()
    #     rUD1 = np.roll(UD1, 1).tolist()
    #     all_slopes = [
    #         # second half of season (sun ra +270 to sun ra +90)
    #         [1, 1,  D, U, 1,  D, U, 1,  D, U, 1] + end[:-2],
    #         [1, 1,  U, D, 1,  U, D, 1,  U, D, 1] + end[:-2],
    #         # first half of season (sun ra +90 to sun ra +270)
    #         [1, 1,  1, U, D,  1, U, D,  1, U, D] + end[:-2],
    #         [1, 1,  1, D, U,  1, D, U,  1, D, U] + end[:-2],
    #     ]
    # else:
    #     all_slopes = [
    #         start + rolling + end,
    #         start + np.roll(rolling, 1).tolist() + end,
    #     ]

    fp_non_wfd = Footprint(mjd_start, sun_ra_start=sun_ra_start, nside=nside)
    rolling_footprints = []
    for i in range(len(all_slopes)):
        step_func = StepSlopes(rise=all_slopes[i])
        rolling_footprints.append(
            Footprint(
                mjd_start,
                sun_ra_start=sun_ra_start,
                step_func=step_func,
                nside=nside,
            )
        )

    wfd = hp_footprints["r"] * 0
    if wfd_indx is None:
        wfd_indx = np.where(hp_footprints["r"] == 1)[0]

    wfd[wfd_indx] = 1
    non_wfd_indx = np.where(wfd == 0)[0]

    split_wfd_indices = slice_quad_galactic_cut(
        hp_footprints, nslice=nslice, wfd_indx=wfd_indx,
        ra_range=(sun_ra_start + 1.5 * np.pi, sun_ra_start + np.pi/2),
    )

    split_wfd_indices_delayed = slice_quad_galactic_cut(
        hp_footprints, nslice=nslice, wfd_indx=wfd_indx,
        ra_range=(sun_ra_start + np.pi / 2, sun_ra_start + 1.5 * np.pi),
    )

    for key in hp_footprints:
        temp = hp_footprints[key] + 0
        temp[wfd_indx] = 0
        fp_non_wfd.set_footprint(key, temp)

        for i in range(nslice):
            # make a copy of the current filter
            temp = hp_footprints[key] + 0
            # Set the non-rolling area to zero
            temp[non_wfd_indx] = 0

            indx = split_wfd_indices[i]
            # invert the indices
            ze = temp * 0
            ze[indx] = 1
            temp = temp * ze
            rolling_footprints[i].set_footprint(key, temp)

        for _i in range(nslice, nslice*2):
            # make a copy of the current filter
            temp = hp_footprints[key] + 0
            # Set the non-rolling area to zero
            temp[non_wfd_indx] = 0

            indx = split_wfd_indices_delayed[_i-nslice]
            # invert the indices
            ze = temp * 0
            ze[indx] = 1
            temp = temp * ze
            rolling_footprints[_i].set_footprint(key, temp)

    result = Footprints([fp_non_wfd] + rolling_footprints)
    return result


def _is_in_ra_range(ra, low, high):
    _low = low % (2.0 * np.pi)
    _high = high % (2.0 * np.pi)
    if _low <= _high:
        return (ra >= _low) & (ra <= _high)
    else:
        return (ra >= _low) | (ra <= _high)


def slice_quad_galactic_cut(
    target_map, nslice=2, wfd_indx=None, ra_range=None
):
    """
    Helper function for generating rolling footprints

    Parameters
    ----------
    target_map : dict of HEALpix maps
        The final desired footprint as HEALpix maps. Keys are filter names
    nslice : `int`
        The number of slices to make, can be 2 or 3.
    wfd_indx : array of ints
        The indices of target_map that should be used for rolling.
        If None, assumes the rolling area should be where target_map['r'] == 1.
    ra_range : tuple of floats, optional
        If not None, then the indices are restricted to the given RA range
        in radians.
    """

    ra, dec = ra_dec_hp_map(nside=hp.npix2nside(target_map["r"].size))

    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
    _, gal_lat = coord.galactic.l.deg, coord.galactic.b.deg

    indx_north = np.intersect1d(np.where(gal_lat >= 0)[0], wfd_indx)
    indx_south = np.intersect1d(np.where(gal_lat < 0)[0], wfd_indx)

    splits_north = slice_wfd_area_quad(
        target_map, nslice=nslice, wfd_indx=indx_north)
    splits_south = slice_wfd_area_quad(
        target_map, nslice=nslice, wfd_indx=indx_south)

    slice_indx = []
    for j in np.arange(nslice):
        indx_temp = []
        for i in np.arange(j + 1, nslice * 2 + 1, nslice):
            indx_temp += indx_north[
                splits_north[i - 1]: splits_north[i]
            ].tolist()
            indx_temp += indx_south[
                splits_south[i - 1]: splits_south[i]
            ].tolist()
        slice_indx.append(indx_temp)

    if ra_range is not None:
        ra_indx = np.where(_is_in_ra_range(ra, *ra_range))[0]
        for j in range(nslice):
            slice_indx[j] = np.intersect1d(ra_indx, slice_indx[j])

    return slice_indx


def slice_wfd_area_quad(target_map, nslice=2, wfd_indx=None):
    """
    Divide a healpix map in an intelligent way

    Parameters
    ----------
    target_map : dict of HEALpix arrays
        The input map to slice
    nslice : int
        The number of slices to divide the sky into (gets doubled).
    wfd_indx : array of int
        The indices of the healpix map to consider as part of the WFD area
        that will be split.
        If set to None, the pixels where target_map['r'] == 1 are
        considered as WFD.
    """
    nslice2 = nslice * 2

    wfd = target_map["r"] * 0
    if wfd_indx is None:
        wfd_indices = np.where(target_map["r"] == 1)[0]
    else:
        wfd_indices = wfd_indx
    wfd[wfd_indices] = 1
    wfd_accum = np.cumsum(wfd)
    split_wfd_indices = np.floor(
        np.max(wfd_accum) / nslice2 * (np.arange(nslice2) + 1)
    ).astype(int)
    split_wfd_indices = split_wfd_indices.tolist()
    split_wfd_indices = [0] + split_wfd_indices

    return split_wfd_indices


def ra_dec_hp_map(nside=None):
    """
    Return all the RA,dec points for the centers of a healpix map, in radians.
    """
    if nside is None:
        nside = set_default_nside()
    ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
    return ra, dec
