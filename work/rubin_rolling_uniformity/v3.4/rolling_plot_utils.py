import glob

import numpy as np
import imageio

from astropy.time import Time
from astropy.coordinates import get_sun
from astropy.coordinates import EarthLocation

RUBIN_LOC = EarthLocation.of_site("Rubin", refresh_cache=True)

MJD_2024 = Time(
    "2024-01-01T00:00:00.00",
    format="isot",
    scale="utc",
    location=RUBIN_LOC,
).utc.mjd

YEAR = 365.25
BASELINE_SURVEY_START_MJD = 60796.0  # 60796 = May 10, 2025


def get_sun_ra_at_mjd(mjd):
    t = Time(mjd, format='mjd', location=RUBIN_LOC)
    return get_sun(t).ra.deg


def get_phase_for_ra_in_mjd(ra, start_time, phase_shift_quarter_years=1):
    sun_ra_start = get_sun_ra_at_mjd(start_time)
    ps = phase_shift_quarter_years * np.pi / 2.0
    return (
        ((sun_ra_start - ra) / 180 * np.pi + ps) % (2.0 * np.pi)
        * (YEAR / 2.0 / np.pi)
    )


def get_season(mjd, ra, start_time, phase_shift_quarter_years=1):
    phase = get_phase_for_ra_in_mjd(ra, start_time, phase_shift_quarter_years=phase_shift_quarter_years)
    return (mjd - start_time + phase) / YEAR
    # phase = get_phase_for_ra_in_mjd(ra, start_time, phase_shift_quarter_years=1)
    # _season = (mjd - start_time + phase) / YEAR  - 1
    # if maf_season:
    #     return _season
    # else:
    #     return _season + 0.25 * 360 / YEAR


def get_cuml_desired_obs(mjd_in, ra, start_time, rise, verbose=False):
    """
    the rise parameter is the pattern of desired rates of accumulating
    observations, normalized so that a uniform survey is 1

    for uniform it is an array of 1s for as many seasons (NOT years) the
    survey runs
    for rolling it is an array of that reflects the increased cadence in
    rolling seasons
    for example:

        [1, 1, 1, U, D, U, D, U, D, 1, 1, 1]

    where U,D must obey the constraint that U+D = 2 and
    typical values are U = 1.9 and D = 0.1
    """
    steps = np.array(rise)
    phase = get_phase_for_ra_in_mjd(ra, start_time)
    t = mjd_in - start_time + phase
    if verbose:
        print("t/period:", t / YEAR)
    season = np.floor(t / YEAR)
    season = season.astype(int)
    if verbose:
        print("season:", season)

    plateus = np.cumsum(steps) - steps[0]

    result = plateus[season]
    if verbose:
        print("plateaus:", result)
    tphased = t % YEAR
    step_inds = np.where(tphased > YEAR / 2.0)[0]
    if verbose:
        print("step_inds", step_inds)
    result[step_inds] += (
        (tphased[step_inds] - YEAR / 2) * steps[season + 1][step_inds]
        / (0.5 * YEAR)
    )
    result[np.where(t < 0)] = 0
    if verbose:
        print("result:", result)
    return result


def plot_years_and_seasons(ax, start_time, n_years):
    sun_ra_start = get_sun_ra_at_mjd(start_time)

    phase_0 = get_phase_for_ra_in_mjd(0, start_time)
    phase_360 = get_phase_for_ra_in_mjd(360, start_time)

    season_left = YEAR - phase_0
    season_right = 2 * YEAR - phase_360

    i = -1
    ax.plot(
        [0, 360],
        [(YEAR * i + season_left + start_time - MJD_2024) / YEAR + 2024,
         (YEAR * i + season_right + start_time - MJD_2024) / YEAR + 2024],
        color="gray",
        linestyle="dashed",
    )

    for i in range(n_years):
        yloc = (start_time - MJD_2024) / YEAR + i + 2024
        ax.axhline(yloc, color="gray", linestyle="dotted")
        ax.plot(
            [0, 360],
            [(YEAR * i + season_left + start_time - MJD_2024) / YEAR + 2024,
             (YEAR * i + season_right + start_time - MJD_2024) / YEAR + 2024],
            color="gray",
            linestyle="dashed",
        )

        ax.text(5.0, yloc + 0.9, "Y%d" % (i+1), ha="left", va="top")
    ax.axhline(
        (start_time + n_years * YEAR - MJD_2024) / YEAR + 2024,
        color="gray",
        linestyle="dashed",
    )
    ax.axvline(sun_ra_start + 90, color="gray", linestyle="dashed")


def make_default_healpix_footprints(nside=32):
    # copied out of https://github.com/lsst-sims/sims_featureScheduler_runs3.4
    # /blob/main/roll_pause/roll_pause.py#L1540C5-L1562C50
    from rubin_scheduler.scheduler.utils import EuclidOverlapFootprint

    # Modify the footprint
    sky = EuclidOverlapFootprint(nside=nside, smc_radius=4, lmc_radius=6)
    footprints_hp_array, labels = sky.return_maps()

    wfd_indx = np.where(
        (labels == "lowdust") | (labels == "LMC_SMC") | (labels == "virgo")
    )[0]
    wfd_footprint = footprints_hp_array["r"] * 0
    wfd_footprint[wfd_indx] = 1

    footprints_hp = {}
    for key in footprints_hp_array.dtype.names:
        footprints_hp[key] = footprints_hp_array[key]

    footprint_mask = footprints_hp["r"] * 0
    footprint_mask[np.where(footprints_hp["r"] > 0)] = 1

    return footprints_hp, wfd_indx


def make_gif(
    fbase, ext="png", pause_fparts=None, num_pause_frames=5, **kwargs
):
    filenames = sorted(glob.glob(
        f"{fbase}/{fbase}" + "_[0-9][0-9][0-9][0-9]." + ext)
    )
    images = []
    for filename in filenames:
        img = imageio.imread(filename)

        if (
            pause_fparts is not None
            and any(filename.endswith(pfp) for pfp in pause_fparts)
        ):
            print("found file to pause at:", filename)
            for _ in range(num_pause_frames):
                images.append(img)
        else:
            images.append(img)
    imageio.mimsave(fbase + ".gif", images, **kwargs)


def make_footprint_gif(
    fbase,
    footprint,
    wfd_idx,
    start_time,
    n_years,
    images_per_year=10,
):
    import joblib
    import os

    def _render(fbase, footprint, mjd, dtime, wfd_idx):
        # import proplot as pplt  # noqa - sets styles
        import matplotlib.pyplot as plt

        d = footprint(mjd, norm=False)

        fig, axs = plt.subplots(figsize=(10, 6))

        rmap = d["r"] * 0
        rmap[wfd_idx] = d["r"][wfd_idx]

        if False:
            import skyproj
            sp = skyproj.McBrydeSkyproj(ax=axs)
            sp.draw_hpxmap(rmap, vmin=0, vmax=10, cmap="rocket_r")
            sp.draw_colorbar(
                location="bottom", pad=0.15,
                label="requested cumulative visits normalized to 10 years",
            )
            sp.ax.text(
                0.1,
                0.9,
                "Year %0.2f" % dtime,
                ha="left",
                va="top",
                transform=sp.ax.transAxes,
                fontsize=16,
            )
        else:
            import healpy as hp

            if np.mean(rmap[wfd_idx]) != 0:
                scat = "%2d" % int(
                    np.std(rmap[wfd_idx])
                    / np.mean(rmap[wfd_idx])
                    * 100
                )
            else:
                scat = "-"

            plt.sca(axs)
            hp.mollview(
                rmap, min=0, max=10, cmap="Reds", hold=True, title="",
                unit="cumul. visits relative to one year",
            )
            hp.visufunc.graticule(dpar=30, dmer=60, color="gray", alpha=0.5)
            plt.gca().text(
                0,
                1,
                "Y%5.2f - %s" % (
                    dtime, fbase
                ),
                transform=axs.transAxes,
                ha="left",
                va="top",
                fontsize=18,
            )
            plt.gca().text(
                1,
                1,
                "RMS: %s%%" % scat,
                transform=axs.transAxes,
                ha="right",
                va="top",
                fontsize=18,
            )

        fig.savefig(
            f"{fbase}/{fbase}" + "_%04d.jpeg" % int(dtime * 100),
            dpi=100,
        )
        plt.close(fig)

    # remove old images
    os.makedirs(fbase, exist_ok=True)
    os.system(f"rm {fbase}/*")
    os.system(f"rm {fbase}.gif")

    jobs = []
    for dtime in np.linspace(0, n_years, n_years * images_per_year + 1):
        mjd = start_time + dtime * YEAR
        jobs.append(
            joblib.delayed(_render)(fbase, footprint, mjd, dtime, wfd_idx)
        )

    with joblib.Parallel(n_jobs=8, backend="loky", verbose=50) as parallel:
        parallel(jobs)

    make_gif(
        fbase,
        ext="jpeg",
        fps=3 / 10 * images_per_year,
        pause_fparts=[
            "_%04d.jpeg" % int(year * 100)
            for year in [1, 4, 7, 10]
            if year <= n_years
        ],
        num_pause_frames=int(np.ceil(
            2.0 * (3/10 * images_per_year)
        )),
    )
