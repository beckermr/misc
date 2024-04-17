import os
import glob

import numpy as np
from rubin_sim import maf
import fitsio
import joblib


from rolling_plot_utils import get_season, BASELINE_SURVEY_START_MJD

class BandSeasonTotVisits(maf.metrics.vector_metrics.VectorMetric):
    """A metric computing the total number of visits in a band for a given season.

    Parameters
    ----------
    bin_col : str, optional
        The column to bin on. The default is 'night'.
    exp_time_col : str, optional
        The column name for the exposure time.
    filter_col : str, optional
        The column name for the filter name.
    ebvlim : float, optional
        The upper limit on E(B-V). Regions with E(B-V) greater than this
        limit are excluded.
    min_exp_time : float, optional
        The minimal exposure time for a visit to contribute to a coadd.
    band : str, optional
        Defaults to "i"
    phase_shift_quarter_years : int, optional
        If True, the phase shift in quarter years.
    use_maf_season_calc : bool, optional
        If True, use the MAF season calculation. If False, use the
        rolling_plot_utils.get_season function.
    """

    def __init__(
        self,
        *,
        exp_time_col="visitExposureTime",
        filter_col="filter",
        ebvlim=0.2,
        min_expTime=15,
        band="i",
        phase_shift_quarter_years=0,
        use_maf_season_calc=False,
    ):
        self.filter_col = filter_col
        self.exp_time_col = exp_time_col
        self.ebvlim = ebvlim
        self.min_exp_time = min_expTime
        self.mjd_col = "observationStartMJD"
        self.band = band
        self.seasons = np.arange(-2, 20) - 0.5
        self.season_vals = ((self.seasons[1:] + self.seasons[:-1]) / 2.0).astype(int)
        self.phase_shift_quarter_years = phase_shift_quarter_years
        self.use_maf_season_calc = use_maf_season_calc
        super().__init__(
            bins=self.seasons,
            bin_col=self.mjd_col,
            col=[self.exp_time_col, self.filter_col, self.mjd_col],
            metric_name=f"visits_per_season_band{self.band}",
            units="#",
            maps=["DustMap"],
        )

    def run(self, data_slice, slice_point):
        res = np.zeros(self.shape, dtype=self.metric_dtype)

        # If the sky is too dusty here, stop.
        if slice_point["ebv"] > self.ebvlim:
            res[:] = self.badval
            return res

        if self.use_maf_season_calc:
            from season_utils import calc_season

            slice_season = calc_season(
                data_slice["fieldRA"],
                data_slice[self.mjd_col],
                mjd_start=BASELINE_SURVEY_START_MJD,
                calc_diagonal=True,
            )
        else:
            slice_season = get_season(
                data_slice[self.mjd_col],
                data_slice["fieldRA"],
                BASELINE_SURVEY_START_MJD,
                phase_shift_quarter_years=self.phase_shift_quarter_years,
            )

        # find all entries where exposure time is long enough and
        # in the detection bands
        exptime_msk = data_slice[self.exp_time_col] > self.min_exp_time
        filter_msk = np.in1d(data_slice[self.filter_col], [self.band])
        tot_msk = exptime_msk & filter_msk

        for i, season in enumerate(self.season_vals):
            season_msk = np.floor(slice_season) == season
            _msk = tot_msk & season_msk
            if np.any(_msk):
                res[i] = np.sum(_msk)
            else:
                res[i] = self.badval

        return res


def _process_file(filename, psqy, use_maf_season_calc):
    nside = 128
    band = "i"
    runname = os.path.basename(filename).replace('.db', '')
    oname = runname + (
        f"_seasons_psqy{psqy}_usemaf{use_maf_season_calc}"
        f"_band{band}_nside{nside}.fits"
    )

    if not os.path.exists(oname):
        print(
            "starting %s w/ ps. quarter years = %s use maf = %s" % (
                os.path.basename(filename),
                psqy,
                use_maf_season_calc,
            ),
            flush=True,
        )

        metric = BandSeasonTotVisits(
            band=band,
            phase_shift_quarter_years=psqy,
            use_maf_season_calc=use_maf_season_calc,
        )

        slicer = maf.HealpixSlicer(nside=nside, use_cache=False)
        sql = 'note not like "%%DD%%" and night <= 3653 '
        bundle = maf.MetricBundle(metric, slicer, sql, run_name=runname)
        bg = maf.MetricBundleGroup([bundle], filename, '.', None)
        bg.run_all()
        maps_arr = bundle.metric_values.copy().T

        maps_arr = maps_arr.filled(fill_value=-9999)

        fitsio.write(
            oname,
            maps_arr,
            clobber=True,
        )


def main():
    fnames = glob.glob(
        os.path.expandvars(
            os.path.join(
                "${RUBIN_SIM_DATA_DIR}",
                'matts_extra_sims',
                '*.db',
            )
        )
    )
    sims = [
        "roll_uniform_early_half_mjdp0_v3.4_10yrs.db",
        "baseline_v3.4_10yrs.db",
        "noroll_mjdp0_v3.4_10yrs.db",
    ]
    fnames = [
        fname for fname in fnames
        if any(fname.endswith(sim) for sim in sims)
    ]
    jobs = []
    for fname in fnames:
        for psqy in [-1, 0, 1, 2]:
            jobs.append(
                joblib.delayed(_process_file)(fname, psqy, False)
            )

        jobs.append(
            joblib.delayed(_process_file)(fname, 0, True)
        )

    print("running %d jobs" % len(jobs), flush=True)
    with joblib.Parallel(n_jobs=4, verbose=100, backend="loky") as par:
        par(jobs)


if __name__ == "__main__":
    main()
