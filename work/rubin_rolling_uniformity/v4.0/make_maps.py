import os
import glob

import numpy as np
from rubin_sim import maf
import fitsio
import joblib


class RIZDetectionCoaddExposureTime(maf.metrics.vector_metrics.VectorMetric):
    """A metric computing the total exposure time of an riz coadd.

    This metric is intended to be used as a proxy for depth fluctuations in
    catalogs detected from coadds of the r, i and z bands together. This
    coadding + detection scheme is used by metadetection (weak lensing
    shear estimator) and will likely be adopted by the Rubin science
    pipelines.

    It counts the total exposure time in all three bands, excluding dusty
    regions, exposures that are too short, or areas where not all bands
    ugrizY are present. We do not make a depth cut explicitly since that is
    circular (and thus confuses MRB's feeble mind :/).

    TODO maybe:
     - apply some sort of inverse variance weighting to the coadd based on sky
       level?
     - use some sort of effective exposure time that accounts for the PSF?
     - @rhiannonlynne nicely suggested this Teff computation:
       rubin_sim/rubin_sim/maf/metrics/technicalMetrics.py

    However, given the unknown nature of detection will look like in LSST,
    a simple sum of exposure time is probably ok.

    Parameters
    ----------
    bins : list of float
        The bin edges. Typically this will be a list of nights for which to
        compute the riz coadd exposure times.
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
    det_bands : list of str, optional
        If not None, the bands to use for detection. If None, defaults to riz.
    min_bands : list of str, optional
        If not None, the bands whose presence is used to cut the survey data.
        If None, defaults to ugrizY.
    """

    def __init__(
        self,
        *,
        bins,
        bin_col="night",
        exp_time_col="visitExposureTime",
        filter_col="filter",
        ebvlim=0.2,
        min_expTime=15,
        det_bands=None,
        min_bands=None,
        **kwargs
    ):
        # Set up the coadd metric (using ExgalM5 adds galactic dust extinction)
        self.filter_col = filter_col
        self.exp_time_col = exp_time_col
        self.ebvlim = ebvlim
        self.min_exp_time = min_expTime
        self.det_bands = det_bands or ["r", "i", "z"]
        self.min_bands = set(min_bands or ["u", "g", "r", "i", "z", "y"])
        super().__init__(
            bins=bins,
            bin_col=bin_col,
            col=[self.exp_time_col, self.filter_col],
            metric_name="riz_detcoadd_exptime",
            units="seconds",
            maps=["DustMap"],
            **kwargs
        )

    def run(self, data_slice, slice_point):
        res = np.zeros(self.shape, dtype=self.metric_dtype)

        # If the sky is too dusty here, stop.
        if slice_point["ebv"] > self.ebvlim:
            res[:] = self.badval
            return res

        data_slice.sort(order=self.bin_col)
        cutinds = np.searchsorted(
            data_slice[self.bin_col], self.bins[1:],
            side="right"
        )
        maxcutind = data_slice.shape[0]
        cutinds = np.clip(cutinds, 0, maxcutind)

        # find all entries where exposure time is long enough and
        # in the detection bands
        exptime_msk = data_slice[self.exp_time_col] > self.min_exp_time
        filter_msk = np.in1d(data_slice[self.filter_col], self.det_bands)
        tot_msk = exptime_msk & filter_msk

        for i, cutind in enumerate(cutinds):
            if cutind == 0:
                res[i] = self.badval
                continue

            # check to make sure there is at least some
            # coverage in the required bands
            filters = set(data_slice[self.filter_col][:cutind])
            if filters != self.min_bands:
                res[i] = self.badval
                continue

            # if nothing passes for detection, we exclude this region
            if not np.any(tot_msk[:cutind]):
                res[i] = self.badval
                continue

            res[i] = np.sum(
                data_slice[self.exp_time_col][:cutind][tot_msk[:cutind]]
            )

        return res


def _process_file(filename, force_remake=False):
    nside = 128
    nbins = 40
    bins = np.linspace(0, 3653, nbins+1)

    runname = os.path.basename(filename).replace('.db', '')
    oname = runname + "_nside%d_bins%d.fits" % (nside, nbins)

    if (not os.path.exists(oname)) or force_remake:
        print("starting %s" % os.path.basename(filename), flush=True)

        metric = RIZDetectionCoaddExposureTime(bins=bins)

        slicer = maf.HealpixSlicer(nside=nside, use_cache=False)
        col = (
            "note"
            if any(old_ver in filename for old_ver in ["v2", "v3.0", "v3.1", "v3.2", "v3.3", "v3.4"])
            else "scheduler_note"
        )
        sql = (
            col + ' not like "%%DD%%" and '
            + col + ' not like "%%twilight%%" and '
            + col + ' not like "ToO%%" and '
            + 'night <= 3653'
        )
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
    force_remake = True

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
        "baseline_v3.4_10yrs.db",
        "roll_uniform_mjdp0_v3.4_10yrs.db",
        "noroll_v3.6_10yrs.db",
        "baseline_v4.0_10yrs.db",
        "four_cycle_v4.0_10yrs.db",
    ]
    fnames = [
        fname for fname in fnames
        if any(fname.endswith(sim) for sim in sims)
    ]
    jobs = [
        joblib.delayed(_process_file)(fname, force_remake=force_remake)
        for fname in fnames
    ]

    print("running %d jobs" % len(jobs), flush=True)
    with joblib.Parallel(n_jobs=4, verbose=100, backend="loky") as par:
        par(jobs)


if __name__ == "__main__":
    main()
