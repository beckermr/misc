import os
import glob

import numpy as np
from rubin_sim import maf
import fitsio
import joblib


def _process_file(filename):
    nside = 128
    nbins = 40
    bins = np.linspace(0, 3653, nbins+1)

    runname = os.path.basename(filename).replace('.db', '')
    oname = runname + "_nside%d_bins%d.fits" % (nside, nbins)

    if not os.path.exists(oname):
        print("starting %s" % os.path.basename(filename), flush=True)

        metric = maf.RIZDetectionCoaddExposureTime(bins=bins)

        slicer = maf.HealpixSlicer(nside=nside, useCache=False)
        sql = 'note not like "%%DD%%" and night <= 3653 '
        bundle = maf.MetricBundle(metric, slicer, sql, runName=runname)
        bg = maf.MetricBundleGroup([bundle], filename, '.', None)
        bg.runAll()
        maps_arr = bundle.metricValues.copy().T

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
    jobs = [
        joblib.delayed(_process_file)(fname)
        for fname in fnames
    ]

    print("running %d jobs" % len(jobs), flush=True)
    with joblib.Parallel(n_jobs=2, verbose=100, backend="loky") as par:
        par(jobs)


if __name__ == "__main__":
    main()