import joblib
import os
import subprocess


def _download(fname):
    baseurl = (
        "https://s3df.slac.stanford.edu/data/rubin/sim-data/"
        "sims_featureScheduler_runs3.6"
    )
    if os.path.exists(os.path.expandvars(os.path.join(
        "${RUBIN_SIM_DATA_DIR}",
        "matts_extra_sims",
        os.path.basename(fname),
    ))):
        switch = "-c"
    else:
        switch = ""

    if fname.startswith("http"):
        baseurl = ""

    if baseurl:
        baseurl += "/"

    subprocess.run(
        "cd ${RUBIN_SIM_DATA_DIR}/matts_extra_sims "
        "&& wget %s %s%s" % (switch, baseurl, fname),
        shell=True,
        capture_output=False,
    )


def main():
    fnames = [
        "baseline/baseline_v3.6_10yrs.db",
        "four_cycle/four_cycle_v3.6_10yrs.db",
    ]

    jobs = [
        joblib.delayed(_download)(fname)
        for fname in fnames
    ]
    with joblib.Parallel(n_jobs=4, backend="threading", verbose=100) as par:
        par(jobs)


if __name__ == "__main__":
    main()
