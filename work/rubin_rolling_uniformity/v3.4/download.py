import joblib
import os
import subprocess


def _download(fname):
    baseurl = (
        "https://s3df.slac.stanford.edu/data/rubin/sim-data/"
        "sims_featureScheduler_runs3.4"
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
        "baseline/baseline_v3.4_10yrs.db",
        "roll_pause/roll_pause_mjdp0_v3.4_10yrs.db",
        "roll_pause/roll_pause_mjdp120_v3.4_10yrs.db",
        "roll_pause/roll_pause_mjdp180_v3.4_10yrs.db",
        "roll_pause/roll_pause_mjdp270_v3.4_10yrs.db",
        "roll_pause/roll_pause_mjdp365_v3.4_10yrs.db",
        "roll_pause/roll_pause_mjdp60_v3.4_10yrs.db",
        "noroll/noroll_mjdp0_v3.4_10yrs.db",
        "noroll/noroll_mjdp120_v3.4_10yrs.db",
        "noroll/noroll_mjdp180_v3.4_10yrs.db",
        "noroll/noroll_mjdp270_v3.4_10yrs.db",
        "noroll/noroll_mjdp365_v3.4_10yrs.db",
        "noroll/noroll_mjdp60_v3.4_10yrs.db",
        "noroll/noroll_v3.4_10yrs.db",
        "roll_pause2_mjdp0_v3.4_10yrs.db",
        "roll_pause2/roll_pause2_mjdp120_v3.4_10yrs.db",
        "roll_pause2/roll_pause2_mjdp180_v3.4_10yrs.db",
        "roll_pause2/roll_pause2_mjdp270_v3.4_10yrs.db",
        "roll_pause2/roll_pause2_mjdp365_v3.4_10yrs.db",
        "roll_pause2/roll_pause2_mjdp60_v3.4_10yrs.db",
        "roll_uniform/roll_uniform_mjdp0_v3.4_10yrs.db",
        "roll_uniform_mjdp180_v3.4_10yrs.db",
        "roll_uniform/roll_uniform_mjdp253_v3.4_10yrs.db",
        "roll_uniform/roll_uniform_mjdp67_v3.4_10yrs.db",
        "roll_uniform_early_half/roll_uniform_early_half_mjdp0_v3.4_10yrs.db",
        "roll_uniform_early_half/roll_uniform_early_half_mjdp180_v3.4_10yrs.db",
        "roll_uniform_early_half/roll_uniform_early_half_mjdp253_v3.4_10yrs.db",
        "roll_uniform_early_half/roll_uniform_early_half_mjdp67_v3.4_10yrs.db"
    ]

    fnames += [
        "https://s3df.slac.stanford.edu/data/rubin/sim-data/"
        "sims_featureScheduler_runs2.2/baseline/baseline_v2.2_10yrs.db",
        "https://s3df.slac.stanford.edu/data/rubin/sim-data/"
        "sims_featureScheduler_runs3.3/baseline/baseline_v3.3_10yrs.db",
    ]

    jobs = [
        joblib.delayed(_download)(fname)
        for fname in fnames
    ]
    with joblib.Parallel(n_jobs=4, backend="threading", verbose=100) as par:
        par(jobs)


if __name__ == "__main__":
    main()
