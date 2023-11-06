import joblib
import os
import subprocess


def _download(fname):
    baseurl = (
        "https://s3df.slac.stanford.edu/data/rubin/sim-data/"
        "sims_featureScheduler_runs2.2"
    )
    if os.path.exists(os.path.expandvars(os.path.join(
        "${RUBIN_SIM_DATA_DIR}",
        "matts_extra_sims",
        fname,
    ))):
        switch = "-N"
    else:
        switch = ""

    subprocess.run(
        "cd ${RUBIN_SIM_DATA_DIR}/matts_extra_sims "
        "&& wget %s %s/%s" % (switch, baseurl, fname),
        shell=True,
        capture_output=True,
    )


def main():
    fnames = "clouds_baseline_cloudso0_ns2_v2.2_10yrs.db \
    clouds_baseline_cloudso0_ns3_v2.2_10yrs.db \
    clouds_baseline_cloudso10_ns2_v2.2_10yrs.db \
    clouds_baseline_cloudso10_ns3_v2.2_10yrs.db \
    clouds_baseline_cloudso11_ns2_v2.2_10yrs.db \
    clouds_baseline_cloudso11_ns3_v2.2_10yrs.db \
    clouds_baseline_cloudso1_ns2_v2.2_10yrs.db \
    clouds_baseline_cloudso1_ns3_v2.2_10yrs.db \
    clouds_baseline_cloudso20_ns2_v2.2_10yrs.db \
    clouds_baseline_cloudso20_ns3_v2.2_10yrs.db \
    clouds_baseline_cloudso21_ns2_v2.2_10yrs.db \
    clouds_baseline_cloudso21_ns3_v2.2_10yrs.db \
    clouds_baseline_cloudso30_ns2_v2.2_10yrs.db \
    clouds_baseline_cloudso30_ns3_v2.2_10yrs.db \
    clouds_baseline_cloudso31_ns2_v2.2_10yrs.db \
    clouds_baseline_cloudso31_ns3_v2.2_10yrs.db \
    clouds_baseline_cloudso35_ns2_v2.2_10yrs.db \
    clouds_baseline_cloudso35_ns3_v2.2_10yrs.db \
    clouds_baseline_cloudso36_ns2_v2.2_10yrs.db \
    clouds_baseline_cloudso36_ns3_v2.2_10yrs.db".split()
    fnames = [os.path.join("clouds_baseline", fn) for fn in fnames]
    fnames += [
        "rolling/noroll_v2.2_10yrs.db",
        "rolling/roll_with_const_roll_indx0_v2.2_10yrs.db",
        "rolling/roll_with_const_roll_indx1_v2.2_10yrs.db",
        "rolling/roll_with_const_roll_indx2_v2.2_10yrs.db",
        "rolling/roll_with_const_roll_indx3_v2.2_10yrs.db",
        "rolling/roll_with_const_roll_indx4_v2.2_10yrs.db",
        "rolling/roll_with_const_roll_indx5_v2.2_10yrs.db",
        "rolling/roll_with_const_roll_indx6_v2.2_10yrs.db",
        "rolling/roll_with_const_roll_indx7_v2.2_10yrs.db",
        "rolling/roll_with_const_roll_indx8_v2.2_10yrs.db",
    ]
    fnames += [
        "baseline/baseline_v2.2_10yrs.db",
    ]

    jobs = [
        joblib.delayed(_download)(fname)
        for fname in fnames
    ]
    with joblib.Parallel(n_jobs=4, backend="threading", verbose=100) as par:
        par(jobs)


if __name__ == "__main__":
    main()
