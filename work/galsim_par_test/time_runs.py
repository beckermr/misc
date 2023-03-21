import subprocess
import time
import glob
import sys

if len(sys.argv) > 1:
    nfiles = int(sys.argv[1])
else:
    nfiles = 128

try:
    for nproc in [128, 128, 64, 32, 16, 8, 4, 2]:
        print("timing %d processes..." % nproc, flush=True)
        subprocess.run(
            "rm -f multi_image_mod_*.fits",
            shell=True,
            capture_output=True,
        )

        t0 = time.time()
        subprocess.run(
            "galsim config.yml "
            "output.nproc=%d output.nfiles=%d" % (nproc, nfiles),
            shell=True,
            capture_output=True,
            check=True,
        )
        t0 = time.time() - t0
        print(nproc, t0, flush=True)

        lfnames = len(glob.glob("multi_image_mod_*.fits"))
        assert lfnames == nfiles, lfnames
finally:
    subprocess.run(
        "rm -f multi_image_mod_*.fits",
        shell=True,
        capture_output=True,
    )
