import subprocess
import time
import glob


try:
    for nproc in [1, 2, 4, 8]:
        print("timing %d processes..." % nproc, flush=True)
        subprocess.run(
            "rm -f multi_image_mod_*.fits",
            shell=True,
            capture_output=True,
        )

        t0 = time.time()
        subprocess.run(
            "galsim config.yml output.nproc=%d" % nproc,
            shell=True,
            capture_output=True,
            check=True,
        )
        t0 = time.time() - t0
        print(nproc, t0, flush=True)

        lfnames = len(glob.glob("multi_image_mod_*.fits"))
        assert lfnames == 16, lfnames
finally:
    subprocess.run(
        "rm -f multi_image_mod_*.fits",
        shell=True,
        capture_output=True,
    )
