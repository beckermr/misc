import subprocess
import time


try:
    for nproc in [2, 4, 8]:
        t0 = time.time()
        subprocess.run(
            "galsim config.yml output.nproc=%d" % nproc,
            shell=True,
            capture_output=True,
        )
        t0 = time.time() - t0
        print(nproc, t0)
finally:
    pass
    # subprocess.run(
    #     "rm -f multi_image_mod_*.fits",
    #     shell=True,
    #     capture_output=True,
    # )
