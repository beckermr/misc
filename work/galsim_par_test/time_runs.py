import subprocess
import time
import glob


try:
    for nproc in [2, 4, 8]:
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
        )
        t0 = time.time() - t0
        print(nproc, t0)

        # fnames = glob.glob("multi_image_mod_*.fits")
        # assert fnames == 16, fnames
finally:
    pass
    # subprocess.run(
    #     "rm -f multi_image_mod_*.fits",
    #     shell=True,
    #     capture_output=True,
    # )
