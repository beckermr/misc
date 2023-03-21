import subprocess
import time
import glob


try:
    subprocess.run(
        "rm -f multi_image_mod_*.fits",
        shell=True,
        capture_output=True,
    )

    for nproc in [2, 4, 8]:
        t0 = time.time()
        subprocess.run(
            "galsim config.yml output.nproc=%d" % nproc,
            shell=True,
            capture_output=True,
        )
        t0 = time.time() - t0
        print(nproc, t0)

    assert len(glob.glob("multi_image_mod_*.fits")) == 16
finally:
    subprocess.run(
        "rm -f multi_image_mod_*.fits",
        shell=True,
        capture_output=True,
    )
