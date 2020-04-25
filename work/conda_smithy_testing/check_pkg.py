import sys

import requests

r = requests.get(
    "https://api.anaconda.org/package/conda-forge/cf-autotick-bot-test-package/files"
)

if r.status_code != 200:
    r.raise_for_status()

pkgs = r.json()

build = "_" + sys.argv[1]
ver = "0.9"

num_found = 0
for linux in ["circle", "travis", "azure"]:
    for osx in ["circle", "travis", "azure"]:
        for subdir in ["linux-64", "osx-64", "linux-ppc64le", "linux-aarch64"]:
            found = False
            cislug = "-%s%s_" % (linux, osx)
            for pkg in pkgs:
                if (
                    pkg["version"] == ver
                    and pkg["basename"].endswith(build + ".tar.bz2")
                    and cislug in pkg["basename"]
                    and pkg["basename"].startswith(subdir)
                ):
                    found = True
                    print("FOUND:", subdir, linux, osx)
                    num_found += 1

            assert found, (subdir, linux, osx)

assert num_found == 36
