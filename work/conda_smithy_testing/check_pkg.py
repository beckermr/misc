import sys

import requests

r = requests.get(
    "https://api.anaconda.org/package/conda-forge/cf-autotick-bot-test-package/files"
)

if r.status_code != 200:
    r.raise_for_status()

pkgs = r.json()

print(pkgs)

if sys.argv[1].startswith("v"):
    bn = str(int(sys.argv[1][1:]) + 14)
else:
    bn = sys.argv[1]

print("check build number: %s" % bn, flush=True)

build = "_" + bn
ver = "0.9"


not_found = set()
num_found = 0
for linux in ["circle", "azure", "travis"]:
    for osx in ["circle", "azure", "travis"]:
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

            if not found:
                not_found.add((subdir, linux, osx))

print("NOT found:", not_found)
print("num found:", num_found)

assert num_found == 36, num_found
