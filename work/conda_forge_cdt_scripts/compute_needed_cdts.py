#!/usr/bin/env python
import os
import subprocess

FEEDSTOCKS = "/Users/beckermr/Desktop/feedstocks"
CDT_SLUGS = [
    "-cos6-x86_64",
    "-cos7-x86_64",
    "-cos7-aarch64",
    "-cos7-ppc64le",
]


def get_curr_cdts():
    ccdt = subprocess.run(
        'compgen -G "%s/feedstocks/*"' % FEEDSTOCKS,
        shell=True,
        check=True,
        capture_output=True
    )
    cdts = set()
    for line in ccdt.stdout.decode("utf-8").splitlines():
        # this is a full path so get basename
        line = os.path.basename(line.strip())
        # if it ends with any of the cdt slugs, then keep it
        if any(line.endswith(c) for c in CDT_SLUGS):
            cdts.add(line)

    return cdts


def compute_all_cdts():
    ccdt = subprocess.run(
        'compgen -G "%s/feedstocks/*/recipe/meta.yaml" | xargs cat | '
        'grep -E "cdt\\(" | awk \'{$1=$1};1\' | sort -u' % FEEDSTOCKS,
        shell=True,
        check=True,
        capture_output=True
    )
    ccos = subprocess.run(
        'compgen -G "%s/feedstocks/*/recipe/meta.yaml" | xargs cat | '
        'grep -E "\\-cos7\\-|\\-cos6\\-" | awk \'{$1=$1};1\' | sort -u' % FEEDSTOCKS,
        shell=True,
        check=True,
        capture_output=True
    )

    cdts = set()
    for line in ccdt.stdout.decode("utf-8").splitlines():
        line = line.strip()
        # this is a jinja2 cdt function (e.g. cdt('blah'))
        # split on cdt( and then )
        # we are left with a string that we can eval to get a python string
        cdt = eval(line.split("cdt(")[1].split(')')[0])
        cdts.add(cdt)

    for line in ccos.stdout.decode("utf-8").splitlines():
        if line.startswith("-"):
            # this is a requirement string
            # strip out starting "- " from yaml list
            line = line.replace("- ", "").strip()
            # split off the version string
            line = line.split(" ")[0]
            # remove the last two items separated by - (e.g., -cos6-x86_64)
            line = line.rsplit("-", maxsplit=1)[0]
            cdt = line.rsplit("-", maxsplit=1)[0]
        else:
            # this is a name entry in the recipe
            # remove the starting name:
            line = line.replace("name: ", "").strip()
            # remove the last two items separated by - (e.g., -cos6-x86_64)
            line = line.rsplit("-", maxsplit=1)[0]
            cdt = line.rsplit("-", maxsplit=1)[0]
        if cdt != "zip":
            cdts.add(cdt)

    return cdts


curr_cdts = get_curr_cdts()
abstract_cdts = compute_all_cdts()

all_cdts = set()
for slug in CDT_SLUGS:
    for cdt in abstract_cdts:
        all_cdts.add(cdt + slug)

print("needed cdts:")
for cdt in sorted(all_cdts - curr_cdts):
    print("    " + cdt)

with open("needed_cdt_list.yaml", "w") as fp:
    for cdt in sorted(all_cdts - curr_cdts):
        fp.write(cdt + ': false\n')
