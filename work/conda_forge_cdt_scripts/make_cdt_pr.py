import sys
import subprocess
import contextlib
import os
from collections.abc import MutableMapping, MutableSequence

from ruamel.yaml import YAML

yaml = YAML(typ='jinja2')
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.width = 320

DEFAULTS_REPO = "/Users/beckermr/Desktop/aggregate"


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def _run_shell_command(cmd):
    subprocess.run(
        cmd,
        shell=True,
        check=True
    )


def _reorder_meta(pth):
    entry_order = [
        'package', 'source', 'build',
        'requirements', 'outputs', 'test', 'about',
        'extra'
    ]

    entries = {}
    preamble = []
    in_preamble = True
    curr_entry = None
    has_license_file = False
    with open(pth, "r") as fp:
        for line in fp.readlines():
            if "license_file:" in line:
                has_license_file = True

            if len(line.strip()) == 0:
                continue

            if any(line.startswith(e+":") for e in entry_order):
                in_preamble = False

            if in_preamble:
                preamble.append(line)
                continue

            if curr_entry is None:
                for e in entry_order:
                    if line.startswith(e+":"):
                        curr_entry = e
            else:
                for e in entry_order:
                    if line.startswith(e+":") and curr_entry != e:
                        curr_entry = e

            elines = entries.get(curr_entry, [])
            elines.append(line)
            entries[curr_entry] = elines
            continue

    with open(pth, "w") as fp:
        for line in preamble:
            fp.write(line)
        if preamble:
            fp.write("\n")

        for e in entry_order:
            elines = entries.get(e, [])
            for line in elines:
                fp.write(line)

                if "license:" in line and not has_license_file:
                    whsp = ""
                    for c in line:
                        if c.isspace():
                            whsp += c
                        else:
                            break
                    fp.write(whsp + "# license_file: in the rpm source\n")
                    has_license_file = True

            if elines and e != "extra":
                fp.write("\n")


with open("cdt_list.yaml", "r") as fp:
    cdts = yaml.load(fp)


cdt_to_do = sys.argv[1]

if cdt_to_do not in cdts:
    raise RuntimeError("could not find CDT!")

if cdts[cdt_to_do]:
    print("CDT already done!")
    sys.exit(0)

_run_shell_command("rm -rf staged-recipes")
_run_shell_command("git clone https://github.com/beckermr/staged-recipes.git")
with pushd("staged-recipes"):
    _run_shell_command(
        "git remote add upstream https://github.com/conda-forge/staged-recipes.git")
    _run_shell_command("git pull upstream master")
    _run_shell_command("git push")
    try:
        _run_shell_command(f"git checkout {cdt_to_do}-build")
    except Exception:
        _run_shell_command(f"git checkout -b {cdt_to_do}-build")
    _run_shell_command(f"cp -r {DEFAULTS_REPO}/{cdt_to_do} recipes/{cdt_to_do}")
    with pushd(f"recipes/{cdt_to_do}"):
        with open("meta.yaml", "r") as fp:
            meta = yaml.load(fp)

        if "extra" not in meta:
            meta["extra"] = {}
            meta["extra"]["recipe-maintainers"] = ["conda-forge/Core"]

        if "source" in meta:
            if (
                isinstance(meta["source"], MutableMapping)
                and "no_hoise" not in meta["source"]
            ):
                meta["source"]["no_hoist"] = True
            elif isinstance(meta["source"], MutableSequence):
                for src in meta["source"]:
                    if "no_hoist" not in src:
                        src["no_hoist"] = True
            else:
                raise RuntimeError("source is not dict or list!")

        if "build" in meta and "number" not in meta["build"]:
            meta["build"]["number"] = 0

        if "test" not in meta:
            meta["test"] = {}
            meta["test"]["commands"] = []
            if cdt_to_do.endswith("-cos6-x86_64"):
                cmd = "test -d \"${PREFIX}/x86_64-conda_cos6-linux-gnu/sysroot\""
            elif cdt_to_do.endswith("-cos7-aarch64"):
                cmd = "test -d \"${PREFIX}/aarch64-conda_cos7-linux-gnu/sysroot\""
            elif cdt_to_do.endswith("-cos7-ppc64le"):
                cmd = "test -d \"${PREFIX}/powerpc64le-conda_cos7-linux-gnu/sysroot\""
            meta["test"]["commands"].append(cmd)

        with open("meta.yaml", "w") as fp:
            yaml.dump(meta, fp)

        _reorder_meta("meta.yaml")

    _run_shell_command(f"git add recipes/{cdt_to_do}/*")
    _run_shell_command(f"git ci -m 'ENH added {cdt_to_do}'")
    _run_shell_command(f"git push --set-upstream origin {cdt_to_do}-build")

cdts[cdt_to_do] = True
with open("cdt_list.yaml", "w") as fp:
    yaml.dump(cdts, fp)
