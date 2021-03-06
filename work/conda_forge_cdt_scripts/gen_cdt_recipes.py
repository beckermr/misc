import os
import subprocess
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from ruamel.yaml import YAML

from cdt_config import (
    LEGACY_CDT_PATH,
    LEGACY_CUSTOM_CDT_PATH,
    CDT_PATH,
    CUSTOM_CDT_PATH,
    BUILD_NUM,
)

yaml = YAML()


def _make_cdt_recipes(*, extra, cdt_path, arch_dist_tuples, cdts, build_num, exec):
    futures = {}
    for arch, dist in arch_dist_tuples:
        for cdt, cfg in cdts.items():
            if cfg["custom"]:
                continue

            pth = os.path.join(
                cdt_path,
                cdt + "-" + dist.replace("ent", "") + "-" + arch,
            )

            if not os.path.exists(pth):
                _extra = extra + " --recursive"
            else:
                _extra = extra

            futures[exec.submit(
                subprocess.run,
                (
                    f"python rpm.py {cdt} --output-dir={cdt_path} "
                    + f"--architecture={arch} --distro={dist} "
                    + f"--build-number={build_num} "
                    + _extra
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=True
            )] = {"cdt": cdt, "arch": arch, "dist": dist}
    return futures


with open("cdt_slugs.yaml", "r") as fp:
    cdts = yaml.load(fp)

os.makedirs(LEGACY_CDT_PATH, exist_ok=True)
os.makedirs(LEGACY_CUSTOM_CDT_PATH, exist_ok=True)
os.makedirs(CDT_PATH, exist_ok=True)
os.makedirs(CUSTOM_CDT_PATH, exist_ok=True)

print("generating CDT recipes...")
futures = {}
with ThreadPoolExecutor(max_workers=20) as exec:
    # legacy CDTs for the old compiler sysroots
    extra = "--conda-forge-style"
    arch_dist_tuples = [
        ("x86_64", "centos6"),
        ("aarch64", "centos7"),
        ("ppc64le", "centos7")
    ]

    futures.update(
        _make_cdt_recipes(
            extra=extra,
            cdt_path=LEGACY_CDT_PATH,
            arch_dist_tuples=arch_dist_tuples,
            cdts=cdts,
            build_num=BUILD_NUM,
            exec=exec)
        )

    # new CDTs for the new compilers with a single sysroot
    extra = "--conda-forge-style --single-sysroot"
    arch_dist_tuples = [
        ("x86_64", "centos6"), ("x86_64", "centos7"),
        ("aarch64", "centos7"), ("ppc64le", "centos7")
    ]
    futures.update(
        _make_cdt_recipes(
            extra=extra,
            cdt_path=CDT_PATH,
            arch_dist_tuples=arch_dist_tuples,
            cdts=cdts,
            build_num=BUILD_NUM,
            exec=exec)
        )

    for fut in tqdm.tqdm(as_completed(futures), total=len(futures)):
        c = fut.result()
        pkg = futures[fut]
        nm = "-".join([pkg["cdt"], pkg["dist"].replace("ent", ""), pkg["arch"]])
        if c.returncode != 0:
            tqdm.tqdm.write("WARNING: making CDT recipe %s failed!" % nm)
