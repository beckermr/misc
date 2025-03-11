"""
Utilities to deal with repodata
"""

import bz2
import json
import os
from functools import lru_cache
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union
from urllib.request import urlretrieve

import bs4
import requests

logger = getLogger(__name__)

SUBDIRS = (
    "linux-64",
    "linux-aarch64",
    "linux-ppc64le",
    "osx-64",
    "osx-arm64",
    "win-64",
    "win-arm64",
    "noarch",
)
CACHE_DIR = Path(".repodata_cache")
DEFAULTS_CHANNELS = ["main", "msys2", "r"]


@lru_cache(maxsize=128)
def all_labels(channel) -> List[str]:
    if True:
        r = requests.get(f"https://anaconda.org/{channel}/repo")
        r.raise_for_status()
        html = r.text
        soup = bs4.BeautifulSoup(html, "html.parser")
        labels = []
        len_prefix = len("/conda-forge/repo?label=")
        for element in soup.select("ul#Label > li > a"):
            href = element.get("href")
            if not href:
                continue
            label = href[len_prefix:]
            if label and label not in ("all", "empty") and "/" not in label:
                labels.append(label)
        return sorted(labels)
    else:
        token = os.environ["BINSTAR_TOKEN"]
        label_info = requests.get(
            "https://api.anaconda.org/channels/{channel}",
            headers={"Authorization": f"token {token}"},
        ).json()

        return sorted(label for label in label_info if "/" not in label)


def fetch_repodata(
    channel: str = "conda-forge",
    subdirs: Iterable[str] = SUBDIRS,
    force_download: bool = False,
    cache_dir: Union[str, Path] = CACHE_DIR,
    label: str = "main",
) -> List[Path]:
    assert all(subdir in SUBDIRS for subdir in subdirs)
    paths = []
    for subdir in subdirs:
        if label == "main":
            repodata = f"https://conda.anaconda.org/{channel}/{subdir}/repodata.json"
        else:
            repodata = (
                f"https://conda.anaconda.org/{channel}/"
                f"label/{label}/{subdir}/repodata.json"
            )
        local_fn = Path(cache_dir, f"{channel}.{subdir}.{label}.json")
        local_fn_bz2 = Path(str(local_fn) + ".bz2")
        paths.append(local_fn)
        if force_download or not local_fn.exists():
            logger.info(f"Downloading {repodata} to {local_fn}")
            local_fn.parent.mkdir(parents=True, exist_ok=True)
            # Download the file
            urlretrieve(f"{repodata}.bz2", local_fn_bz2)
            with open(local_fn_bz2, "rb") as compressed, open(local_fn, "wb") as f:
                f.write(bz2.decompress(compressed.read()))
            local_fn_bz2.unlink()
    return paths


def repodata(channel: str, subdir: str) -> Dict[str, Any]:
    assert subdir in SUBDIRS
    path = fetch_repodata(channel=channel, subdirs=(subdir,))[0]
    return json.loads(path.read_text())
