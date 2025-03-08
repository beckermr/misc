"""utilities for conda oci work."""
import re

from conda.models.version import VersionOrder

# see https://github.com/opencontainers/distribution-spec/blob/main/spec.md#pulling-manifests
VALID_NAME_RE = re.compile(r"^[a-z0-9]+((\.|_|__|-+)[a-z0-9]+)*(\/[a-z0-9]+((\.|_|__|-+)[a-z0-9]+)*)*$")
VALID_TAG_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127}$")

# see conda/schemas
VALID_CONDA_NAME_RE = re.compile(r"^(([a-z0-9])|([a-z0-9_](?!_)))[._-]?([a-z0-9]+(\.|-|_|$))*$")


def encode_name_to_oci(name):
    """Encode a conda package name to an OCI image name."""

    if name.startswith("_"):
        name = "z" + name[1:]
    else:
        name = "c" + name

    return name


def decode_name_from_oci(name):
    """Decode an OCI image name to a conda package name."""

    if name.startswith("z"):
        name = "_" + name[1:]
    elif name.startswith("c"):
        name = name[1:]

    return name


def encode_version_build_to_oci(version_or_build):
    """Encode a conda package version or build string to an OCI image tag."""

    return (
        version_or_build
        .replace("_", "__")
        .replace("+", "_P")
        .replace("!", "_N")
        .replace("=", "_E")
    )


def decode_version_build_from_oci(version_or_build):
    """Decode an OCI image tag to a conda package version or build string."""

    return (
        version_or_build
        .replace("_P", "+")
        .replace("_N", "!")
        .replace("_E", "=")
        .replace("__", "_")
    )


def encode_conda_dist_to_oci_dist(dist, channel=None, subdir=None):
    """Convert a conda package name to an OCI image name."""

    if dist.endswith(".tar.bz2"):
        dist = dist[:-8]
    elif dist.endswith(".conda"):
        dist = dist[:-6]

    name, ver, build = dist.rsplit("-", maxsplit=2)

    name = encode_name_to_oci(name)
    ver = encode_version_build_to_oci(ver)
    build = encode_version_build_to_oci(build)

    if channel is not None and subdir is not None:
        prefix = f"{channel}/{subdir}/"
    else:
        prefix = ""

    return prefix + f"{name}:{ver}-{build}"


def is_valid_oci_dist(dist):
    """Check if an oci dist name is valid."""
    name, tag = dist.rsplit(":", maxsplit=1)

    if not VALID_TAG_RE.match(tag):
        return False

    if name.startswith("oci://"):
        name = name[6:]

    if len(name) > 255:
        return False

    if not VALID_NAME_RE.match(name):
        return False

    return True


def is_valid_conda_dist(dist):
    """Check if a conda dist is valid."""

    if dist.endswith(".tar.bz2"):
        dist = dist[:-8]
    elif dist.endswith(".conda"):
        dist = dist[:-6]

    name, ver, build = dist.rsplit("-", maxsplit=2)

    if not VALID_CONDA_NAME_RE.match(name):
        return False

    try:
        VersionOrder(ver)
    except Exception:
        return False

    return True
