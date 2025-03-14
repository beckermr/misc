"""utilities for conda oci work."""

import hashlib
import re

from conda.models.version import VersionOrder

# see https://github.com/opencontainers/distribution-spec/blob/main/spec.md#pulling-manifests
VALID_NAME_RE = re.compile(
    r"^[a-z0-9]+((\.|_|__|-+)[a-z0-9]+)*(\/[a-z0-9]+((\.|_|__|-+)[a-z0-9]+)*)*$"
)
VALID_TAG_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127}$")

# see conda/schemas
VALID_CONDA_PKG_NAME_RE = re.compile(r"^(([a-z0-9])|([a-z0-9_](?!_)))[._-]?([a-z0-9]+(\.|-|_|$))*$")
VALID_CONDA_CHANNEL_RE = re.compile(r"^[a-z0-9]+((-|_|.)[a-z0-9]+)*$")
VALID_CONDA_SUBDIR_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
VALID_CONDA_LABEL_RE = re.compile(r"^[a-zA-Z][0-9a-zA-Z_\-\.\/:\s]*")
VALID_CONDA_BUILD_STRING_RE = re.compile(r"^[a-zA-Z0-9_\.+]+$")


def encode_pkg_to_oci(name):
    """Encode a conda package name to an OCI image name."""
    return "c" + name


def decode_pkg_from_oci(name):
    """Decode an OCI image name to a conda package name."""
    return name[1:]


def encode_version_build_to_oci(version_or_build):
    """Encode a conda package version or build string to an OCI image tag."""
    return (
        version_or_build
        .replace("_", "__")
        .replace("+", "_P")
        .replace("!", "_N")
    )


def decode_version_build_from_oci(version_or_build):
    """Decode an OCI image tag to a conda package version or build string."""
    return (
        version_or_build
        .replace("_N", "!")
        .replace("_P", "+")
        .replace("__", "_")
    )


def encode_conda_dist_to_oci_dist(dist):
    """Convert a conda package name to an OCI image name."""

    if dist.endswith(".tar.bz2"):
        dist = dist[:-8]
    elif dist.endswith(".conda"):
        dist = dist[:-6]

    name, ver, build = dist.rsplit("-", maxsplit=2)
    channel, subdir, name = name.rsplit("/", maxsplit=2)

    name = encode_pkg_to_oci(name)
    ver = encode_version_build_to_oci(ver)
    build = encode_version_build_to_oci(build)

    channel_subdir = f"{channel}/{subdir}"
    oci_name = f"{channel_subdir}/{name}"
    oci_tag = f"{ver}-{build}"

    if len(oci_name) > 128 or len(oci_tag) > 128:
        oci_tag = "h" + hashlib.sha1(oci_tag.encode("ascii")).hexdigest()
        oci_name = (
            channel_subdir + "/h" + hashlib.sha1(name.encode("ascii")).hexdigest()
        )

    return f"{oci_name}:{oci_tag}"


def decode_oci_dist_to_conda_dist(dist, urlencode_label=True):
    """Convert an OCI image name to a conda package name."""

    if dist.startswith("oci://"):
        # assume name is oci://<registry>/<image>:tag
        # strip out oci:// and the registry
        dist = dist[6:]
        dist = dist.split("/", maxsplit=1)[-1]

    name, tag = dist.rsplit(":", maxsplit=1)
    if tag.startswith("h"):
        raise ValueError(
            "OCI dist names with hashed components cannot be "
            "decoded. Read the image metadata to find the "
            "conda package name."
        )

    name_parts = name.rsplit("/", maxsplit=2)
    if len(name_parts) != 3:
        raise ValueError(
            "channel and subdir information must be "
            "prepended in the format <channel>/<subdir>"
            f"/<oci dist>. Got {name} which cannot be interpreted."
        )
    channel, subdir, name = name_parts
    if name.startswith("h"):
        raise ValueError(
            "OCI dist names with hashed components cannot be "
            "decoded. Read the image metadata to find the "
            "conda package name."
        )

    name = decode_pkg_from_oci(name)
    ver, build = tag.rsplit("-", maxsplit=2)
    ver = decode_version_build_from_oci(ver)
    build = decode_version_build_from_oci(build)

    if channel is not None and subdir is not None:
        prefix = f"{channel}/{subdir}/"
    else:
        prefix = ""

    return prefix + f"{name}-{ver}-{build}"


def separate_channel_label(channel_label):
    parts = channel_label.split("/")
    if any(part == "label" for part in parts) and parts[-1] != "label":
        # get last index of part that is "label"
        for i, part in enumerate(parts):
            if part == "label":
                label_index = i
        channel = "/".join(parts[:label_index])
        label = "/".join(parts[label_index + 1 :])
    else:
        channel = channel_label
        label = None

    return channel, label


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
    if "/" in name:
        channel, subdir, name = name.rsplit("/", maxsplit=2)
    else:
        channel = None
        subdir = None

    if channel is not None:
        channel, label = separate_channel_label(channel)
    else:
        label = None

    if channel is not None and not VALID_CONDA_CHANNEL_RE.match(channel):
        return False

    if label is not None and not VALID_CONDA_LABEL_RE.match(label):
        return False

    if subdir is not None and not VALID_CONDA_SUBDIR_RE.match(subdir):
        return False

    if not VALID_CONDA_PKG_NAME_RE.match(name):
        return False

    if not VALID_CONDA_BUILD_STRING_RE.match(build):
        return False

    try:
        VersionOrder(ver)
    except Exception:
        return False

    return True
