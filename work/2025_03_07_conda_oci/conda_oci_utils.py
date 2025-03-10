"""utilities for conda oci work."""

import hashlib
import re
import urllib.parse

from conda.models.version import VersionOrder

# see https://github.com/opencontainers/distribution-spec/blob/main/spec.md#pulling-manifests
VALID_NAME_RE = re.compile(
    r"^[a-z0-9]+((\.|_|__|-+)[a-z0-9]+)*(\/[a-z0-9]+((\.|_|__|-+)[a-z0-9]+)*)*$"
)
VALID_TAG_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127}$")

# see conda/schemas
VALID_CONDA_NAME_RE = re.compile(
    r"^(([a-z0-9])|([a-z0-9_](?!_)))[._-]?([a-z0-9]+(\.|-|_|$))*$"
)
VALID_CONDA_CHANNEL_SUBDIR_RE = re.compile(r"^[a-z0-9]+((-|_|.)[a-z0-9]+)*$")
VALID_CONDA_LABEL_RE = re.compile(r"^[a-zA-Z][0-9a-zA-Z_\-\.\/:\s]*")


def encode_underscore_to_oci(name):
    """Encode a conda package name to an OCI image name."""

    if name.startswith("_"):
        name = "z" + name[1:]
    else:
        name = "c" + name

    return name


def decode_underscore_from_oci(name):
    """Decode an OCI image name to a conda package name."""

    if name.startswith("z"):
        name = "_" + name[1:]
    elif name.startswith("c"):
        name = name[1:]

    return name


def encode_label_version_build_to_oci(version_or_build):
    """Encode a conda package version or build string to an OCI image tag."""

    return (
        version_or_build.replace("_", "__")
        .replace("+", "_P")
        .replace("!", "_N")
        .replace("=", "_E")
        .replace(":", "_C")
        .replace("/", "_S")
        .replace(" ", "_B")
        .replace("\t", "_T")
        .replace("\r", "_R")
        .replace("\n", "_L")
    )


def decode_label_version_build_from_oci(version_or_build):
    """Decode an OCI image tag to a conda package version or build string."""

    return (
        version_or_build.replace("_L", "\n")
        .replace("_R", "\r")
        .replace("_T", "\t")
        .replace("_B", " ")
        .replace("_S", "/")
        .replace("_C", ":")
        .replace("_E", "=")
        .replace("_N", "!")
        .replace("_P", "+")
        .replace("__", "_")
    )


def _split_channel_label_subdir_name(channel_label_subdir_name):
    channel_label_subdir, name = channel_label_subdir_name.rsplit("/", maxsplit=1)
    channel, label_subdir = channel_label_subdir.split("/", maxsplit=1)
    res = label_subdir.rsplit("/", maxsplit=1)
    if len(res) == 2:
        label, subdir = res
    else:
        label = None
        subdir = res[0]
    return channel, label, subdir, name


def encode_conda_dist_to_oci_dist(dist):
    """Convert a conda package name to an OCI image name."""

    if dist.endswith(".tar.bz2"):
        dist = dist[:-8]
    elif dist.endswith(".conda"):
        dist = dist[:-6]

    name, ver, build = dist.rsplit("-", maxsplit=2)
    channel, label, subdir, name = _split_channel_label_subdir_name(name)

    name = encode_underscore_to_oci(name)
    ver = encode_label_version_build_to_oci(ver)
    build = encode_label_version_build_to_oci(build)
    if label:
        label = urllib.parse.unquote(label)
        label = encode_label_version_build_to_oci(label)

    channel_subdir = f"{channel}/{subdir}"
    oci_name = f"{channel_subdir}/{name}"
    if label is not None:
        oci_tag = f"{label}-{ver}-{build}"
    else:
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

    name_parts = name.split("/")
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

    name = decode_underscore_from_oci(name)
    tag_parts = tag.rsplit("-", maxsplit=2)
    if len(tag_parts) == 3:
        label, ver, build = tag_parts
    else:
        label = None
        ver, build = tag_parts
    ver = decode_label_version_build_from_oci(ver)
    build = decode_label_version_build_from_oci(build)
    if label:
        label = decode_label_version_build_from_oci(label)
        if urlencode_label:
            label = urllib.parse.quote(label, safe="")

    if channel is not None and subdir is not None:
        if label is not None:
            prefix = f"{channel}/{label}/{subdir}/"
        else:
            prefix = f"{channel}/{subdir}/"
    else:
        prefix = ""

    return prefix + f"{name}-{ver}-{build}"


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
        channel, label, subdir, name = _split_channel_label_subdir_name(name)
    else:
        channel = None
        label = None
        subdir = None

    if channel is not None and not VALID_CONDA_CHANNEL_SUBDIR_RE.match(channel):
        return False

    if label is not None and not VALID_CONDA_LABEL_RE.match(label):
        return False

    if subdir is not None and not VALID_CONDA_CHANNEL_SUBDIR_RE.match(subdir):
        return False

    if not VALID_CONDA_NAME_RE.match(name):
        return False

    try:
        VersionOrder(ver)
    except Exception:
        return False

    return True
