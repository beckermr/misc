from conda_oci_utils import (
    encode_name_to_oci,
    decode_name_from_oci,
    encode_version_build_to_oci,
    decode_version_build_from_oci,
    encode_conda_dist_to_oci_dist,
    is_valid_oci_dist,
)

import pytest

@pytest.mark.parametrize("name,ename", [
    ("foo", "cfoo"),
    ("_foo", "zfoo"),
    ("bar", "cbar"),
    ("_bar", "zbar"),
    ("zbar", "czbar"),
    ("z_bar", "cz_bar"),
    ("_zbar", "zzbar"),
    ("_z_bar", "zz_bar"),
    ("_foo_bar", "zfoo_bar"),
    ("foo_bar", "cfoo_bar"),
    ("foo_zbar", "cfoo_zbar"),
    ("foo_z_bar", "cfoo_z_bar"),
])
def test_encode_decode_name_to_oci(name, ename):
    assert encode_name_to_oci(name) == ename
    assert decode_name_from_oci(ename) == name


@pytest.mark.parametrize("vb,ebv", [
    ("1.0.0", "1.0.0"),
    ("1.0.0_1", "1.0.0__1"),
    ("1.0.0+1", "1.0.0_P1"),
    ("1.0.0!1", "1.0.0_N1"),
    ("1.0.0=1", "1.0.0_E1"),
    ("1.0.0+1_1", "1.0.0_P1__1"),
    ("1.0.0+1!1", "1.0.0_P1_N1"),
    ("1.0.0_+1_1", "1.0.0___P1__1"),
])
def test_encode_decode_version_build_to_oci(vb, ebv):
    assert encode_version_build_to_oci(vb) == ebv
    assert decode_version_build_from_oci(ebv) == vb


@pytest.mark.parametrize("dist,oci_dist", [
    ("foo-1.0.0-1.conda", "cfoo:1.0.0-1"),
    ("foo-1.0.0_1-0", "cfoo:1.0.0__1-0"),
    ("foo-1.0.0+1-h34243_0", "cfoo:1.0.0_P1-h34243__0"),
    ("foo-1.0.0!1-h34243_0", "cfoo:1.0.0_N1-h34243__0"),
    ("foo-1.0.0=1-h34243_0.tar.bz2", "cfoo:1.0.0_E1-h34243__0"),
    ("foo-1.0.0+1_1-h34243_0", "cfoo:1.0.0_P1__1-h34243__0"),
    ("foo-1.0.0+1!1-h34243_0", "cfoo:1.0.0_P1_N1-h34243__0"),
    ("_foo-1.0.0_+1_1-h34243_0", "zfoo:1.0.0___P1__1-h34243__0"),
    ("cfoo-1.0.0_+1_1-h34243_0", "ccfoo:1.0.0___P1__1-h34243__0"),
    ("z_foo-1.0.0_+1_1-h34243_0", "cz_foo:1.0.0___P1__1-h34243__0"),
    ("cz_foo-1.0.0_+1_1-h34243_0", "ccz_foo:1.0.0___P1__1-h34243__0"),
])
def test_encode_conda_dist_to_oci_dist(dist, oci_dist):
    assert encode_conda_dist_to_oci_dist(dist) == oci_dist
    assert is_valid_oci_dist(oci_dist)


@pytest.mark.parametrize("dist,is_valid", [
    ("f" * 300 + ":1.0.0-1", False),
    ("_f" + ":1.0.0-1", False),
    ("_f" + ":1.0.0-1" + "4" * 300, False),
    ("b__f" + ":1.0.0-1", True),
    ("B__f" + ":1.0.0-1", False),
    ("b__f___g" + ":1.0.0-1", False),
])
def test_is_valid_oci_dist(dist, is_valid):
    assert is_valid_oci_dist(dist) is is_valid

