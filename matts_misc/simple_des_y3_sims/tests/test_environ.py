import os

from ..files import environ


def test_environ():
    assert os.environ['USER'] != 'xyz'
    val = os.environ['USER']
    with environ(USER='xyz'):
        assert os.environ['USER'] == 'xyz'
    assert os.environ['USER'] == val
