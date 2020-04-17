import pytest

import exception_test
import testLib


def test_exception_catching_python_smoke():
    with pytest.raises(exception_test.LSSTException):
        testLib.failLSSTException1("message1")


def test_exception_catching_python_smoke_catch():
    try:
        testLib.failLSSTException1("message2")
    except exception_test.LSSTException as e:
        assert "message2" in repr(e)


def test_exception_catching_python_inherit():
    with pytest.raises(exception_test.LSSTException) as e:
        testLib.failCustomError1("message1")
    print(repr(e.value))


def test_exception_catching_python_inherit_catch():
    try:
        testLib.failCustomError1("message2")
    except exception_test.LSSTException as e:
        print(repr(e))
        assert "message2" in repr(e)
