import pytest

import exception_test
import testLib


def test_exception_catching_python_smoke():
    with pytest.raises(exception_test.LSSTException):
        testLib.fail("message1")
