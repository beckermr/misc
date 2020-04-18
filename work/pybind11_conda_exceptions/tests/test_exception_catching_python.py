from __future__ import print_function
import exception_test  # noqa
import testLib


try:
    testLib.fail("message1")
except Exception as e:
    print("\nexception:", repr(e), "\n")
    assert repr(e) == "LSSTException('message1')"

print("\nPASSED")
