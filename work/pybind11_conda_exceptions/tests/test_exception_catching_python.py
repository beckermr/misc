from __future__ import print_function
import testLib


try:
    testLib.fail("message1")
except Exception as e:
    print(repr(e))
    assert repr(e) == "LSSTException('message1')"
