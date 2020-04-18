#include <string>
#include <pybind11/pybind11.h>
#include "../src/Exception.h"

PYBIND11_MODULE(testLib, mod) {
  mod.def("fail", [](const std::string &message) { throw lsst_exceptions::LSSTException(message); });
}
