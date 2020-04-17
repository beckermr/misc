#include <string>

#include <pybind11/pybind11.h>

#include "../src/Exception.h"

using namespace exceptions;

template <typename T>
void fail1(std::string const &message) {
    throw T(message);
}

#define LSST_FAIL_TEST(name)                                                                 \
    mod.def("fail" #name "1", [](const std::string &message) { fail1<name>(message); });

PYBIND11_MODULE(testLib, mod) {
  LSST_FAIL_TEST(CustomError)
  LSST_FAIL_TEST(LSSTException)
}
