# COMP=/usr/bin/clang++
COMP=${CXX}
ifeq ($(shell uname -s), "Linux")
	EXTRA_FLAGS=
else
	EXTRA_FLAGS=-undefined dynamic_lookup
endif


test: pymod
	cd tests && \
	${COMP} ${CXXFLAGS} -fPIC \
		`python -m pybind11 --includes` \
		-c testLib.cc \
		-o _testLib.o && \
	${COMP} ${CXXFLAGS} ${LDFLAGS} ${EXTRA_FLAGS} -shared -fPIC \
		`python -m pybind11 --includes` \
		_testLib.o \
		../src/Exception.o \
		-o testLib`python3-config --extension-suffix` && \
	 pytest -vvs test_exception_catching_python.py && \
	cd -

# chmod u+x testLib`python3-config --extension-suffix` && \

pymod: cpp
	cd exception_test && \
	${COMP} ${CXXFLAGS} -fPIC \
		`python -m pybind11 --includes` \
		-c exceptions.cc \
		-o _exceptions.o && \
	${COMP} ${CXXFLAGS} ${LDFLAGS} ${EXTRA_FLAGS} -shared -fPIC \
		`python -m pybind11 --includes` \
		_exceptions.o \
		../src/Exception.o \
		-o exceptions`python3-config --extension-suffix` && \
	cd - && \
	pip install -e .

cpp:
	cd src && \
	${COMP} ${CXXFLAGS} -fPIC \
		-c Exception.cc \
		-o Exception.o && \
	cd -
