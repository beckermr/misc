COMP=${CXX}
ifeq ($(shell uname -s), Linux)
	EXTRA_FLAGS=
	PRINTLIBS=
	SUFFIX=$(shell python3-config --extension-suffix)
else
	EXTRA_FLAGS=-undefined dynamic_lookup
	PRINTLIBS=DYLD_PRINT_LIBRARIES=YES
	SUFFIX=`python3-config --extension-suffix`
	USE_SYSTEM?=NO

	ifeq ($(USE_SYSTEM), YES)
		COMP=/usr/bin/clang++
		CXXFLAGS=-fPIC -O2 -stdlib=libc++ -std=c++14 -isystem /Users/beckermr/miniconda3/envs/pybind11-test
		LDFLAGS=
		SUFFIX=.so
	else
	  SUFFIX=$(shell python3-config --extension-suffix)
	endif


endif

test: pymod
	cd tests && \
	${COMP} ${CXXFLAGS} -fPIC \
		`python -m pybind11 --includes` \
		-c testLib.cc \
		-o _testLib.o && \
	${COMP} ${CXXFLAGS} ${LDFLAGS} ${EXTRA_FLAGS} -shared \
		`python -m pybind11 --includes` \
		_testLib.o \
		../src/Exception.o \
		-o testLib${SUFFIX} && \
	  ${PRINTLIBS} python test_exception_catching_python.py && \
	cd -

# chmod u+x testLib`python3-config --extension-suffix` && \

pymod: cpp
	cd exception_test && \
	${COMP} ${CXXFLAGS} -fPIC \
		`python -m pybind11 --includes` \
		-c exceptions.cc \
		-o _exceptions.o && \
	${COMP} ${CXXFLAGS} ${LDFLAGS} ${EXTRA_FLAGS} -shared \
		`python -m pybind11 --includes` \
		_exceptions.o \
		../src/Exception.o \
		-o exceptions${SUFFIX} && \
	cd - && \
	python -m pip install -e .

cpp:
	cd src && \
	${COMP} ${CXXFLAGS} -fPIC \
		-c Exception.cc \
		-o Exception.o && \
	cd -

clean:
	rm -f src/*.o src/*.so tests/*.o tests/*.so exception_test/*.o exception_test/*.so
