COMP=${CXX}
ifeq ($(shell uname -s), Linux)
	EXTRA_FLAGS=
	PRINTLIBS=
else
	EXTRA_FLAGS=-undefined dynamic_lookup
	PRINTLIBS=DYLD_PRINT_LIBRARIES=YES

	USE_SYSTEM?=NO

	ifeq ($(USE_SYSTEM), YES)
		COMP=/usr/bin/clang++
		CXXFLAGS=\
			-march=core2 -mtune=haswell -mssse3 -ftree-vectorize -fPIC -fPIE \
			-fstack-protector-strong -O2 -pipe -stdlib=libc++ -fvisibility-inlines-hidden \
			-std=c++14 -fmessage-length=0
		LDFLAGS=
	endif


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
	  ${PRINTLIBS} pytest -vvs test_exception_catching_python.py && \
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

clean:
	rm -f src/*.o src/*.so tests/*.o tests/*.so exception_test/*.o exception_test/*.so
