COMP=${CXX}
ifeq ($(shell uname -s), Linux)
	EXTRA_FLAGS=
	PRINTLIBS=LD_LIBRARY_PATH=../src
	SUFFIX=$(shell python3-config --extension-suffix)
	SOEXT=.so
else
	EXTRA_FLAGS=-undefined dynamic_lookup
	PRINTLIBS=DYLD_FALLBACK_LIBRARY_PATH=../src DYLD_PRINT_LIBRARIES=YES
	SUFFIX=`python3-config --extension-suffix`
	USE_SYSTEM?=NO
	SOEXT=.dylib

	ifeq ($(USE_SYSTEM), YES)
		COMP=/usr/bin/clang++
		CXXFLAGS=-fPIC -O2 -stdlib=libc++ -std=c++14 -isystem /Users/beckermr/miniconda3/envs/pybind11-test
		LDFLAGS=
		SUFFIX=.so
	else
	  SUFFIX=$(shell python3-config --extension-suffix)
	endif


endif

test: clean pymod
	cd tests && \
	${COMP} ${CXXFLAGS} ${LDFLAGS} ${EXTRA_FLAGS} -shared -fPIC \
		`python -m pybind11 --includes` \
		testLib.cc \
		-L../src -lException \
		-o testLib${SUFFIX} && \
		echo "\n\n============================================\ntesting\n============================================\n" && \
	  ${PRINTLIBS} python test_exception_catching_python.py && \
	cd -

# chmod u+x testLib`python3-config --extension-suffix` && \

pymod: cpp
	cd exception_test && \
	${COMP} ${CXXFLAGS} ${LDFLAGS} ${EXTRA_FLAGS} -shared -fPIC \
		`python -m pybind11 --includes` \
		exceptions.cc \
		-L../src -lException \
		-o exceptions${SUFFIX} && \
	cd - && \
	python -m pip install -e .

cpp:
	cd src && \
	${COMP} ${CXXFLAGS} ${LDFLAGS} ${EXTRA_FLAGS} -shared -fPIC \
		Exception.cc \
		-o libException${SOEXT} && \
	cd -

clean:
	rm -f \
		src/*.o src/*.so src/*.dylib \
		tests/*.o tests/*.so tests/*.dylib \
		exception_test/*.o exception_test/*.so exception_test/*.dylib
