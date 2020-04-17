test: pymod
	cd tests && \
	${CXX} ${CXXFLAGS} -undefined dynamic_lookup -shared -fPIC \
		`python -m pybind11 --includes` \
		-c testLib.cc \
		-o _testLib.o && \
	${CXX} ${CXXFLAGS} -undefined dynamic_lookup -shared -fPIC \
		`python -m pybind11 --includes` \
		_testLib.o \
		../src/Exception.o \
		-o testLib`python3-config --extension-suffix` && \
	cd -

# chmod u+x testLib`python3-config --extension-suffix` && \

pymod: cpp
	cd exception_test && \
	${CXX} ${CXXFLAGS} -undefined dynamic_lookup -shared -fPIC \
		`python -m pybind11 --includes` \
		-c exceptions.cc \
		-o _exceptions.o && \
	${CXX} ${CXXFLAGS} -undefined dynamic_lookup -shared -fPIC \
		`python -m pybind11 --includes` \
		_exceptions.o \
		../src/Exception.o \
		-o exceptions`python3-config --extension-suffix` && \
	cd - && \
	pip install -e .

cpp:
	cd src && \
	${CXX} ${CXXFLAGS} -undefined dynamic_lookup -shared -fPIC \
		-c Exception.cc \
		-o Exception.o && \
	cd -
