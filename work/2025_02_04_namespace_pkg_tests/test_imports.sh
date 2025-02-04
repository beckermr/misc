#!/usr/bin/env bash

find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

export PYTHONPATH=$(pwd)/pkg_a:$(pwd)/pkg_b

echo "test b.c"
python -c "import lsst.b.c; print('worked!')"

echo "test a, b, b.c"
python -c "import lsst.a; import lsst.b; import lsst.b.c; print('worked!')"

echo "test b"
python -c "import lsst.b; print('worked!')"

find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

export PYTHONPATH=$(pwd)/pkg_b:$(pwd)/pkg_a

echo "test b.c"
python -c "import lsst.b.c; print('worked!')"

echo "test a, b, b.c"
python -c "import lsst.a; import lsst.b; import lsst.b.c; print('worked!')"

echo "test b"
python -c "import lsst.b; print('worked!')"
