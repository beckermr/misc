# tests of exception passing in pybind11

### install your env

```bash
conda create -n pybind11-test python=3.7 pybind11 compilers pytest git make
```

### run

```bash
make test
cd tests
pytest -vvs test_exception_catching_python
```

### old lib install

```bash
conda install -c conda-forge/label/broken libcxx==9.0.0-0
```
