# tests of exception passing in pybind11

### install your env

```bash
conda create -n pybind11-test python=3.7 pybind11 compilers ipython pytest
```

### run

```bash
make cpp
make pymod
make test
pytest -vvs tests/test_exception_catching_python
```
