# tests of exception passing in pybind11

### install your env

```bash
conda create -n pybind11-test python=3.7 pybind11 compilers ipython pytest
```

### run

```bash
make test
cd tests
pytest -vvs test_exception_catching_python
```
