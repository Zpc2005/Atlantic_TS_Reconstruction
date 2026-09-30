# Formal Environment Final Report

## Status

PASS

## Formal runtime

| Field | Value |
|---|---|
| Python executable | `project-local Python 3.12.10 runtime` |
| Python version | 3.12.10 |
| Environment location | `.venv` project-local runtime |
| Platform | Windows-11-10.0.26200-SP0, AMD64 |
| Device | CPU |
| CUDA available | false |
| CUDA version | None |
| PyTorch | 2.4.1+cpu |
| NumPy | 1.26.4 |
| pandas | not installed; not imported by frozen v2 inference |
| xarray | not installed; not imported by frozen v2 inference |
| scikit-learn | not installed; not imported by frozen v2 inference |

## Actual v2 inference dependencies

The frozen v2 inference path imports Python standard-library modules, NumPy, and
PyTorch. PyTorch installed its required runtime dependencies: filelock, fsspec,
Jinja2, MarkupSafe, mpmath, networkx, sympy, typing_extensions, and setuptools.
No CUDA, graph-library, pandas, xarray, scikit-learn, baseline-model, or
training-only dependency was installed.

## Installation decision

The repository does not preserve a historical PyTorch/NumPy lock or CUDA
record. The formal environment therefore uses CPython 3.12.10, the final
full-maintenance Python 3.12 Windows binary release, and CPU-only PyTorch
2.4.1 selected for the frozen API and checkpoint compatibility gate. This is a
recorded compatibility recovery, not a claim of historical version identity.

The `.venv` runtime is project-local and independent of the temporary Codex
runtime. `pip check` reports no broken requirements. The exact package set is
frozen in `Environment_Dependency_Freeze.txt`.


