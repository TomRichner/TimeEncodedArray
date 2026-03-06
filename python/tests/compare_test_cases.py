"""
compare_test_cases.py — Compare Python vs MATLAB TEA files from Python's perspective.

For each case K=1..8, reads py_caseK.mat and ml_caseK.mat via h5py
and compares every variable for identical values.

Tolerance: doubles < 1e-12, char/logical/int exact.
"""

import sys
import os
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def get_cv_dir():
    d = os.path.join(os.path.dirname(__file__), '..', '..', 'cross_validation_data')
    return os.path.abspath(d)


def read_matlab_var(f, name):
    """Read an HDF5 dataset and return it in a normalized form."""
    ds = f[name]
    data = ds[()]

    # Determine MATLAB class from attribute
    matlab_class = None
    if 'MATLAB_class' in ds.attrs:
        matlab_class = ds.attrs['MATLAB_class']
        if isinstance(matlab_class, bytes):
            matlab_class = matlab_class.decode('ascii')

    if matlab_class == 'char':
        # Char array: uint16 values → string
        return ''.join(chr(c) for c in data.ravel())
    elif matlab_class == 'logical':
        return ('logical', bool(np.squeeze(data)))
    else:
        # Numeric: transpose from HDF5 (col-major storage) to row-major
        arr = np.squeeze(data).astype(np.float64)
        if data.ndim == 2:
            arr = data.T.astype(np.float64)
        return arr


def compare_files(py_file, ml_file):
    n_pass = 0
    n_fail = 0

    with h5py.File(py_file, 'r') as fpy, h5py.File(ml_file, 'r') as fml:
        py_vars = sorted([k for k in fpy.keys() if not k.startswith('#')])
        ml_vars = sorted([k for k in fml.keys() if not k.startswith('#')])

        all_vars = sorted(set(py_vars) | set(ml_vars))

        for var in all_vars:
            if var not in py_vars:
                print(f"  {var:<20s} FAIL (missing in Python)")
                n_fail += 1
                continue
            if var not in ml_vars:
                print(f"  {var:<20s} FAIL (missing in MATLAB)")
                n_fail += 1
                continue

            val_py = read_matlab_var(fpy, var)
            val_ml = read_matlab_var(fml, var)

            # String comparison
            if isinstance(val_py, str) and isinstance(val_ml, str):
                if val_py == val_ml:
                    print(f"  {var:<20s} PASS (char)")
                    n_pass += 1
                else:
                    print(f"  {var:<20s} FAIL (char: '{val_py}' vs '{val_ml}')")
                    n_fail += 1
                continue

            # Logical comparison
            if isinstance(val_py, tuple) and isinstance(val_ml, tuple):
                if val_py[1] == val_ml[1]:
                    print(f"  {var:<20s} PASS (logical)")
                    n_pass += 1
                else:
                    print(f"  {var:<20s} FAIL (logical: {val_py[1]} vs {val_ml[1]})")
                    n_fail += 1
                continue

            # Numeric comparison
            val_py = np.atleast_1d(val_py)
            val_ml = np.atleast_1d(val_ml)

            # Both empty = pass (handle shape variations like (0,0) vs (0,))
            if val_py.size == 0 and val_ml.size == 0:
                print(f"  {var:<20s} PASS (empty)")
                n_pass += 1
                continue

            if val_py.shape != val_ml.shape:
                print(f"  {var:<20s} FAIL (shape: {val_py.shape} vs {val_ml.shape})")
                n_fail += 1
                continue

            max_diff = np.max(np.abs(val_py.ravel() - val_ml.ravel()))
            if max_diff < 1e-12:
                print(f"  {var:<20s} PASS ({val_py.shape}, max_diff={max_diff:.2e})")
                n_pass += 1
            else:
                print(f"  {var:<20s} FAIL ({val_py.shape}, max_diff={max_diff:.2e})")
                n_fail += 1

    return n_pass, n_fail


def main():
    cv_dir = get_cv_dir()
    print("Python Comparator: Python vs MATLAB TEA files")
    print(f"Directory: {cv_dir}")
    print("=" * 60)

    total_pass = 0
    total_fail = 0

    for k in range(1, 9):
        py_file = os.path.join(cv_dir, f'py_case{k}.mat')
        ml_file = os.path.join(cv_dir, f'ml_case{k}.mat')

        print(f"\nCase {k}:")

        if not os.path.exists(py_file):
            print(f"  SKIP — py_case{k}.mat not found")
            continue
        if not os.path.exists(ml_file):
            print(f"  SKIP — ml_case{k}.mat not found")
            continue

        np_, nf = compare_files(py_file, ml_file)
        total_pass += np_
        total_fail += nf

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_pass} passed, {total_fail} failed")
    if total_fail == 0:
        print("ALL IDENTICAL")
    else:
        print("SOME DIFFERENCES FOUND")

    return total_fail == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
