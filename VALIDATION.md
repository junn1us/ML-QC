# Local validation record — November 6, 2026

## Executed checks

* Python **3.14.2**, isolated starter virtual environment. Python 3.12 was not
  tested. NumPy 2.4.1, SciPy 1.17.0, scikit-learn 1.8.0, pandas 3.0.0,
  Matplotlib 3.10.8.
* **36/36 unittest checks passed**: 29 mathematical/statistical tests plus seven
  adapter/workflow tests. External actual-reward mode was exercised with
  explicitly synthetic test fixtures, not project measurements.
* Full default demo completed with seed 20261106, 1,500 fitting sheets, independent
  20,000-sheet calibration/validation pools, 256 paths and 4,000-sheet horizons.
* All ten expected artifacts were produced; calibration and validation have six
  rows each, fault comparisons 54 rows, sheet trace 500 rows, subgroup trace 100.
* Saved NPZ shape `(500,100,3)` agrees with the exported maximum-Uz features.
  CUSUM/EWMA CSV recurrences were independently recomputed from saved loss and
  model moments. Subgroup endings are exactly sheets 5, 10, ..., 500.
* Source hashes in metadata agree with the code and requirements used.
* PNG decoded successfully, is 2400 × 2080, and is nonblank. This was a
  **programmatic image check**, not a claim of manual visual inspection.
* No editor diagnostics were reported for the four Python source/test files.

## Actual independent-pool results

| Method | Calibration restricted mean (sheets) | Validation mean (sheets) | Conditional MCSE (sheets) |
|---|---:|---:|---:|
| x-bar/S | 370.43 | 408.87 | 25.90 |
| Directional Hotelling T² | 372.02 | 360.77 | 21.62 |
| Joint PCA T²/Q | 375.45 | 440.00 | 28.26 |
| One-class SVM | 370.59 | 468.67 | 28.82 |
| Upper loss CUSUM | 369.80 | 338.41 | 20.85 |
| Upper startup loss EWMA | 367.86 | 341.79 | 22.23 |

No simulated calibration, validation, or fault trajectory was censored in this
run. Validation means are therefore uncensored simulation estimates for these
paths, not proof of negligible population tails. MCSE is conditional on the
fitted baseline and empirical pools; it excludes baseline/pool uncertainty.
The calibration objective is discrete, so some targets cannot be hit exactly.

**These results do not establish matched in-control ARLs.** No acceptable ARL
tolerance was prespecified for a final study. Do not rank methods from the fault
table as if their false-alarm performance were equal. Obtain more independent
baseline/calibration information and repeated full experiments before doing so;
retain a fresh final test set after any design revision.

These are new toy-generator results, not the numerical table from the supplied
guide and not results from the existing deformation surrogate.

## Environment and integration notes

The editor dependency installer initially targeted the existing workspace virtual
environment and installed the requested QC libraries there. The same pinned
direct dependencies were subsequently installed explicitly into the starter's
separate virtual environment; the reported full run used that environment.
No existing RL source or checkpoints were edited and no RL training was launched.

The scalar surrogate does not supply verified signed XYZ. Its terminal reward
contract is not uniformly [0,1]. Consequently this work does not claim completed
physical-surrogate integration; see [DATA_CONTRACT.md](DATA_CONTRACT.md).

See [demo_output/run_metadata.json](demo_output/run_metadata.json),
[demo_output/calibration.csv](demo_output/calibration.csv), and
[demo_output/validation.csv](demo_output/validation.csv) for unrounded records.