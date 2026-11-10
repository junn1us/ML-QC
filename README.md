# Fixture-placement statistical quality control

Standalone Python experiments and editable LaTeX reports for deformation
monitoring motivated by fixture placement. **All included data are synthetic;
these results are not manufacturing measurements or validated Ansys outputs.**

## Start here

- **Latest report:** [journal manuscript PDF](iise_report/IISE_QC_PhaseI_PhaseII.pdf)
  and [editable LaTeX](iise_report/IISE_QC_PhaseI_PhaseII.tex).
  Four main-text pages, one references page, five visual-appendix pages.
- **Study design and results:** [Phase I / Phase II guide](iise_report/README.md).
- **All six exploratory charts:** [illustrated report](report/QC_Control_Charts_Report.pdf)
  and [figure guide](report/README.md).
- **Detailed APIs and statistical definitions:** [implementation guide](IMPLEMENTATION_GUIDE.md).
- **Real-data integration:** [data contract](DATA_CONTRACT.md).
- **Packaging and checks:** [repository validation record](REPOSITORY_VALIDATION.md).

## Two experiments, kept distinct

| Study | Methods | Interpretation |
|---|---|---|
| Exploratory | Paired x-bar/S, directional Hotelling T², joint PCA T²/Q, one-class SVM, CUSUM, EWMA | Six alternatives; realized ARLs were **not matched**. |
| Fresh Phase I / II | Joint PCA T²/Q versus a classical individual Shewhart maximum-displacement chart | Independent fitting, calibration and evaluation; verified ARL matching **within a prespecified ±10% tolerance**, not exact equality. |

The fresh study uses 5,000 fitting sheets, 2 million calibration sheets, 2 million
in-control validation sheets, and 200,000 sheets for each of six fault conditions.
Limits are frozen Phase I held-out order statistics, targeting ARL₀ = 370.

| Method | Realized conditional ARL₀ | Marginal 95% interval |
|---|---:|---:|
| PCA T²/Q | 374.9 | 365.0–385.1 |
| Shewhart maximum | 364.4 | 354.9–374.2 |

ARLs use the IID memoryless relation ARL = 1/p, with transformed exact binomial
intervals. Uncertainty is conditional on one baseline, not repeated-baseline
uncertainty. There is no uniform detection winner. See the report for fault sizes,
matching intervals, sensing differences, and limitations. The visual appendix
reproduces the earlier exploratory results and does not relabel them as matched.

## Setup and tests

Run commands from this repository's root. Tested with **Python 3.14.2** and the
pinned direct dependencies; Python 3.12 was not tested. No environment is bundled.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.mplconfig"
python -m unittest -v test_qc_monitoring.py test_demo.py test_matched_phase_study.py
```

On Windows, use an installed Python 3.14 interpreter to create the environment,
activate with `.venv\Scripts\Activate.ps1`, and set the variables with PowerShell
`$env:NAME = 'value'` syntax. Use [requirements.txt](requirements.txt) for numerical
work only; [requirements-dev.txt](requirements-dev.txt) additionally supplies
`pypdf` for report auditing. Transitive dependencies are not fully locked, so
bitwise numerical reproduction across platforms is not guaranteed.

## Build and verify the reports

Install [Tectonic](https://tectonic-typesetting.github.io/) separately and make
`tectonic` available on PATH (tested version: 0.17.0). On macOS, Homebrew users can
install it with `brew install tectonic`. Its first build needs internet access to
download TeX support assets. The publisher class, bibliography style, figures,
tables and bibliography are already included; no numerical rerun is necessary.

```bash
tectonic --keep-logs --keep-intermediates iise_report/IISE_QC_PhaseI_PhaseII.tex
python verify_iise_report.py
tectonic --keep-logs --keep-intermediates report/QC_Control_Charts_Report.tex
```

The verifier checks recorded code/plan hashes, alarm counts, independent exact
interval calculations, matching, template authenticity, PDF numbers and the
4+1+5 page structure. Build first: the verifier requires the generated LaTeX log,
which Git deliberately ignores. After scientific or layout edits, a failed
verification requires review, not silently relaxing the checks.

For VS Code, open the repository folder, open the main LaTeX source, and use
LaTeX Workshop's **View LaTeX PDF file** command to view the source and PDF side
by side. Compilation above does not depend on the extension's default recipe.

## Reproduce experiments (optional)

Included results are sufficient for reading and auditing. Full regeneration is
more expensive than the tests and **was not repeated merely to package this repository**.
Always choose new output directories; scripts reject existing experiment outputs.

```bash
python demo.py --out reproduction_output
python matched_phase_study.py --out iise_report/reproduction_results
```

Do not overwrite the preserved [exploratory outputs](demo_output/run_metadata.json)
or [matched results](iise_report/results/study_metadata.json). The matched run
generates 5,205,000 profiles in chunks instead of storing all raw data. Seeds,
draw order and chunk size are part of the reproducibility contract.

To regenerate report assets from the **preserved** outputs, without refitting:

```bash
python visualize_report.py
python build_iise_assets.py
```

These commands overwrite generated figures/tables and their manifests. The
matched renderer reads the original results, not the reproduction directory.
Rebuild both reports afterward. If figure versions change, explicitly synchronize
the appendix copies and review verification; changing figures does not update
manuscript prose. Changing an audited scientific source invalidates its historical
hash: preserve provenance instead of presenting a modified run as the original.

## Repository contents

| Path | Purpose |
|---|---|
| [qc_monitoring.py](qc_monitoring.py) | Reusable statistics, frozen models, calibration and censoring-aware summaries |
| [demo.py](demo.py) | Six-chart workflow and strict external NPZ adapter |
| [matched_phase_study.py](matched_phase_study.py) | Streaming, leakage-free matched comparison |
| [test_qc_monitoring.py](test_qc_monitoring.py), [test_demo.py](test_demo.py), [test_matched_phase_study.py](test_matched_phase_study.py) | 40 regression tests |
| [visualize_report.py](visualize_report.py), [build_iise_assets.py](build_iise_assets.py) | Saved-data figure/table renderers |
| [verify_iise_report.py](verify_iise_report.py) | Numerical, provenance and PDF audit |
| [demo_output](demo_output) | Saved exploratory results and sample displacement trace |
| [report](report) | Original illustrated report source, PDF and figures |
| [iise_report](iise_report) | Latest source, PDF, appendix, plan, frozen model, result tables and template |
| [VALIDATION.md](VALIDATION.md) | Historical exploratory validation and original environment notes |

Environments, caches, platform-specific compiler binaries, unrelated RL training
code/checkpoints, solver integrations and redundant source ZIPs are excluded.
The original parent project is not needed to run tests or compile the reports.
External-data validation cannot establish physical units or statistical independence;
those remain the data owner's responsibility.

## Attribution and publication

See [third-party notices](THIRD_PARTY_NOTICES.md) and
[template provenance](iise_report/template/PROVENANCE.md). No blanket license has
been assigned: choose appropriate terms before public distribution. Current
IISE-specific submission requirements still need confirmation. Code and manuscript
were prepared with AI assistance and need independent scientific review.

This delivery initializes a **local Git repository only**, with files staged for
review; no commit, remote, or GitHub publication is created automatically. The
companion source ZIP excludes Git metadata. Review licensing and contents before
making an initial commit or publishing to a remote.# Fixture-placement control-chart starter

**Freeze the process → extract sheet features → fit → calibrate the complete
alarm rule → evaluate independently.**

This is a new implementation of the supplied guide, checked against the uploaded
21-page November 2 mathematical framework. The starter ZIP was not available.
The synthetic generator is therefore newly implemented: **do not expect the
guide's numerical results to be reproduced**. Neither its toy data nor the
generated results are manufacturing measurements or Ansys validation.

## Run

From this directory, using a fresh virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_qc_monitoring.py test_demo.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python demo.py --out demo_output
```

On Windows, activate with `.venv\Scripts\Activate.ps1`; set threading environment
variables using PowerShell syntax. These setup commands target Python 3.12 as in
the guide. **Local execution was tested on Python 3.14.2, not Python 3.12**;
3.12 was not installed in the standard locations. The local `.venv` is 3.14.2.
The pinned direct dependencies in [requirements.txt](requirements.txt) are the
versions used locally. Saved metadata records the actual interpreter and versions.
Transitive dependencies are not fully locked; use an environment lock for archival
reproduction across machines. The PDF reader used during development is not a
runtime dependency.

Use a **new output directory for each experiment**; existing outputs are rejected.
To use the workspace proposal's target of 200 rather than the guide's illustrative
370 sheets, pass `--target-sheets 200` before calibration. This changes the
experiment, not a labeling convention. Default sizes: 1,500 fitting sheets,
20,000 each calibration/validation pool, 256 replications, 4,000-sheet horizon.
Options `--fit-size`, `--pool-size`, `--replications`, `--horizon`, `--seed`, and
`--target-sheets` support other prespecified designs. Subgroup size is five;
synthetic fitting/pool sizes and horizons must be divisible by five.

## Files and implemented methods

| File | Purpose |
|---|---|
| [qc_monitoring.py](qc_monitoring.py) | Statistics, frozen estimators, memory scores, calibration, censoring-aware summaries |
| [demo.py](demo.py) | Six-method synthetic workflow and strict external NPZ adapter |
| [test_qc_monitoring.py](test_qc_monitoring.py) | 29 deterministic mathematical/statistical tests |
| [test_demo.py](test_demo.py) | Seven adapter, data isolation, sensing, and workflow tests |
| [DATA_CONTRACT.md](DATA_CONTRACT.md) | Data requirements and current surrogate integration limitations |
| [VALIDATION.md](VALIDATION.md) | Executed checks, actual default-run results and environment notes |

| Framework | Entry point | Evaluated alarm |
|---|---|---|
| §3 x-bar/S | `XbarS.fit`, `statistics`, `joint_score`, `moment_limits`, `probability_limits` | Mean OR SD leaves jointly calibrated limits |
| §4 directional T² | `Hotelling.fit`, `score`, `analytic_limit` | Upper T² at fixed node 44 (zero-based) |
| §5 PCA | `PCAMonitor.fit`, `scores`, `joint_score` | T² OR Q above jointly calibrated limits |
| §6 SVM | `fit_models`, `score_pool` in demo | Negative decision function above calibrated limit |
| §7 loss | `loss_from_reward`, `bounded_loss` | Fixed input definition, not an alarm itself |
| §8 CUSUM | `cusum_table`, `cusum_score_paths` | Upper CUSUM > h, k=0.5 |
| §9 EWMA | `ewma_table`, `ewma_score_paths` | Upper startup-standardized score > L, lambda=0.1 |
| §10 extensions | `ma_score`, `mewma_score` | Building blocks, not in six-method comparison |
| §11 evaluation | `upper_limit`, `calibrate_run_length`, `first_alarm`, `run_length_summary`, `fault_delay_summary` | First strict exceedance |

All six rules are **alternatives**, not a six-chart production union. Calibrate
any larger union independently. Eight plot panels show the two components of
each paired x-bar/S and PCA rule.

### Statistical definitions

* `extract_features(U, critical_node=44)` returns max absolute Uz per sheet,
  signed XYZ at one fixed node, and signed Uz profiles. Never select a different
  critical node per sheet. Never treat spatial sites as independent sheets.
* `subgroup(d, n=5)` rejects incomplete groups. It cannot infer regime changes:
  partition by frozen layout, material, loading, policy, and reward version first.
* Xbar/S uses within-group sample SD (`ddof=1`) and estimates individual-sheet
  sigma by mean(S)/c4. The subgroup mean alone uses sigma/sqrt(n). Normal-model
  probability limits are references, not exact fitted-parameter false-alarm rates.
* `Hotelling(mean, covariance)` uses known-parameter chi-square limits.
  `Hotelling.fit(x)` records the baseline count: phase 1 gives the marginal
  retrospective beta limit, phase 2 the independent predictive F limit. The
  implementation solves a Cholesky system and rejects ill-conditioned covariance.
* PCA standardizes with training sample SDs, removes training-constant coordinates,
  and freezes that mask. Changes confined to removed coordinates are invisible.
  `standardize=False` preserves raw-coordinate weighting. T² sums squared retained
  scores divided by sample eigenvalues; Q is residual squared Euclidean norm.
  Their joint score is `max(T2/r, Q/q_scale)`, with training mean Q as `q_scale`.
* SVM fits only whitened retained scores, RBF kernel, nu=.05, gamma=1/r. Nu is not
  a future false-alarm probability. Negating `decision_function` makes high scores
  anomalous; the native zero boundary is not the calibrated threshold.
* Synthetic loss is `clip(maxabsUz/3,0,1)`. **3 mm is not an acceptance tolerance.**
  Reward mode is `1-reward` for a fixed terminal reward in [0,1]. Out-of-range
  rewards are rejected, not silently rescaled using future data. Clipping loses
  sensitivity; metadata records loss saturation in held-out pools.
* CUSUM has zero initial state and zero reset boundary, not a zero process mean.
  Its table includes the lower side and consecutive-positive run counts for
  diagnostics, but only the upper side is calibrated. EWMA starts at baseline mu
  and uses the time-dependent startup variance under independent inputs.

## Fitting, calibration, and evaluation

`fit_models` fits all preprocessing, moments, PCA, and SVM only on fitting sheets.
`score_pool` evaluates without refitting. Each `iter_chart_paths` pass reuses the
same sheet indices across methods, preserving paired comparisons and all
within-sheet correlations. Nonoverlapping groups are formed along each path,
never across replication boundaries.

The demo fixes hyperparameters in advance. If you tune them, reserve a separate
tuning split and recalibrate the selected pipeline. The default NumPy seed is
20261106, with distinct SeedSequence children for fitting, calibration pools and
paths, validation pools and paths, each fault, and the separate illustrated trace.

`calibrate_run_length` searches a monotone, finite-sample step function for the
closest restricted mean `mean(min(tau,H))` to the target in **sheets**. A subgroup
alarm is charged at the end of its five sheets. Excessive calibration censoring
(>1% by default) raises an error. Increase H, not just replications, when tails
are truncated. The search is not a proof of a population ARL and may not hit the
target exactly.

`upper_limit` implements the finite-rank held-out threshold with strict
exceedance and rejects insufficient samples. Its exchangeability statement is
marginal over calibration samples, not a conditional ARL guarantee. Never use
a one-step quantile as sequential calibration for memory charts.

Independent validation reports restricted mean, conditional MCSE, and censoring.
`arl_sheets` is absent/blank (`None` in Python) if any run is censored. An alarm
at the last update is an observed event; no alarm is represented internally by
H+1 but capped at H for restricted means. Zero observed censoring still does not
prove a negligible population tail.

**Inspect validation before interpreting fault comparisons.** Predeclare an ARL
tolerance and precision. Repeat complete fit/calibrate/test experiments across
independent baselines for final inference. More bootstrapped paths alone do not
remove baseline or finite-pool uncertainty. If validation informs a redesign,
obtain a fresh final test set. No automatic method ranking or matched-ARL claim
is made here.

Fault comparisons use three toy families at coefficients 0.5, 1, 2: broad Uz
shift, Ux shift, and local Uz distortion. Coefficients are not standardized score
shifts. All changes begin at sheet one (zero-state). A long-horizon detection
probability near one includes ordinary false alarms and is not sufficient evidence
of sensitivity. Directional T², profile methods, and scalar methods see different
information; only directional T² can see a Ux-only change in this setup.

For later changes, concatenate the in-control prefix and faulty suffix within
each path and compute the memory score **once**. Pass the count of pre-change
updates as `change_index` to `fault_delay_summary`. It reports early alarms and
conditional delay/detection among survivors. No survivors yields `None`, not
zero. For x-bar/S, align the change to a subgroup boundary. Delays with censoring
are restricted, not uncensored means.

## Outputs

**New Phase I/II study:** [iise_report/README.md](iise_report/README.md)
documents a fresh tolerance-matched PCA-versus-Shewhart comparison and a
[four-page journal manuscript](iise_report/IISE_QC_PhaseI_PhaseII.pdf), with
references on an additional page. It does not replace or relabel the earlier
unmatched six-method results below.

For the illustrated LaTeX report and individual figure downloads, see
[report/README.md](report/README.md) and
[report/QC_Control_Charts_Report.pdf](report/QC_Control_Charts_Report.pdf).
The reproducible [visualize_report.py](visualize_report.py) renders saved results
without refitting or recalibrating charts.

| Output | Interpretation |
|---|---|
| [demo_output/run_metadata.json](demo_output/run_metadata.json) | Actual runtime, seeds, split sizes, source hashes, fitted transforms/moments, thresholds, units and caveats |
| [demo_output/calibration.csv](demo_output/calibration.csv) | Restricted-mean calibration diagnostics and update intervals |
| [demo_output/validation.csv](demo_output/validation.csv) | Independent in-control checks; read before fault results |
| [demo_output/fault_comparison.csv](demo_output/fault_comparison.csv) | 54 method/scenario rows, zero-state delay, detection, censoring |
| [demo_output/sheet_trace.csv](demo_output/sheet_trace.csv) | Per-sheet features, illustrative reward, loss, scores and alarms |
| [demo_output/subgroup_trace.csv](demo_output/subgroup_trace.csv) | Mean, sample SD and limits at subgroup ending sheets |
| [demo_output/cusum_table.csv](demo_output/cusum_table.csv), [demo_output/ewma_table.csv](demo_output/ewma_table.csv) | Recurrences and diagnostic columns |
| [demo_output/example_displacements.npz](demo_output/example_displacements.npz) | 500 × 100 × 3 toy displacement trace and normalized planar coordinates |
| [demo_output/control_charts_demo.png](demo_output/control_charts_demo.png) | Eight panels with change at sheet 201 and first-alarm markers |

The 500-sheet plot continues after alarms without reset, solely for illustration.
Its change after 200 sheets is separate from zero-state fault comparisons.
Metadata is not a deployment-ready serialized SVM/model artifact. Before live use,
persist immutable transforms/model identifiers, feature order, limits, and an
approved investigation/reset policy. Batch helpers intentionally restart; retain
state across live sheets and buffer partial subgroups. Freeze estimation during
each run and stop/investigate at its first alarm.

## Additional course-method recipes (not validated demo methods)

* R/x-bar–R: `np.ptp(groups, axis=1)` and normal-range constants d2(n), d3(n) for
  the actual n. Sigma estimate mean(R)/d2; mean limits mu ± L sigma/sqrt(n),
  R limits max(0,(d2−L d3)sigma) and (d2+L d3)sigma. Calibrate the paired OR.
* I–MR: absolute consecutive differences, sigma=mean(MR)/1.128 for independent
  normal pairs, n=2 factors. Overlap induces dependence; recalibrate full paths.
* p/np: distinct nonconforming sheets under a fixed verified acceptance criterion;
  pooled p0=sum(D)/sum(n). Use `scipy.stats.binom`, especially with sparse counts;
  np requires fixed n. Specify integer rejection and verify each tail probability.
* c/u: genuine nonconformity counts and fixed exposure for c, variable positive
  exposure for u; u0=sum(C)/sum(exposure). Use `scipy.stats.poisson`, validate
  dispersion, and check discrete tails. Correlated spatial nodes are not trials.
* k-means: fit on training features; minimum `KMeans.transform` distance on new
  data. Freeze Euclidean versus squared-distance definition before calibration.
* k-NN: fit `NearestNeighbors(n_neighbors=k)` on training features and use the
  kth held-out query distance; don't calibrate with self-neighbors.
* Isolation forest: negate `score_samples` so high means anomalous, then calibrate
  separately. `contamination` is not a guaranteed future alarm rate.

All new scores need frozen feature transforms and held-out calibration. MA and
MEWMA are already implemented and tested but require their own full-path limits.

## References and development record

Primary mathematical source: the user-supplied November 2, 2026 framework,
sections 3–11 and appendices A/B. The earlier workspace
[proposal](../ISYE_6960_QC_Project_Proposal.tex) is a narrower profile study with
ARL target 200; it does not establish that the six-method XYZ data exist.

Official APIs: [leakage prevention](https://scikit-learn.org/stable/common_pitfalls.html),
[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html),
[OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html),
[SciPy distributions](https://docs.scipy.org/doc/scipy/reference/stats.html).

AI-use note: code, tests, and documentation were generated with AI assistance.
Automated consistency tests support the mathematics, not surrogate physical
validity. Review scientific choices before using this for the final study.