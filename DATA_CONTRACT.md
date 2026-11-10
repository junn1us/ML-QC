# Project data contract and integration boundary

## One observation

Provide an NPZ array named `U` with shape `(N, G, 3)`, containing finite signed
Ux, Uy, Uz in **millimetres**, at fixed ordered measurement locations. The
candidate fixture grid is not automatically the measurement grid. Resolve the
displacement frame, datum, signs, units, and 2000 × 2000 × 2 mm geometry before
exporting any observations.

Optional arrays:

* `reward`: shape `(N,)`, fixed terminal reward in [0,1], not an accumulated
  training return. Must be present in **all three splits or none**.
* `measurement_coordinates`: `(G,2)` or `(G,3)`, identical values and ordering in
  all splits. Coordinate units must be recorded externally. If supplied in any
  split, supply in all. Absence is flagged in metadata, not treated as validation.

Keep a separate row-aligned sheet table with sheet ID, time/seed, material/loading
regime, raw fixture-layout ID, policy/checkpoint hash, reward version, simulation
version, noise/input parameters and terminal reward. Save immutable provenance
under the CLI `--data-id`. Fault labels, magnitudes and change times belong to
evaluation metadata, never detector inputs. The loader does **not** validate the
sheet table, physical units, provenance declarations, or independence of sheets.

The CLI uses one fixed regime, node 44, subgroup size five and three PCs. If those
choices do not match the frozen study, change the adapter **before** calibration,
or call `fit_models` with explicit arguments. Remove redundant vector coordinates
for a custom Hotelling model; do not silently regularize and reuse its old limits.

## External workflow

Prepare independent fit, calibration and in-control test archives. This is an
interface example; no project archives are supplied:

```bash
python demo.py --fit-file baseline_fit.npz \
  --cal-file baseline_calibration.npz --test-file baseline_test.npz \
  --data-id frozen-layout-policy-reward-experiment-version \
  --target-sheets 370 --out external_output
```

The loader uses `allow_pickle=False`, validates finite values, schema, counts,
reward consistency, grid equality when coordinates are supplied, and rejects
identical paths or file hashes. These checks **cannot prove disjointness**:
overlapping rows or differently packaged copies must be excluded by the data
owner. Use disjoint batches/runs/seeds. IID bootstrap is only defensible for
independent comparable sheets; dependent production data need repeated independent
trajectories, a time-series model, or a validated block design.

External mode writes only calibration, validation, and metadata. It deliberately
does not manufacture a fault comparison or a synthetic actual-reward trace.
`fit_models(..., reward=fit_reward)` estimates loss moments from actual rewards;
`score_pool(..., reward=new_reward)` requires the same convention for every new
pool. In contrast to manually replacing two moments, this interface prevents
accidentally reverting to deformation-only loss on later observations.

For physical fault studies, build each shared sheet pool through the same verified
surrogate/sensor/reward pipeline, then pass its scores and shared resampling
indices to `iter_chart_paths`. Use the same frozen thresholds for all fault sizes.
Keep all labels outside scoring. `save_trace` is explicitly synthetic-only and
rejects models trained with actual rewards.

## Historical integration boundary (external project not bundled)

The original parent workspace's `FixtureCore` implementation, which is not a
dependency of this standalone repository,
stores planar sensor coordinates in `coords` (metres) and `disp` of shape `(G,2)`.
The first displacement column is zero; the second is a nonnegative scalar sag
surrogate in millimetres. This is **not a measured signed XYZ field**. The kernel
is a heuristic gravity/suppression field, not a three-dimensional FE solution.
Anchor ordering and sensor ordering differ; `_anchor_node_idx` maps between them.

Do not append two noise columns or zero-pad sag and present the result as
directional evidence. For an explicitly scalar-only study, `PCAMonitor.fit`
already accepts `(N,G)` profiles and `XbarS.fit` accepts per-sheet maxima; define
and label the transverse sign convention, and design a separate scalar/profile
workflow. Its Hotelling comparator could use nonredundant summaries, but would
be a different detector from the XYZ chart here and must be calibrated separately.

The external project's reward logic
supports several definitions; lower clipping is not guaranteed. A last-step reward
may differ from accumulated return or a fixed terminal quality score. Select a
documented terminal definition/transform before fitting; don't silently clip RL
rewards to force compatibility.

Changing only a random seed does not make the deterministic core produce random
physical sheets. Reconstruct the surrogate as needed for independently sampled,
documented load/thickness/material parameters, since fields are precomputed in
its constructor. Support-strength and influence-width changes exist, but local
partial fixture degradation, position jitter, and measurement-noise distributions
need explicit modeling/validation. Retain the gravity-feedback gate and audit its
behavior under fixture-loss faults; zeroing gravity is not a physical response to
losing support.

The external Ansys adapter offers solver displacement access, but its model,
node IDs/order, coordinate-system alignment, units, initial displacement and
mapping to fixed observation sites must be verified before it supplies this
contract. The existence of that interface is not solver validation.

## Before claiming project results

1. Audit checkpoint compatibility and the raw chosen layout; disclose any heuristic
   fixture completion. Freeze layout and policy; do not rerun policy selection for
   every monitored sheet.
2. Define independent sheet-level inputs/noise and coherent physical faults.
3. Verify units, frame, fixed nodes, terminal reward and observation-order mapping.
4. Fix chart choices, feature masks, loss definitions, sampling and reset policy.
5. Obtain independent fit/tuning/calibration/evaluation data; preserve a fresh test
   set whenever validation causes a redesign.
6. Predeclare acceptable ARL tolerance/precision, repeat baseline fits, report
   censoring, and use information-matched ablations before algorithm rankings.
7. Serialize and version a deployable fitted pipeline separately. On an alarm,
   retain the raw record and investigate; don't silently retrain or delete points.

No RL training code, historical checkpoints, or project measurements are changed
by this starter implementation.