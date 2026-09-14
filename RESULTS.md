# First completed baseline — 2026-09-11

The unforced prototype runs end to end on the Mac. No cloud resources were created.

## Completed runs

Both cases use `R=2`, `Re_jet=100`, `Sc=1`, exponential inlet `delta99/D=0.75`,
a pipe depth of `2D`, and final time `t U_inf / D=4`.

| Quantity | Smoke | Baseline |
|---|---:|---:|
| Cells per diameter | 4 | 8 |
| Active fluid cells | 10,336 | 82,752 |
| Timesteps | 240 | 591 |
| Measured simulation runtime | 7.35 s | 169.38 s |
| Discrete pipe area / D² | 0.7500 | 0.8125 |
| Maximum sampled divergence | 1.58e-10 | 1.08e-9 |
| Maximum sampled scalar mass-balance error | 6.84e-14 | 1.78e-13 |
| Maximum sampled relative volume imbalance | 1.22e-12 | 2.75e-12 |
| Final scalar centroid y/D at x/D=2, interpolated | 1.6570 | 1.9556 |
| Final scalar spanwise standard deviation / D at x/D=2, interpolated | 0.6074 | 0.5922 |
| Final upstream reversed-shear area / D² | 0 | 0 |

Runtime is wall time reported by each solver run, excludes figure generation, and was not measured under
identical background machine load. It is useful for planning, not a controlled scaling benchmark.
The original smoke run completed September 8; it was inspected and rendered during this continuation.

The centroid values above interpolate the saved profiles to the **same physical x=2D**.
The history CSV samples its nearest cell center (1.875D on the smoke grid and 1.9375D on the baseline grid).
Likewise the upstream circulation boxes have mesh-dependent snapped bounds. Compare their plots as sensitivity
diagnostics, not as strict same-contour convergence measurements.

The centroid height differs by about 18% between the two grids. Geometry error, scalar numerical diffusion,
and flow-discretization error all contribute; this is not a converged physical prediction.
The reference circular pipe area is pi/4 = 0.785398 D². Fixed discrete bulk speed with different pipe areas
also changes volume flow between the grids.

There is a downstream recirculation region in the saved fields. Neither final run has reversed streamwise
wall shear in the diagnostic upstream strip (`-2 < x/D < -0.5`). These observations do not establish the
presence or absence of a coherent horseshoe system at a converged resolution or another flow regime.

## Verification

`13 passed` from `tests/test_numerics.py`:

- Projection removes arbitrary divergence while preserving prescribed normal boundary velocities.
- Known pressure gradients are recovered by the discrete projection.
- Linear near-wall shear and its no-slip ghost value match their analytical values.
- Rigid rotation and pure strain yield the expected velocity gradients, signed vorticity and Q.
- Manufactured nonlinear momentum and viscous shear-mode spatial errors decrease approximately quadratically.
- Scalar transport closes its global budget and stays bounded; symmetric forcing preserves spanwise symmetry.
- Checkpoint continuation matches uninterrupted stepping, and incompatible restarts are rejected.
- Invalid case parameters are rejected.

An additional CLI restart from the baseline checkpoint advanced from t=4 to t=4.1 in 4.53 seconds,
with final divergence 4.19e-12 and scalar balance error 1.87e-13.

These checks verify parts of the numerical implementation. Experimental validation, full pressure/time
convergence, domain independence, statistically stationary means and local vorticity budgets remain open.
The 16-cells/D case is supplied but was not run.

## Open the results

- [Baseline time-slider report](results/baseline/report.html)
- [Resolution comparison](results/comparison/comparison.png)
- [Baseline manifest](results/baseline/manifest.json)
- [Restart manifest](results/restart-check/manifest.json)
- [Run instructions and scientific limitations](README.md)

The next useful calculation is a controlled refinement study with consistent geometry/flux and diagnostic
locations, followed by longer averaging. The current baseline takes under three minutes locally; extra compute
is not required for running or inspecting this version.
