# Unforced transverse jet

A runnable 3-D incompressible jet-in-crossflow research prototype for Fabrice Schlegel.
The initial scope is the **unforced** pipe/plate/jet interaction and upstream horseshoe-region diagnostics.
There is no actuator or controller in this version.

## Run locally

Create a project-local environment with Python 3.10 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

Then, from this directory:

```sh
./run.sh run --config cases/smoke.json --output results/new-smoke
./run.sh report results/new-smoke
./run.sh run --config cases/baseline.json --output results/new-baseline
./run.sh report results/new-baseline
./run.sh compare results/new-smoke results/new-baseline --output results/new-comparison
```

Open the resulting `report.html` in a browser. Its slider moves through saved physical times;
the color scales are fixed across frames. No server, account, or internet access is required.
Output folders must be empty to prevent accidental overwriting.

Run numerical tests:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q
```

## Physics and units

- Diameter `D = 1`, crossflow speed `U_inf = 1`, density `rho = 1`.
- `x` is downstream, `y` is upward from the plate, `z` is spanwise.
- The pipe is centered at `x=z=0`, with radius `0.5`, and extends below `y=0`.
- `velocity_ratio = U_jet_bulk / U_inf`; `re_jet = U_jet_bulk D / nu`, so `nu=velocity_ratio/re_jet`.
- Default `R=2`, `Re_jet=100`, `Re_crossflow=50`, `Sc=1`: a deliberately modest test case,
  **not a reconstruction of the original MIT case and not a turbulent DNS**.
- Inlet crossflow profile is `u=1-exp(-ln(100)*y/delta99)` above the plate.
  `delta99` is prescribed at the **inlet**, not measured at the nozzle; this is not a Blasius profile.
- The pipe inlet profile is parabolic, discretely normalized to preserve the requested bulk velocity.
  The exit profile evolves inside the explicitly included pipe.
- Plate and pipe walls: impermeable and no-slip via staggered wall faces and reflected tangential ghosts.
- Top and spanwise sides: impermeable free-slip boundaries. They model finite confinement, not an exact unbounded domain.
- Outlet: zero pressure on the outlet plane and extrapolated predictor velocity. Scalar backflow is ambient (`c=0`).
- Scalar: pipe reservoir `c=1`, crossflow reservoir `c=0`, zero wall diffusive flux.

## Discretization

Velocities lie on MAC faces and pressure/scalar at cell centers. The pressure matrix is assembled as
`-D G` on the connected fluid-cell graph; the matching face gradient enforces discrete incompressibility.
CG with smoothed-aggregation multigrid solves pressure. A failed pressure solve or nonfinite/divergent state stops the run.

Momentum uses centered second-order spatial derivatives and interpolated advecting velocities.
This advective form is **not a discrete kinetic-energy-conserving scheme**. SSP-RK3 stages are each
projected. This does **not** establish third-order accuracy for the complete open-boundary pressure/velocity scheme;
the nonincremental pressure treatment and boundary splitting require their own convergence assessment.

Scalar transport uses conservative first-order donor-cell advection and centered diffusion. It is never
clipped to conceal overshoots. Its integrated mass is checked against the RK-weighted boundary flux.
Numerical scalar diffusion prevents quantitative claims about molecular mixing on these coarse grids.

The circular pipe is a **staircase approximation**. At 4 cells/D its area is 0.75 versus pi/4 analytically.
Changing resolution changes the represented pipe area and therefore its volume flow at fixed bulk speed.
This geometric error is recorded in each manifest and must converge before physical comparisons.

## Saved data and diagnostics

- `config.json`: exact input parameters.
- `manifest.json`: completion status, source digest, runtime, grid/pipe area, solver backend, numerical checks.
- `history.csv`: flux balance, divergence, scalar budget/bounds, energy, wall reversal, circulation, centroid and width.
- `snapshots/state_*.npz`, `final.npz`: staggered velocities, pressure, scalar, centered velocity, signed vorticity,
  Q, wall shear, coordinates, geometry masks, and restart bookkeeping.
- `mean.npz`: physical-time-weighted averages over the configured window. Averaging does not establish stationarity.
- `report.html`, `figures/`: time-slider report, centerplane fields, wall shear, a 3-D scalar cloud and histories.

Read an NPZ with `numpy.load(path, allow_pickle=False)`. Velocity array shape is `[component,x,y,z]`;
vorticity has the same convention. The derivative-valid mask excludes outer and solid-adjacent stencils.
Wall shear is `[streamwise/spanwise,x,z]`. Q uses `0.5*(||Omega||^2-||S||^2)`.

The upstream-box circulation is a **signed rectangular contour integral**, not an isolated horseshoe strength.
Its snapped contour bounds are in each snapshot. A background boundary layer contributes circulation even without
a coherent vortex. Negative streamwise wall shear indicates local reversal; it does not independently establish
three-dimensional horseshoe topology. The scalar cloud is concentration, not a vortex surface.

Restart into a **new output directory**:

```sh
./run.sh run --config cases/baseline.json --restart results/baseline/final.npz --end-time 8 --output results/baseline-continued
```

Geometry, physics and numerical settings must match the checkpoint. End time and output/averaging settings may change.
Statistics for the continuation cover its own time interval; restart does not silently merge old averages.

## Validation boundary and next scientific steps

Tests check the projection/boundary constraints, manufactured nonlinear momentum convergence, viscous shear-mode
spatial convergence including wall rows, rigid-rotation/strain diagnostics, wall shear, scalar conservation and bounds,
symmetry, and checkpoint continuation. These are numerical verification checks, not experimental validation.

The included 4-, 8- and 16-cells/D cases are refinement starting points. Before a horseshoe claim:

1. Recover or choose an explicit target flow regime and experimental geometry.
2. Converge pipe geometry, wall-normal spacing and lip resolution; then timestep and outer boundaries.
3. Run past startup and demonstrate statistically stable diagnostics over longer windows.
4. Compare signed vorticity, wall-shear topology, circulation and full 3-D structure with published data.
5. Add a closed local vorticity budget before attributing the mechanism to tilting, stretching or wall generation.

For substantially higher Reynolds numbers, use a validated body-fitted high-order DNS/LES solver such as nekRS.
This local finite-difference prototype is not that high-fidelity implementation and has no subgrid model.

## Compute

**Recorded decision (2026-09-15):** use Numba-threaded stencils first for a
multicore MacBook Pro M4 implementation, with PETSc/petsc4py plus MPI reserved
for the pressure projection if profiling justifies it. Use nekRS 26.0 with CUDA
for the AWS NVIDIA GPU implementation. See [COMPUTE_PLAN.md](COMPUTE_PLAN.md).

The code uses CPU NumPy/SciPy and a multigrid pressure preconditioner. It has no GPU or distributed MPI implementation.
More vCPUs do not automatically accelerate one run. Use the measured `elapsed_seconds` before renting capacity;
run independent cases concurrently if memory permits. Halving spacing multiplies allocated cell counts by about eight,
and increases the timestep count: expect substantially more than eight times the work.

The existing code is portable to a Linux CPU instance after installing its dependencies, but no AWS infrastructure
has been created. A one-instance benchmark is appropriate before any cluster or GPU purchase.

## References

- [Schlegel, Wee & Ghoniem (2008), fast particle/transport-element method](https://doi.org/10.1016/j.jcp.2008.03.036).
- [Schlegel et al. (2011), wall-boundary-layer contributions in transverse jets](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/contributions-of-the-wall-boundary-layer-to-the-formation-of-the-counterrotating-vortex-pair-in-transverse-jets/CFB372F97339DE7888FC9E8F2E936377).
- [Brown, Cortez & Minion (2001), projection-method accuracy](https://www.sciencedirect.com/science/article/pii/S0021999101967154).
- [Chauvat et al. (2020), pipe/exit-profile and stability effects](https://doi.org/10.1017/jfm.2020.85).
- [nekRS documentation](https://nekrs.readthedocs.io/en/latest/).

These sources motivate the research design; this custom solver is not a reproduction of their implementations.
