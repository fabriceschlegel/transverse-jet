# Compute decision

**Status: DECIDED — 2026-09-15**

This project will keep the existing CPU NumPy/SciPy MAC-grid solver as the
numerical reference while developing two separate execution paths.

## MacBook Pro M4 path

The first multicore implementation will use Numba `njit(parallel=True)` for
the explicit momentum, passive-scalar, divergence, gradient, and diagnostic
stencils. The current PyAMG-preconditioned SciPy CG pressure projection will
remain in place for the first benchmark.

Before and after that change, timings will separate:

1. momentum and viscous terms;
2. scalar transport;
3. the three pressure projections per SSP-RK3 timestep;
4. diagnostics and output.

PETSc through `petsc4py` and MPI is the next pressure-solver option only if
profiling shows that pressure dominates after the stencil parallelization.
Adopting PETSc means distributing the pressure vectors and matrix and measuring
MPI overhead; merely installing PETSc will not make the current Python process
multicore. A threaded geometric multigrid implementation remains an alternative
for this structured grid.

Independent cases and refinement studies may be run concurrently immediately,
with one process and one BLAS thread per case.

## AWS NVIDIA GPU path

The selected GPU solver is **nekRS 26.0 with the CUDA backend**. The first trial
will use one `g6.2xlarge` instance and one MPI rank for its single NVIDIA L4 GPU.
The existing `cloud/aws/user-data-nekrs.sh` script builds nekRS and runs its
bundled channel case as the installation smoke test.

After that smoke test passes, the transverse-jet case will use a body-fitted
three-dimensional hexahedral mesh, including the pipe, nozzle lip, plate, and
upstream boundary layer. Gmsh plus `gmsh2nek` is the initial meshing route.
nekRS will solve incompressible Navier-Stokes and passive-scalar transport.

## Comparison rule

The MAC and nekRS calculations will use matching nondimensional geometry,
velocity ratio, Reynolds number, Schmidt number, boundary conditions, sampling
times, and diagnostic locations. Agreement will be evaluated in physical
quantities and convergence trends rather than cell-by-cell fields because the
MAC solver uses a staircase finite-difference geometry while nekRS uses a
body-fitted high-order spectral-element discretization.

Actuation remains outside the current scope. The unforced pipe/plate/jet case
and horseshoe-region diagnostics must be established first.
