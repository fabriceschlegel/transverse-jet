# Compute decision

**Status: PETSc/GAMG MPI PATH IMPLEMENTED — 2026-09-15**

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

Profiling showed pressure at 96% of baseline runtime, so PETSc through `petsc4py`
and OpenMPI is now implemented as the optional `petsc-gamg` backend. The pressure
matrix and vectors are distributed; explicit fields remain replicated. On the
662,016-active-cell refined pilot, four ranks reduced pressure time from 62.32 s
to 7.22 s and total runtime from 68.22 s to 14.15 s. Eight ranks were slower
overall, so four is the measured MacBook default. A fully domain-decomposed or
threaded geometric multigrid implementation remains a future scaling option.

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
