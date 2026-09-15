import csv
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import numba
from .diagnostics import summary, vortex_fields, wall_shear, trajectory, upstream_circulation


def source_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def save_snapshot(solver, path):
    g, s = solver.grid, solver.state
    centered = g.centers(s.velocity)
    omega, q, valid = vortex_fields(centered, g.h, g.fluid)
    tau, plate = wall_shear(g, centered)
    yc, zc, width, mass = trajectory(g, s.scalar)
    _, box = upstream_circulation(g, centered)
    np.savez_compressed(path, time=s.time, step=s.step,
                        x=g.x, y=g.y, z=g.z, fluid=g.fluid,
                        u=s.velocity[0], v=s.velocity[1], w=s.velocity[2],
                        velocity=centered, scalar=s.scalar, pressure=s.pressure,
                        vorticity=omega, Q=q, derivative_valid=valid,
                        wall_shear=tau, plate_mask=plate,
                        trajectory_y=yc, trajectory_z=zc, scalar_width_z=width,
                        scalar_section_integral=mass, circulation_box=box,
                        initial_scalar_mass=solver.initial_scalar_mass,
                        scalar_boundary_integral=s.scalar_boundary_integral,
                        config_json=json.dumps(vars(solver.config)))


def restore(solver, path):
    with np.load(path, allow_pickle=False) as a:
        old_config = json.loads(str(a["config_json"]))
        ignored = {"end_time", "average_start", "output_interval", "sample_interval"}
        for key, value in vars(solver.config).items():
            if key not in ignored and value != old_config[key]:
                raise ValueError(f"Restart configuration mismatch for {key}")
        solver.state.velocity = [a[name].copy() for name in ("u", "v", "w")]
        solver.state.scalar = a["scalar"].copy()
        solver.state.pressure = a["pressure"].copy()
        solver.state.time, solver.state.step = float(a["time"]), int(a["step"])
        solver.state.scalar_boundary_integral = float(a["scalar_boundary_integral"])
        solver.initial_scalar_mass = float(a["initial_scalar_mass"])
    if solver.state.time >= solver.config.end_time:
        raise ValueError("Restart time must precede end_time")


def run(solver, directory, restart=None):
    out = Path(directory)
    projection = solver.projection
    is_root = projection.rank == 0
    comm = getattr(projection, "PETSc", None)
    not_empty = out.exists() and any(out.iterdir())
    if comm is not None:
        # Ensure every worker inspects the pre-existing filesystem state before
        # rank zero is allowed to create the new output structure.
        comm.COMM_WORLD.barrier()
    if not_empty:
        # The shared-filesystem check is intentionally performed on every rank
        # so a root-only exception cannot strand workers at the barrier.
        raise FileExistsError(f"Output directory is not empty: {out}. Choose a new directory.")
    if is_root:
        out.mkdir(parents=True, exist_ok=True)
        (out / "snapshots").mkdir()
    if comm is not None:
        comm.COMM_WORLD.barrier()
    if restart:
        restore(solver, restart)
    c, g = solver.config, solver.grid
    if is_root:
        c.save(out / "config.json")
    manifest = {
        "status": "running", "fidelity": "unvalidated coarse research prototype; not DNS/LES evidence",
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "source_sha256": source_digest(), "pressure_backend": solver.projection.backend,
        "mpi_ranks": projection.mpi_size,
        "active_cells": int(g.fluid.sum()), "grid_shape": list(g.shape),
        "cells_per_diameter": 1 / c.h, "discrete_pipe_area": float(g.pipe_inlet.sum() * g.h**2),
        "analytic_pipe_area": float(np.pi / 4), "re_crossflow": 1 / c.nu,
        "restart": str(Path(restart).resolve()) if restart else None,
        "initial_time": solver.state.time,
        "inflow": "exponential velocity profile with prescribed delta99; not Blasius",
        "side_and_top": "impermeable free-slip; domain-confinement study required",
        "actuator": "none",
        "stencil_backend": "Numba parallel CPU",
        "numba": numba.__version__, "numba_threads": numba.get_num_threads(),
        "jit_warmup_seconds": solver.jit_warmup_seconds,
    }
    manifest_path = out / "manifest.json"
    if is_root:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.monotonic()
    stats = summary(solver)
    peak_div, peak_mass, peak_flux = stats["divergence_linf"], abs(stats["scalar_balance_error"]), stats["relative_volume_imbalance"]
    mean_u, mean_c = np.zeros((3, *g.shape)), np.zeros(g.shape)
    mean_duration = 0.0
    next_output = solver.state.time + c.output_interval
    next_sample = solver.state.time + c.sample_interval
    last_print = time.monotonic()
    if is_root:
        save_snapshot(solver, out / "snapshots" / f"state_{solver.state.step:07d}.npz")
    try:
        context = (out / "history.csv").open("w", newline="") if is_root else nullcontext()
        with context as f:
            writer = csv.DictWriter(f, fieldnames=list(stats)) if is_root else None
            if is_root:
                writer.writeheader(); writer.writerow(stats)
            while solver.state.time < c.end_time - 1e-12:
                old_time = solver.state.time
                # Synchronize snapshots and samples to requested physical times.
                dt = min(solver.timestep(), c.end_time - old_time, next_output - old_time, next_sample - old_time)
                old_u, old_c = g.centers(solver.state.velocity), solver.state.scalar
                solver.advance(dt)
                duration = max(0.0, solver.state.time - max(old_time, c.average_start))
                if duration:
                    # Time-weighted trapezoidal endpoint integration on this window.
                    fraction = (max(old_time, c.average_start) - old_time) / dt
                    new_u = g.centers(solver.state.velocity)
                    first_u = old_u + fraction * (new_u - old_u)
                    first_c = old_c + fraction * (solver.state.scalar - old_c)
                    mean_u += 0.5 * (first_u + new_u) * duration
                    mean_c += 0.5 * (first_c + solver.state.scalar) * duration
                    mean_duration += duration
                if solver.state.time >= next_sample - 1e-10 or solver.state.time >= c.end_time - 1e-10:
                    stats = summary(solver)
                    peak_div = max(peak_div, stats["divergence_linf"])
                    peak_mass = max(peak_mass, abs(stats["scalar_balance_error"]))
                    peak_flux = max(peak_flux, stats["relative_volume_imbalance"])
                    if is_root:
                        writer.writerow(stats); f.flush()
                    next_sample += c.sample_interval
                if solver.state.time >= next_output - 1e-10:
                    if is_root:
                        save_snapshot(solver, out / "snapshots" / f"state_{solver.state.step:07d}.npz")
                    next_output += c.output_interval
                if time.monotonic() - last_print > 10:
                    if is_root:
                        print(f"t={solver.state.time:.3f}/{c.end_time:g}, step={solver.state.step}, div={stats['divergence_linf']:.2e}", flush=True)
                    last_print = time.monotonic()
        if is_root:
            save_snapshot(solver, out / "final.npz")
        if mean_duration and is_root:
            np.savez_compressed(out / "mean.npz", velocity=mean_u / mean_duration,
                                scalar=mean_c / mean_duration, duration=mean_duration,
                                start=max(c.average_start, manifest["initial_time"]), end=solver.state.time,
                                x=g.x, y=g.y, z=g.z, fluid=g.fluid)
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - start,
                        final_time=solver.state.time, steps=solver.state.step,
                        sampled_max_divergence=peak_div, sampled_max_scalar_balance_error=peak_mass,
                        sampled_max_relative_volume_imbalance=peak_flux,
                        mean_duration=mean_duration, phase_timings=solver.timings,
                        final_diagnostics=summary(solver))
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc), final_time=solver.state.time,
                        elapsed_seconds=time.monotonic() - start)
        if is_root:
            save_snapshot(solver, out / "last_valid.npz")
        raise
    finally:
        if is_root:
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
