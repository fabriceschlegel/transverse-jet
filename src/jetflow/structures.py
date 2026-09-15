"""Publication-style mean-flow structure plots for completed simulations."""

import json
from pathlib import Path

import numpy as np

from .diagnostics import vortex_fields
from .report import pyplot


def _mean_plane(field, indices):
    values = field[..., indices]
    count = np.isfinite(values).sum(axis=-1)
    return np.divide(np.nansum(values, axis=-1), count,
                     out=np.full(count.shape, np.nan), where=count > 0)


def _plate_and_nozzle(ax):
    ax.plot([-3, -0.5], [0, 0], color="#111827", lw=2.2)
    ax.plot([0.5, 5], [0, 0], color="#111827", lw=2.2)
    ax.plot([-0.5, -0.5], [-0.08, 0], color="#111827", lw=2.2)
    ax.plot([0.5, 0.5], [-0.08, 0], color="#111827", lw=2.2)


def midplane_figure(x, y, z, fluid, velocity, scalar, omega, destination, run_label):
    plt = pyplot()
    kz = np.argsort(np.abs(z))[:2]
    valid = np.all(np.take(fluid, kz, axis=-1), axis=-1)
    u = np.take(velocity[0], kz, axis=-1).mean(axis=-1)
    v = np.take(velocity[1], kz, axis=-1).mean(axis=-1)
    speed = np.sqrt(u*u + v*v)
    concentration = np.take(scalar, kz, axis=-1).mean(axis=-1)
    omega_z = _mean_plane(omega[2], kz)
    extent = (x >= -2.5) & (x <= 5.0)
    vertical = (y >= 0) & (y <= 3.5)
    window = valid & extent[:, None] & vertical[None, :]
    omega_limit = max(0.5, float(np.nanpercentile(np.abs(omega_z[window]), 98)))

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True, sharex=True)
    ax = axes[0]
    image = ax.pcolormesh(x, y, np.ma.masked_where(~valid, speed).T,
                          shading="nearest", cmap="viridis", vmin=0,
                          vmax=float(np.nanpercentile(speed[window], 99)))
    ax.streamplot(x, y, np.ma.masked_where(~valid, u).T,
                  np.ma.masked_where(~valid, v).T, density=1.15,
                  color="#ffffffb8", linewidth=0.65, arrowsize=0.75)
    ax.contour(x, y, concentration.T, levels=[0.1, 0.5],
               colors=["#facc15", "#f97316"], linewidths=[1.1, 1.5])
    masked_u = np.ma.masked_where(~valid, u)
    ax.contourf(x, y, masked_u.T, levels=[-10, 0], colors=["#ef4444"], alpha=0.24)
    ax.contour(x, y, masked_u.T, levels=[0], colors=["#7f1d1d"], linewidths=1.4)
    _plate_and_nozzle(ax)
    ax.set(title="Mean velocity magnitude and streamlines on the vertical midplane (z ≈ 0)",
           ylabel="y / D", xlim=(-2.5, 5), ylim=(0, 3.5))
    ax.text(1.13, 0.46, "recirculation\n$u_x<0$", color="#7f1d1d",
            ha="center", va="center", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
    fig.colorbar(image, ax=ax, label=r"$|\overline{\mathbf{u}}|/U_\infty$")

    ax = axes[1]
    image = ax.pcolormesh(x, y, np.ma.masked_where(~valid, omega_z).T,
                          shading="nearest", cmap="RdBu_r",
                          vmin=-omega_limit, vmax=omega_limit)
    ax.contourf(x, y, masked_u.T, levels=[-10, 0], colors=["#f59e0b"], alpha=0.22)
    ax.contour(x, y, masked_u.T, levels=[0], colors=["#78350f"], linewidths=1.2)
    ax.contour(x, y, concentration.T, levels=[0.1, 0.5],
               colors=["#334155", "#0f172a"], linewidths=[0.8, 1.1])
    _plate_and_nozzle(ax)
    ax.set(title="Mean signed spanwise vorticity; amber marks the recirculating region",
           xlabel="x / D", ylabel="y / D", xlim=(-2.5, 5), ylim=(0, 3.5))
    fig.colorbar(image, ax=ax, label=r"$\overline{\omega}_z D/U_\infty$")
    for ax in axes:
        ax.grid(alpha=0.12)
    fig.suptitle(run_label, fontsize=15)
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return omega_limit


def crossflow_figure(x, y, z, fluid, velocity, scalar, omega, destination):
    plt = pyplot()
    requested = [0.5, 1.0, 2.0, 3.0]
    indices = [int(np.argmin(np.abs(x - value))) for value in requested]
    selected = np.concatenate([omega[0, index][np.isfinite(omega[0, index])]
                               for index in indices])
    limit = max(0.5, float(np.percentile(np.abs(selected), 98)))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), constrained_layout=True,
                             sharex=True, sharey=True)
    image = None
    for ax, index in zip(axes, indices):
        valid = fluid[index]
        field = np.ma.masked_where(~valid, omega[0, index])
        image = ax.pcolormesh(z, y, field, shading="nearest", cmap="RdBu_r",
                              vmin=-limit, vmax=limit)
        concentration = np.ma.masked_where(~valid, scalar[index])
        ax.contour(z, y, concentration, levels=[0.1, 0.5],
                   colors=["#334155", "#0f172a"], linewidths=[0.8, 1.2])
        streamwise = np.ma.masked_where(~valid, velocity[0, index])
        if np.nanmin(streamwise) < 0:
            ax.contourf(z, y, streamwise, levels=[-10, 0], colors=["#f59e0b"], alpha=0.25)
            ax.contour(z, y, streamwise, levels=[0], colors=["#78350f"], linewidths=1.0)
        stride_y, stride_z = 6, 4
        yy, zz = np.meshgrid(y[::stride_y], z[::stride_z], indexing="ij")
        ax.quiver(zz, yy, velocity[2, index, ::stride_y, ::stride_z],
                  velocity[1, index, ::stride_y, ::stride_z], color="#111827",
                  alpha=0.65, angles="xy", scale_units="xy", scale=6.0, width=0.004)
        ax.set(title=f"x / D = {x[index]:.2f}", xlabel="z / D",
               xlim=(-1.6, 1.6), ylim=(0, 3.2), aspect="equal")
        ax.grid(alpha=0.10)
    axes[0].set_ylabel("y / D")
    fig.colorbar(image, ax=axes, shrink=0.84,
                 label=r"streamwise vorticity $\overline{\omega}_x D/U_\infty$")
    fig.suptitle("Counter-rotating vortex pair in downstream cross-sections\n"
                 "red/blue: signed ωx · arrows: (w,v) · amber: ux < 0 · contours: c = 0.1, 0.5",
                 fontsize=14)
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return limit, [float(x[index]) for index in indices]


def _surface(volume, level, origin, spacing, step_size=2):
    from skimage.measure import marching_cubes
    vertices, faces, _, _ = marching_cubes(volume.astype(np.float32), level=level,
                                            spacing=(spacing, spacing, spacing),
                                            step_size=step_size,
                                            allow_degenerate=False)
    vertices += np.asarray(origin)
    # Matplotlib coordinates are streamwise x, spanwise z, wall-normal y.
    plotted = vertices[:, [0, 2, 1]]
    return plotted[faces]


def structure_3d_figure(x, y, z, fluid, velocity, omega, q, destination,
                        q_level=1.0, omega_level=0.75):
    plt = pyplot()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch

    ix = (x >= -1.0) & (x <= 4.0)
    iy = (y >= 0.0) & (y <= 3.2)
    iz = (z >= -1.6) & (z <= 1.6)
    xs, ys, zs = x[ix], y[iy], z[iz]
    crop = np.ix_(ix, iy, iz)
    valid = fluid[crop]
    q_crop = np.where(valid & np.isfinite(q[crop]) & (q[crop] > 0), q[crop], 0.0)
    wx = np.where(valid & np.isfinite(omega[0][crop]), omega[0][crop], 0.0)
    positive = np.maximum(wx, 0.0)
    negative = np.maximum(-wx, 0.0)
    recirculation = (valid & (velocity[0][crop] < 0)).astype(float)
    h = float(x[1] - x[0])
    origin = (float(xs[0]), float(ys[0]), float(zs[0]))

    surfaces = []
    for field, level, color, alpha, label in (
        (q_crop, q_level, "#7c3aed", 0.14, f"Q = {q_level:g}"),
        (positive, omega_level, "#dc2626", 0.42, f"ωx = +{omega_level:g}"),
        (negative, omega_level, "#2563eb", 0.42, f"ωx = −{omega_level:g}"),
        (recirculation, 0.5, "#f59e0b", 0.30, "recirculation ux < 0"),
    ):
        if np.nanmax(field) > level:
            surfaces.append((_surface(field, level, origin, h), color, alpha, label))

    fig = plt.figure(figsize=(15, 6.5), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, 1, projection="3d"),
            fig.add_subplot(1, 2, 2, projection="3d")]
    for ax in axes:
        for triangles, color, alpha, _ in surfaces:
            collection = Poly3DCollection(triangles, facecolor=color, edgecolor="none", alpha=alpha)
            ax.add_collection3d(collection)
        angle = np.linspace(0, 2*np.pi, 120)
        ax.plot(0.5*np.cos(angle), 0.5*np.sin(angle), np.zeros_like(angle),
                color="#111827", lw=1.4)
        ax.set(xlabel="x / D", ylabel="z / D", zlabel="y / D",
               xlim=(-1, 4), ylim=(-1.6, 1.6), zlim=(0, 3.2))
        ax.set_box_aspect((5, 3.2, 3.2))
    axes[0].view_init(elev=24, azim=-62)
    axes[0].set_title("Oblique view")
    axes[1].view_init(elev=9, azim=2)
    axes[1].set_title("Downstream-axis view: paired red/blue vortex cores")
    axes[0].legend(handles=[Patch(facecolor=color, alpha=alpha, label=label)
                            for _, color, alpha, label in surfaces],
                   loc="upper left", frameon=False)
    fig.suptitle("Mean three-dimensional vortical and recirculating structures\n"
                 "positive Q identifies rotation-dominated regions; signed ωx exposes the counter-rotating pair",
                 fontsize=14)
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_structures(directory):
    out = Path(directory).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only completed runs can be visualized")
    with np.load(out / "final.npz", allow_pickle=False) as final:
        x, y, z = final["x"].copy(), final["y"].copy(), final["z"].copy()
    with np.load(out / "mean.npz", allow_pickle=False) as mean:
        velocity, scalar, fluid = (mean[name].copy() for name in ("velocity", "scalar", "fluid"))
        average = {name: float(mean[name]) for name in ("start", "end", "duration")}
    h = float(x[1] - x[0])
    omega, q, _ = vortex_fields(velocity, h, fluid)
    destination = out / "structures"
    destination.mkdir(exist_ok=True)
    midplane = destination / "mean-midplane-velocity-vorticity.png"
    crossflow = destination / "mean-crossflow-crvp.png"
    structure_3d = destination / "mean-3d-vortices-recirculation.png"
    run_label = (
        f"{manifest['cells_per_diameter']:g} cells/D · {manifest['pressure_backend']} · "
        f"time-averaged over {average['start']:g} ≤ tU∞/D ≤ {average['end']:g}"
    )
    omega_z_limit = midplane_figure(
        x, y, z, fluid, velocity, scalar, omega, midplane, run_label)
    omega_x_limit, sections = crossflow_figure(x, y, z, fluid, velocity, scalar, omega, crossflow)
    structure_3d_figure(x, y, z, fluid, velocity, omega, q, structure_3d)
    recirculation = fluid & (velocity[0] < 0) & (y[None, :, None] >= 0)
    recirculation_cells = int(recirculation.sum())
    recirculation_volume = float(recirculation_cells * h**3)
    locations = np.argwhere(recirculation)
    recirculation_bounds = {
        "x_D": [float(x[locations[:, 0]].min()), float(x[locations[:, 0]].max())],
        "y_D": [float(y[locations[:, 1]].min()), float(y[locations[:, 1]].max())],
        "z_D": [float(z[locations[:, 2]].min()), float(z[locations[:, 2]].max())],
    } if locations.size else None
    record = {
        "source_run": str(out),
        "field_basis": "physical-time-weighted mean velocity and scalar",
        "average_window": average,
        "cells_per_diameter": manifest["cells_per_diameter"],
        "thresholds": {"Q": 1.0, "abs_streamwise_vorticity": 0.75,
                       "recirculation": "mean streamwise velocity < 0",
                       "midplane_omega_limit": omega_z_limit,
                       "crossflow_omega_limit": omega_x_limit},
        "crossflow_sections_x": sections,
        "recirculation_cells": recirculation_cells,
        "recirculation_volume_D3": recirculation_volume,
        "recirculation_cell_center_bounds": recirculation_bounds,
        "interpretation": "Q and signed vorticity surfaces are visualization thresholds, not uniquely identified vortex boundaries.",
        "figures": [str(path) for path in (midplane, crossflow, structure_3d)],
    }
    (destination / "structures.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
