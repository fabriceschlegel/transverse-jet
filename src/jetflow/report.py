import csv
import html
import json
import os
from pathlib import Path
import numpy as np


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f8fafc", "axes.facecolor": "#f8fafc"})
    return plt


def read_history(directory):
    with (Path(directory) / "history.csv").open() as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(row[key]) if row[key] else np.nan for row in rows]) for key in rows[0]}


def snapshot_figure(path, destination, omega_limit, tau_limit):
    plt = pyplot()
    with np.load(path, allow_pickle=False) as a:
        x, y, z = a["x"], a["y"], a["z"]
        kz = np.argsort(abs(z))[:2]
        velocity = np.mean(a["velocity"][:, :, :, kz], axis=-1)
        scalar = a["scalar"][:, :, kz].mean(axis=-1)
        oz = a["vorticity"][2][:, :, kz]
        count = np.isfinite(oz).sum(axis=-1)
        omega = np.divide(np.nansum(oz, axis=-1), count,
                          out=np.full(count.shape, np.nan), where=count > 0)
        fluid = np.all(a["fluid"][:, :, kz], axis=-1)
        t = float(a["time"])
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        im = axes[0, 0].pcolormesh(x, y, np.ma.masked_where(~fluid, scalar).T, cmap="viridis", vmin=0, vmax=1, shading="nearest")
        uu = np.ma.masked_where(~fluid, velocity[0]).T
        vv = np.ma.masked_where(~fluid, velocity[1]).T
        axes[0, 0].streamplot(x, y, uu, vv, density=0.9, color="#ffffff90", linewidth=0.55, arrowsize=0.7)
        axes[0, 0].set(title="Jet scalar and instantaneous streamlines · z ≈ 0", xlabel="x / D", ylabel="y / D")
        fig.colorbar(im, ax=axes[0, 0], label="Jet concentration c")
        im = axes[0, 1].pcolormesh(x, y, omega.T, cmap="RdBu_r", vmin=-omega_limit, vmax=omega_limit, shading="nearest")
        axes[0, 1].set(title="Signed spanwise vorticity · upstream region", xlim=(-2.5, 1), ylim=(0, 1.75), xlabel="x / D", ylabel="y / D")
        fig.colorbar(im, ax=axes[0, 1], label="ωz D / U∞")
        box = a["circulation_box"]
        axes[0, 1].plot([box[0], box[1], box[1], box[0], box[0]], [box[2], box[2], box[3], box[3], box[2]], color="#334155", lw=1, ls="--")
        tau = a["wall_shear"]
        im = axes[1, 0].pcolormesh(x, z, tau[0].T, cmap="RdBu_r", vmin=-tau_limit, vmax=tau_limit, shading="nearest")
        if np.nanmin(tau[0]) < 0 < np.nanmax(tau[0]):
            axes[1, 0].contour(x, z, tau[0].T, levels=[0], colors=["#1e293b"], linewidths=0.8)
        stride = max(1, len(x) // 25)
        axes[1, 0].quiver(x[::stride], z[::stride], tau[0, ::stride, ::stride].T, tau[1, ::stride, ::stride].T, color="#334155", alpha=0.6)
        axes[1, 0].add_patch(plt.Circle((0, 0), 0.5, fill=False, color="#0f172a", lw=1))
        axes[1, 0].set(title="Plate shear · black contour marks τx = 0", xlim=(-2.5, 3), xlabel="x / D", ylabel="z / D")
        fig.colorbar(im, ax=axes[1, 0], label="τx / (ρ U∞²)")
        ix = (x > 0.5) & (a["scalar_section_integral"] > 1e-5)
        axes[1, 1].plot(x[ix], a["trajectory_y"][ix], lw=2, color="#0f766e", label="Scalar centroid height")
        axes[1, 1].plot(x[ix], a["scalar_width_z"][ix], lw=2, color="#c2410c", label="Scalar spanwise standard deviation")
        axes[1, 1].set(title="Concentration-weighted geometry", xlabel="x / D", ylabel="Distance / D", ylim=(0, None))
        axes[1, 1].legend(frameon=False)
        for ax in axes.flat:
            ax.grid(alpha=0.12)
        fig.suptitle(f"Unforced jet in crossflow  |  t U∞ / D = {t:.2f}\nCoarse prototype · no claim of resolved horseshoe dynamics", fontsize=15)
        fig.savefig(destination, dpi=140, bbox_inches="tight")
        plt.close(fig)


def render(directory):
    plt = pyplot()
    out = Path(directory).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only completed runs can be rendered")
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    paths = sorted((out / "snapshots").glob("state_*.npz"))
    with np.load(out / "final.npz", allow_pickle=False) as a:
        final_time = float(a["time"])
    last_time = -np.inf
    if paths:
        with np.load(paths[-1], allow_pickle=False) as a:
            last_time = float(a["time"])
    if last_time < final_time - 1e-8:
        paths.append(out / "final.npz")
    omega_limit, tau_limit = 1e-3, 1e-3
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            omega_limit = max(omega_limit, float(np.nanpercentile(abs(a["vorticity"][2]), 98)))
            tau_limit = max(tau_limit, float(np.nanpercentile(abs(a["wall_shear"][0]), 98)))
    frames = []
    for i, path in enumerate(paths):
        target = figures / f"frame_{i:03d}.png"
        snapshot_figure(path, target, omega_limit, tau_limit)
        with np.load(path, allow_pickle=False) as a:
            frames.append({"time": float(a["time"]), "src": str(target.relative_to(out))})
    h = read_history(out)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(h["time"], h["upstream_box_circulation"], color="#0f766e")
    axes[0, 0].set(title="Signed upstream-box circulation (not isolated vortex strength)", ylabel="Γ / (U∞ D)")
    axes[0, 1].plot(h["time"], h["upstream_reversed_shear_area"], color="#c2410c")
    axes[0, 1].set(title="Upstream plate area with reversed streamwise shear", ylabel="Area / D²")
    axes[1, 0].semilogy(h["time"], np.maximum(h["divergence_linf"], 1e-16), label="Max discrete divergence")
    axes[1, 0].semilogy(h["time"], np.maximum(h["relative_volume_imbalance"], 1e-16), label="Relative volume imbalance")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set(title="Incompressibility and global flux checks", ylabel="Nondimensional residual")
    axes[1, 1].plot(h["time"], h["scalar_mass"], label="Integrated scalar", color="#0f766e")
    axes[1, 1].set(title="Conservative scalar transport", ylabel="Scalar amount / D³")
    for ax in axes.flat:
        ax.set_xlabel("t U∞ / D"); ax.grid(alpha=0.15)
    fig.savefig(figures / "history.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # An actual 3-D field view, with a clearly labelled visualization threshold.
    with np.load(out / "final.npz", allow_pickle=False) as a:
        X, Y, Z = np.meshgrid(a["x"], a["y"], a["z"], indexing="ij")
        mask = (a["scalar"] > 0.12) & (Y >= 0) & a["fluid"]
        inds = np.flatnonzero(mask)[::max(1, int(mask.sum()) // 10000)]
        fig = plt.figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.add_subplot(projection="3d")
        dots = ax.scatter(X.ravel()[inds], Z.ravel()[inds], Y.ravel()[inds], c=a["scalar"].ravel()[inds],
                          s=5, alpha=0.2, cmap="viridis", vmin=0, vmax=1, linewidths=0)
        angle = np.linspace(0, 2 * np.pi, 100)
        ax.plot(0.5 * np.cos(angle), 0.5 * np.sin(angle), np.zeros_like(angle), color="#0f172a")
        ax.set(xlabel="x / D", ylabel="z / D", zlabel="y / D", zlim=(0, float(a["y"].max())),
               title=f"3-D jet scalar at t = {final_time:.2f} · cells with c > 0.12\nConcentration cloud, not a vortex surface")
        ax.view_init(elev=22, azim=-65)
        fig.colorbar(dots, ax=ax, shrink=0.65, label="c")
        fig.savefig(figures / "jet-3d.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    d = manifest["final_diagnostics"]
    cards = [("Grid", f"{manifest['cells_per_diameter']:g} cells / D"),
             ("Fluid cells", f"{manifest['active_cells']:,}"),
             ("Final time", f"{final_time:g} D / U∞"),
             ("Max sampled divergence", f"{manifest['sampled_max_divergence']:.2e}"),
             ("Scalar balance error", f"{manifest['sampled_max_scalar_balance_error']:.2e}")]
    markup = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unforced transverse jet — simulation report</title><style>
body{font:16px/1.55 system-ui,sans-serif;background:#f8fafc;color:#172b3a;margin:0}main{max-width:1200px;margin:auto;padding:38px 24px}
h1{font-size:38px;line-height:1.15;letter-spacing:-1px}h2{margin-top:36px}p{max-width:900px;color:#435466}.eyebrow{color:#0f766e;font-weight:650;letter-spacing:2px;font-size:12px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:25px 0}.card{background:white;border:1px solid #dde5e9;border-radius:10px;padding:16px;min-width:150px}.card b{display:block;font-size:22px;color:#0f766e}.card span{font-size:12px;color:#526676}
img{max-width:100%;border:1px solid #e1e8ed;border-radius:12px;background:white}.slider{display:flex;gap:20px;align-items:center;padding:18px 0}input{flex:1;accent-color:#0f766e}a{color:#0f766e}.note{border-left:3px solid #ca8a04;padding:10px 18px;background:#fffbeb}code{background:#e8eef1;padding:2px 5px;border-radius:4px}
</style><main><div class="eyebrow">COMPUTATIONAL FLUID DYNAMICS · RESEARCH PROTOTYPE</div>
<h1>Unforced transverse jet</h1><p>A three-dimensional, incompressible jet issuing through a pipe into a crossflow boundary layer. The initial focus is the upstream wall interaction and the development of the jet.</p>
"""
    markup += '<div class="cards">' + ''.join(f'<div class="card"><b>{html.escape(value)}</b><span>{html.escape(key)}</span></div>' for key, value in cards) + '</div>'
    markup += '<p class="note">This run verifies the computational workflow. Its grid, staircase pipe, short duration, and donor-cell scalar transport do not establish resolved horseshoe dynamics or quantitative mixing. Negative wall shear is evidence of local reversal, not proof of a coherent horseshoe vortex.</p>'
    markup += f'<h2>Follow the flow in time</h2><p>Color scales stay fixed across frames. Blank vorticity pixels exclude solid-adjacent derivative stencils. Streamlines are instantaneous, not particle trajectories.</p><div class="slider"><label for="time">Snapshot</label><input id="time" type="range" min="0" max="{len(frames)-1}" value="{len(frames)-1}" step="1"><output id="label"></output></div><img id="frame" alt="Scalar, upstream vorticity, plate shear, and scalar trajectory" src="{frames[-1]["src"]}">'
    markup += '<h2>Three-dimensional concentration field</h2><img src="figures/jet-3d.png" alt="Three-dimensional scalar concentration cloud"><h2>Flow diagnostics and conservation</h2><img src="figures/history.png" alt="Time histories of upstream flow and conservation residuals">'
    markup += f'<p>Last sampled volume-flow mismatch: <code>{d["relative_volume_imbalance"]:.2e}</code>. Scalar range: <code>{d["scalar_min"]:.6g}–{d["scalar_max"]:.6g}</code>. Time-weighted averages cover <code>{manifest["mean_duration"]:.3f} D/U∞</code>; stationarity has not been established.</p>'
    markup += '<h2>Reproducible data</h2><p><a href="config.json">Case parameters</a> · <a href="manifest.json">Run provenance and checks</a> · <a href="history.csv">Diagnostic time series</a> · <a href="final.npz">Final 3-D fields</a> · <a href="mean.npz">Time-weighted mean fields</a></p><p>Axes: x along the crossflow, y upward from the plate, z spanwise. Lengths use D; velocities use U∞. The nozzle is centered at x = z = 0. No actuator is included.</p>'
    markup += '<script>const frames=' + json.dumps(frames) + ';const slider=document.getElementById("time");function update(){const f=frames[+slider.value];document.getElementById("frame").src=f.src;document.getElementById("label").textContent="t = "+f.time.toFixed(2)+" D/U∞";}slider.addEventListener("input",update);update();</script></main></html>'
    (out / "report.html").write_text(markup)
    return out / "report.html"


def compare_runs(directories, output):
    plt = pyplot()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    keys = ["upstream_box_circulation", "upstream_reversed_shear_area", "scalar_centroid_y", "scalar_width_z"]
    labels = ["Upstream-box circulation", "Reversed-shear area", "Scalar centroid height near x=2D", "Scalar spanwise width near x=2D"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    records = []
    for directory in directories:
        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text())
        h = read_history(path)
        label = f"{manifest['cells_per_diameter']:g} cells/D · {path.name}"
        records.append({"directory": str(path.resolve()), "cells_per_diameter": manifest["cells_per_diameter"],
                        "final_time": manifest["final_time"], "final_diagnostics": manifest["final_diagnostics"]})
        for ax, key, title in zip(axes.flat, keys, labels):
            ax.plot(h["time"], h[key], label=label); ax.set(title=title, xlabel="t U∞ / D"); ax.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Resolution sensitivity · agreement of two coarse grids does not establish convergence")
    fig.savefig(out / "comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    (out / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    return out / "comparison.png"
