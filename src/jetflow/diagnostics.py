import numpy as np


def velocity_gradient(velocity, h):
    """J[i,j] = du_i / dx_j for cell-centered fields."""
    return np.stack([np.stack(np.gradient(a, h, edge_order=2)) for a in velocity])


def vortex_fields(velocity, h, fluid):
    J = velocity_gradient(velocity, h)
    omega = np.stack([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
    S, O = (J + J.swapaxes(0, 1)) / 2, (J - J.swapaxes(0, 1)) / 2
    q = 0.5 * np.sum(O**2 - S**2, axis=(0, 1))
    # Exclude solid-adjacent stencils; avoid interpreting artificial solid gradients.
    valid = fluid.copy()
    for axis in range(3):
        valid &= np.roll(fluid, 1, axis=axis) & np.roll(fluid, -1, axis=axis)
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = 0, -1
        valid[tuple(a)] = False
        valid[tuple(b)] = False
    omega[:, ~valid], q[~valid] = np.nan, np.nan
    return omega, q, valid


def wall_shear(grid, centered_velocity):
    g = grid
    j = g.wall_j
    plate = g.fluid[:, j, :] & ~g.fluid[:, j - 1, :]
    # Quadratic reconstruction through u(0)=0, u(h/2), u(3h/2).
    tau = g.config.nu * (9 * centered_velocity[[0, 2], :, j, :]
                         - centered_velocity[[0, 2], :, j + 1, :]) / (3 * g.h)
    tau[:, ~plate] = np.nan
    return tau, plate


def trajectory(grid, scalar):
    c = scalar[:, grid.wall_j:, :]
    mass = c.sum(axis=(1, 2))
    y, z = grid.y[grid.wall_j:], grid.z
    denominator = np.where(mass > 1e-8, mass, np.nan)
    yc = (c * y[None, :, None]).sum(axis=(1, 2)) / denominator
    zc = (c * z[None, None, :]).sum(axis=(1, 2)) / denominator
    width = np.sqrt((c * (z[None, None, :] - zc[:, None, None])**2).sum(axis=(1, 2)) / denominator)
    return yc, zc, width, mass * grid.h**2


def upstream_circulation(grid, centered):
    """Counterclockwise circulation of an upstream x-y rectangle, z=0.

    This is a signed box diagnostic, NOT an identified horseshoe-vortex strength.
    Bounds snap to cell-center x and y locations; bottom lies on the plate.
    """
    g = grid
    iz = np.argsort(np.abs(g.z))[:2]
    uv = centered[:, :, :, iz].mean(axis=-1)
    left = int(np.argmin(abs(g.x + min(2.0, -g.config.x_min - g.h))))
    right = int(np.argmin(abs(g.x + 0.5 + g.h / 2)))
    top = int(np.argmin(abs(g.y - 1.0)))
    j = g.wall_j
    ys = np.r_[0.0, g.y[j:top + 1]]
    right_line = np.r_[0.0, uv[1, right, j:top + 1]]
    left_line = np.r_[0.0, uv[1, left, j:top + 1]]
    gamma = (np.trapezoid(right_line - left_line, ys)
             - np.trapezoid(uv[0, left:right + 1, top], g.x[left:right + 1]))
    return float(gamma), [float(g.x[left]), float(g.x[right]), 0.0, float(g.y[top])]


def summary(solver):
    g, s = solver.grid, solver.state
    centered = g.centers(s.velocity)
    divergence = g.divergence(s.velocity)[g.fluid]
    tau, plate = wall_shear(g, centered)
    upstream = plate & (g.x[:, None] < -0.5) & (g.x[:, None] > -2)
    qin = (s.velocity[0][0].sum() + s.velocity[1][:, 0, :].sum()) * g.h**2
    qout = s.velocity[0][-1].sum() * g.h**2
    mass = s.scalar.sum() * g.h**3
    gamma, _ = upstream_circulation(g, centered)
    yc, zc, width, cmass = trajectory(g, s.scalar)
    ix = int(np.argmin(abs(g.x - 2)))
    valid = cmass[ix] > 1e-8
    return {
        "time": float(s.time), "step": s.step,
        "divergence_linf": float(np.max(abs(divergence))),
        "divergence_rms": float(np.sqrt(np.mean(divergence**2))),
        "volume_inflow": float(qin), "volume_outflow": float(qout),
        "relative_volume_imbalance": float(abs(qin - qout) / max(abs(qin), 1e-14)),
        "scalar_mass": float(mass),
        "scalar_balance_error": float(mass - solver.initial_scalar_mass - s.scalar_boundary_integral),
        "scalar_min": float(s.scalar[g.fluid].min()), "scalar_max": float(s.scalar.max()),
        "kinetic_energy": float(0.5 * np.sum(centered[:, g.fluid]**2) * g.h**3),
        "max_speed": float(np.max(np.linalg.norm(centered[:, g.fluid], axis=0))),
        "outlet_backflow_fraction": float(np.mean(s.velocity[0][-1][g.outlet] < 0)),
        "upstream_box_circulation": gamma,
        "upstream_reversed_shear_area": float(np.sum((tau[0] < 0) & upstream) * g.h**2),
        "upstream_shear_min": float(np.nanmin(tau[0][upstream])) if upstream.any() else None,
        "trajectory_x": float(g.x[ix]),
        "scalar_centroid_y": float(yc[ix]) if valid else None,
        "scalar_centroid_z": float(zc[ix]) if valid else None,
        "scalar_width_z": float(width[ix]) if valid else None,
        "pressure_iterations": solver.projection.iterations,
    }

