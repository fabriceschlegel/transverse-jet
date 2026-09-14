from dataclasses import replace
import numpy as np
import pytest
from jetflow.config import Config
from jetflow.grid import Grid
from jetflow.solver import Solver
from jetflow.diagnostics import summary, velocity_gradient, vortex_fields, wall_shear
from jetflow.output import save_snapshot, restore


@pytest.fixture
def solver():
    return Solver(Config(x_min=-2, x_max=3, height=2, half_width=1, pipe_depth=1))


def test_projection_removes_divergence_and_preserves_boundaries(solver):
    g = solver.grid
    rng = np.random.default_rng(812)
    u = [rng.normal(size=a.shape) for a in g.free]
    g.boundary_conditions(u)
    inlet = u[0][0].copy()
    jet = u[1][:, 0].copy()
    solver.projection.apply(u, 0.013)
    assert np.max(abs(g.divergence(u)[g.fluid])) < 1e-7
    np.testing.assert_array_equal(u[0][0], inlet)
    np.testing.assert_array_equal(u[1][:, 0], jet)
    for axis, a in enumerate(u):
        solid_faces = g.boundary[axis].copy()
        index = [slice(None)] * 3
        index[axis] = 0
        solid_faces[tuple(index)] = False
        index[axis] = -1
        solid_faces[tuple(index)] = False
        assert np.all(a[solid_faces] == 0)
    assert abs(u[0][-1].sum() - inlet.sum() - jet.sum()) < 1e-7


def test_pressure_gradient_recovery(solver):
    g = solver.grid
    rng = np.random.default_rng(22)
    p = np.zeros(g.shape)
    p[g.fluid] = rng.normal(size=g.fluid.sum())
    velocity = solver.projection.gradient(p)
    recovered = solver.projection.apply(velocity, 1)
    np.testing.assert_allclose(recovered[g.fluid], p[g.fluid], atol=3e-8)
    assert max(np.max(abs(a)) for a in velocity) < 1e-7


def test_wall_ghost_and_wall_shear(solver):
    g = solver.grid
    velocity = g.zeros()
    velocity[0][:] = np.maximum(g.y, 0)[None, :, None]
    g.boundary_conditions(velocity)
    i = 2  # flat plate away from pipe and outer boundary
    j, k = g.wall_j, len(g.z) // 2
    minus, plus = g.neighbors(velocity[0], 0, 1)
    assert minus[i, j, k] == -velocity[0][i, j, k]
    assert abs((plus[i, j, k] - 2 * velocity[0][i, j, k] + minus[i, j, k])) < 1e-14
    shear, plate = wall_shear(g, g.centers(velocity))
    assert plate[i, k]
    assert shear[0, i, k] == pytest.approx(g.config.nu)


def test_rotation_and_strain_diagnostics():
    h = 0.1
    x, y, z = np.meshgrid(np.arange(10)*h, np.arange(10)*h, np.arange(10)*h, indexing="ij")
    velocity = np.stack([-2 * y, 2 * x, np.zeros_like(z)])
    omega, q, valid = vortex_fields(velocity, h, np.ones(x.shape, bool))
    np.testing.assert_allclose(omega[2][valid], 4, atol=1e-12)
    np.testing.assert_allclose(q[valid], 4, atol=1e-12)
    J = velocity_gradient(np.stack([x, -y, z * 0]), h)
    np.testing.assert_allclose(J[0, 0], 1, atol=1e-12)
    _, q, valid = vortex_fields(np.stack([x, -y, z * 0]), h, np.ones(x.shape, bool))
    np.testing.assert_allclose(q[valid], -1, atol=1e-12)


def test_scalar_conservation_bounds_symmetry_and_restart(solver, tmp_path):
    for _ in range(12):
        solver.advance(0.005)
    stats = summary(solver)
    assert abs(stats["scalar_balance_error"]) < 1e-10
    assert stats["scalar_min"] >= -1e-8 and stats["scalar_max"] <= 1+1e-8
    assert abs(stats["scalar_centroid_z"] or 0) < 1e-8
    path = tmp_path / "checkpoint.npz"
    save_snapshot(solver, path)
    restarted = Solver(solver.config)
    restore(restarted, path)
    for _ in range(5):
        solver.advance(0.005)
        restarted.advance(0.005)
    np.testing.assert_allclose(restarted.state.scalar, solver.state.scalar, atol=1e-10)
    for a, b in zip(restarted.state.velocity, solver.state.velocity):
        np.testing.assert_allclose(a, b, atol=1e-9)
    assert abs(summary(restarted)["scalar_balance_error"]) < 1e-10


def test_incompatible_restart_rejected(solver, tmp_path):
    path = tmp_path / "checkpoint.npz"
    save_snapshot(solver, path)
    other = Solver(replace(solver.config, velocity_ratio=1.5))
    with pytest.raises(ValueError, match="velocity_ratio"):
        restore(other, path)


def test_momentum_diffusion_converges_with_no_slip_and_free_slip_walls():
    # Exact shear eigenmode u(y,t)=sin(pi*y/(2H))*exp(-nu*k^2*t).
    # Evaluate the production momentum operator on a homogeneous full channel;
    # the bottom is no-slip and the top is free-slip. Tests both boundary rows.
    errors = []
    for h in [0.25, 0.125, 0.0625]:
        config = Config(h=h, x_min=-1, x_max=2, pipe_depth=1, height=1, half_width=1)
        base = Grid(config)
        grid = Grid(config, fluid=np.ones(base.shape, bool))
        model = Solver.__new__(Solver)
        model.config, model.grid = config, grid
        velocity = grid.zeros()
        H = config.height + config.pipe_depth
        k = np.pi / (2 * H)
        mode = np.sin(k * (grid.y + config.pipe_depth))
        velocity[0][:] = mode[None, :, None]
        actual = model.momentum_rhs(velocity)[0][3, :, 3]
        exact = -config.nu * k*k * mode
        errors.append(np.sqrt(np.mean((actual-exact)**2)))
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8


def test_scalar_flux_budget_with_arbitrary_scalar(solver):
    g = solver.grid
    scalar = np.random.default_rng(43).uniform(size=g.shape) * g.fluid
    rhs, net_in = solver.scalar_rhs(scalar, solver.state.velocity)
    assert rhs.sum() * g.h**3 == pytest.approx(net_in, abs=1e-12)


def test_manufactured_nonlinear_momentum_converges():
    # Taylor-Green velocity. Exact -u.grad(u) = (-sin(x)cos(x),-sin(y)cos(y),0).
    # Interior test; not a test of open-boundary pressure accuracy.
    errors = []
    for h in [0.25, 0.125, 0.0625]:
        c = Config(h=h, x_min=-1, x_max=2, pipe_depth=1, height=1, half_width=1)
        base = Grid(c)
        g = Grid(c, fluid=np.ones(base.shape, bool))
        model = Solver.__new__(Solver)
        model.config, model.grid = c, g
        velocity = g.zeros()
        for axis, a in enumerate(velocity):
            coords = [g.x, g.y, g.z]
            coords[axis] = np.r_[coords[axis] - h/2, coords[axis][-1]+h/2]
            x, y, z = np.meshgrid(*coords, indexing="ij")
            if axis == 0:
                a[:] = np.sin(x)*np.cos(y)
            elif axis == 1:
                a[:] = -np.cos(x)*np.sin(y)
        rhs = model.momentum_rhs(velocity)
        errors_here = []
        for axis in (0, 1):
            coords = [g.x, g.y, g.z]
            coords[axis] = np.r_[coords[axis]-h/2, coords[axis][-1]+h/2]
            x, y, z = np.meshgrid(*coords, indexing="ij")
            t = (x, y)[axis]
            exact = -np.sin(t)*np.cos(t)-2*c.nu*velocity[axis]
            interior = (x > -0.5) & (x < 1.5) & (y > -0.5) & (y < 0.5) & (abs(z) < 0.5)
            errors_here.extend((rhs[axis][interior]-exact[interior]).tolist())
        errors.append(np.sqrt(np.mean(np.array(errors_here)**2)))
    assert errors[0]/errors[1] > 3.5
    assert errors[1]/errors[2] > 3.5


@pytest.mark.parametrize("kwargs", [{"h": 0.23}, {"schmidt": 0}, {"cfl": 1}, {"end_time": float("nan")}])
def test_invalid_cases_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs).validate()
