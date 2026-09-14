from dataclasses import dataclass
import numpy as np
from .grid import Grid
from .pressure import Projection


@dataclass
class State:
    velocity: list
    scalar: np.ndarray
    pressure: np.ndarray
    time: float = 0.0
    step: int = 0
    scalar_boundary_integral: float = 0.0


class Solver:
    def __init__(self, config):
        self.config = config
        self.grid = g = Grid(config)
        self.projection = Projection(g)
        vel = g.zeros()
        vel[0][:] = g.cross_profile[None, :, None]
        # Fill the submerged pipe with its imposed parabolic inlet profile.
        yf = np.arange(g.shape[1] + 1) * g.h - config.pipe_depth
        vel[1][:] = g.pipe_profile[:, None, :] * (yf[None, :, None] <= 0)
        g.boundary_conditions(vel, extrapolate_outlet=True)
        pressure = self.projection.apply(vel, 1.0)
        scalar = ((g.y[None, :, None] < 0) & g.fluid).astype(float)
        self.state = State(vel, scalar, pressure)
        self.initial_scalar_mass = scalar.sum() * g.h**3

    def momentum_rhs(self, velocity):
        g, c = self.grid, self.config
        result = []
        for component_axis, a in enumerate(velocity):
            rate = np.zeros_like(a)
            for axis in range(3):
                minus, plus = g.neighbors(a, component_axis, axis)
                advector = g.at_component(velocity, axis, component_axis)
                rate += -advector * (plus - minus) / (2 * g.h)
                rate += c.nu * (plus - 2 * a + minus) / g.h**2
            rate[~g.free[component_axis]] = 0
            result.append(rate)
        return result

    def scalar_rhs(self, scalar, velocity):
        """Conservative donor-cell advection plus centered molecular diffusion.

        No clipping: conservation and scalar bounds are checked independently.
        Scalar numerical diffusion is explicitly a prototype limitation.
        """
        g, c = self.grid, self.config
        fluxes = g.zeros()
        for axis, (vel, flux) in enumerate(zip(velocity, fluxes)):
            low, high, mid = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
            low[axis], high[axis], mid[axis] = slice(None, -1), slice(1, None), slice(1, -1)
            low, high, mid = tuple(low), tuple(high), tuple(mid)
            left, right = scalar[low], scalar[high]
            connected = g.fluid[low] & g.fluid[high]
            flux[mid] = (vel[mid] * np.where(vel[mid] >= 0, left, right)
                         - c.kappa * (right - left) / g.h) * connected
        # Crossflow inlet c=0, pipe reservoir c=1; Dirichlet half-cell diffusion.
        fluxes[0][0] = (velocity[0][0] * np.where(velocity[0][0] >= 0, 0, scalar[0])
                         - 2 * c.kappa * scalar[0] / g.h) * g.inlet
        fluxes[1][:, 0, :] = (velocity[1][:, 0, :] * np.where(velocity[1][:, 0, :] >= 0, 1, scalar[:, 0, :])
                              - 2 * c.kappa * (scalar[:, 0, :] - 1) / g.h) * g.pipe_inlet
        # Zero diffusive outlet flux; ambient scalar on any outlet backflow.
        fluxes[0][-1] = velocity[0][-1] * np.where(velocity[0][-1] >= 0, scalar[-1], 0) * g.outlet
        rhs = -g.divergence(fluxes)
        rhs[~g.fluid] = 0
        net_in = (fluxes[0][0].sum() + fluxes[1][:, 0, :].sum() - fluxes[0][-1].sum()) * g.h**2
        return rhs, float(net_in)

    def timestep(self):
        c, g, s = self.config, self.grid, self.state
        speed_rate = sum(np.max(np.abs(a)) for a in s.velocity) / g.h
        # 8 rather than 6 covers half-cell Dirichlet diffusion at inlet corners.
        return min(c.dt_max, c.cfl / (speed_rate + 8 * max(c.nu, c.kappa) / g.h**2))

    def advance(self, dt=None):
        """SSP-RK3 method-of-lines stages, each followed by a MAC projection."""
        dt = self.timestep() if dt is None else dt
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        g, s = self.grid, self.state
        u0, c0 = s.velocity, s.scalar
        def stage(velocity, scalar, base_weight):
            rhs_u = self.momentum_rhs(velocity)
            rhs_c, net = self.scalar_rhs(scalar, velocity)
            result = [base_weight * b + (1 - base_weight) * (a + dt * rate)
                      for b, a, rate in zip(u0, velocity, rhs_u)]
            new_scalar = base_weight * c0 + (1 - base_weight) * (scalar + dt * rhs_c)
            g.boundary_conditions(result, extrapolate_outlet=True)
            pressure = self.projection.apply(result, (1 - base_weight) * dt)
            return result, new_scalar, pressure, net
        u1, c1, _, q0 = stage(u0, c0, 0)
        u2, c2, _, q1 = stage(u1, c1, 0.75)
        u3, c3, p3, q2 = stage(u2, c2, 1 / 3)
        if not all(np.all(np.isfinite(a)) for a in [*u3, c3, p3]):
            raise RuntimeError("Nonfinite solution; reduce timestep/refine grid")
        if c3.min() < -1e-7 or c3.max() > 1 + 1e-7:
            raise RuntimeError(f"Scalar bounds violated: [{c3.min()}, {c3.max()}]")
        divergence = np.max(np.abs(g.divergence(u3)[g.fluid]))
        if divergence > 1e-6:
            raise RuntimeError(f"Discrete divergence {divergence:g} exceeds acceptance tolerance")
        self.state = State(u3, c3, p3, s.time + dt, s.step + 1,
                           s.scalar_boundary_integral + dt * (q0 / 6 + q1 / 6 + 2 * q2 / 3))
        return dt
