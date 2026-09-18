from dataclasses import dataclass
from collections import deque
from time import perf_counter
import numpy as np
from .grid import Grid
from .pressure import Projection
from .acceleration import momentum_component, scalar_rhs_kernel


@dataclass
class State:
    velocity: list
    scalar: np.ndarray
    pressure: np.ndarray
    time: float = 0.0
    step: int = 0
    scalar_boundary_integral: float = 0.0


class Solver:
    def _add_timing(self, key, duration):
        # Some operator-verification tests construct a lightweight Solver via
        # __new__ to isolate the stencil without building a pressure matrix.
        if hasattr(self, "timings"):
            self.timings[key] += duration

    def __init__(self, config, pressure_backend="pyamg"):
        self.timings = {"momentum_seconds": 0.0, "scalar_seconds": 0.0,
                        "pressure_seconds": 0.0, "advance_seconds": 0.0}
        self.config = config
        self.grid = g = Grid(config)
        self.projection = Projection(g, pressure_backend)
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
        self.reset_growth_guard()
        # Compile outside the measured simulation interval. Numba caches the
        # machine code, so later processes normally pay only cache-load cost.
        start = perf_counter()
        self.momentum_rhs(vel)
        self.scalar_rhs(scalar, vel)
        self.jit_warmup_seconds = perf_counter() - start
        for key in self.timings:
            self.timings[key] = 0.0

    def momentum_rhs(self, velocity):
        g, c = self.grid, self.config
        start = perf_counter()
        result = []
        for component_axis, a in enumerate(velocity):
            advectors = [g.at_component(velocity, axis, component_axis)
                         for axis in range(3)]
            result.append(momentum_component(a, *advectors,
                                             g.support[component_axis],
                                             g.free[component_axis], g.h, c.nu,
                                             c.momentum_advection == "skew-symmetric"))
        self._add_timing("momentum_seconds", perf_counter() - start)
        return result

    def scalar_rhs(self, scalar, velocity):
        """Conservative donor-cell advection plus centered molecular diffusion.

        No clipping: conservation and scalar bounds are checked independently.
        Scalar numerical diffusion is explicitly a prototype limitation.
        """
        g, c = self.grid, self.config
        start = perf_counter()
        rhs = scalar_rhs_kernel(scalar, *velocity, g.fluid, g.h, c.kappa)
        inlet_x = (velocity[0][0] * np.where(velocity[0][0] >= 0, 0, scalar[0])
                   - 2 * c.kappa * scalar[0] / g.h) * g.inlet
        inlet_y = (velocity[1][:, 0, :] * np.where(velocity[1][:, 0, :] >= 0, 1, scalar[:, 0, :])
                   - 2 * c.kappa * (scalar[:, 0, :] - 1) / g.h) * g.pipe_inlet
        outlet_x = velocity[0][-1] * np.where(velocity[0][-1] >= 0, scalar[-1], 0) * g.outlet
        net_in = (inlet_x.sum() + inlet_y.sum() - outlet_x.sum()) * g.h**2
        self._add_timing("scalar_seconds", perf_counter() - start)
        return rhs, float(net_in)

    def timestep(self):
        c, g, s = self.config, self.grid, self.state
        speed_rate = sum(np.max(np.abs(a)) for a in s.velocity) / g.h
        # 8 rather than 6 covers half-cell Dirichlet diffusion at inlet corners.
        return min(c.dt_max, c.cfl / (speed_rate + 8 * max(c.nu, c.kappa) / g.h**2))

    def _flow_measures(self, velocity):
        max_velocity = max(max(float(a.max()), float(-a.min())) for a in velocity)
        face_energy = 0.5 * self.grid.h**3 * sum(
            float(np.vdot(a.ravel(), a.ravel()).real) for a in velocity)
        return max_velocity, face_energy

    def reset_growth_guard(self):
        """Reset the rolling guard after initialization or checkpoint restore."""
        speed, energy = self._flow_measures(self.state.velocity)
        self._growth_history = deque([(self.state.time, speed, energy)])

    def _check_rapid_growth(self, velocity, time):
        speed, energy = self._flow_measures(velocity)
        history = self._growth_history
        window = self.config.growth_guard_window
        # Keep the last sample immediately before the rolling-window boundary,
        # so the comparison interval is never shorter merely because dt shrank.
        while len(history) > 1 and time - history[1][0] >= window:
            history.popleft()
        base_time, base_speed, base_energy = history[0]
        speed_ratio = speed / max(base_speed, np.finfo(float).tiny)
        energy_ratio = energy / max(base_energy, np.finfo(float).tiny)
        if (time > base_time and
                (speed_ratio > self.config.max_velocity_growth_factor or
                 energy_ratio > self.config.max_energy_growth_factor)):
            raise RuntimeError(
                "Rapid flow growth detected over "
                f"Δt={time - base_time:.4g}: max|u| ratio={speed_ratio:.3g} "
                f"(limit {self.config.max_velocity_growth_factor:g}), "
                f"face-energy ratio={energy_ratio:.3g} "
                f"(limit {self.config.max_energy_growth_factor:g})")
        history.append((time, speed, energy))

    def advance(self, dt=None):
        """SSP-RK3 method-of-lines stages, each followed by a MAC projection."""
        advance_start = perf_counter()
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
            pressure_start = perf_counter()
            pressure = self.projection.apply(result, (1 - base_weight) * dt)
            self.timings["pressure_seconds"] += perf_counter() - pressure_start
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
        self._check_rapid_growth(u3, s.time + dt)
        self.state = State(u3, c3, p3, s.time + dt, s.step + 1,
                           s.scalar_boundary_integral + dt * (q0 / 6 + q1 / 6 + 2 * q2 / 3))
        self.timings["advance_seconds"] += perf_counter() - advance_start
        return dt
