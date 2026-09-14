import numpy as np


class Grid:
    """Uniform MAC grid; circular pipe is explicitly a staircase approximation."""

    def __init__(self, config, fluid=None):
        self.config = c = config.validate()
        self.h = c.h
        self.x = np.arange(round((c.x_max - c.x_min) / c.h)) * c.h + c.x_min + c.h / 2
        self.y = np.arange(round((c.height + c.pipe_depth) / c.h)) * c.h - c.pipe_depth + c.h / 2
        self.z = np.arange(round(2 * c.half_width / c.h)) * c.h - c.half_width + c.h / 2
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.fluid = ((Y > 0) | ((X**2 + Z**2) < 0.25)) if fluid is None else fluid.copy()
        self.shape = self.fluid.shape
        self.wall_j = round(c.pipe_depth / c.h)
        self.ids = np.full(self.shape, -1, dtype=int)
        self.ids[self.fluid] = np.arange(self.fluid.sum())
        self.free, self.support, self.boundary = [], [], []
        for axis in range(3):
            dims = list(self.shape)
            dims[axis] += 1
            a, b = np.zeros(dims, bool), np.zeros(dims, bool)
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            a[tuple(lo)] = self.fluid
            b[tuple(hi)] = self.fluid
            self.free.append(a & b)
            self.support.append(a | b)
            self.boundary.append(a ^ b)
        # Only outlet-normal velocity is pressure-correctable on outer boundaries.
        self.free[0][-1] = self.fluid[-1]
        self.inlet = self.fluid[0]
        self.pipe_inlet = self.fluid[:, 0, :]
        self.outlet = self.fluid[-1]
        # Exact normalization on the discrete pipe cross section.
        r2 = self.x[:, None]**2 + self.z[None, :]**2
        raw = np.maximum(1 - 4 * r2, 0) * self.pipe_inlet
        if raw.sum() == 0:
            raise ValueError("Pipe has no resolved inlet cells")
        self.pipe_profile = raw * c.velocity_ratio * self.pipe_inlet.sum() / raw.sum()
        self.cross_profile = np.where(self.y > 0, 1 - np.exp(-np.log(100) * np.maximum(self.y, 0) / c.delta99), 0)

    def zeros(self):
        return [np.zeros(a.shape) for a in self.free]

    def boundary_conditions(self, velocity, extrapolate_outlet=False):
        u, v, w = velocity
        for component, free in zip(velocity, self.free):
            component[~free] = 0
        u[0] = self.cross_profile[:, None] * self.inlet
        v[:, 0, :] = self.pipe_profile
        if extrapolate_outlet:
            u[-1] = u[-2] * self.outlet
        return velocity

    def centers(self, velocity):
        return np.stack([(a.take(indices=range(a.shape[k] - 1), axis=k)
                          + a.take(indices=range(1, a.shape[k]), axis=k)) / 2
                         for k, a in enumerate(velocity)])

    def divergence(self, velocity):
        return sum(np.diff(a, axis=k) for k, a in enumerate(velocity)) / self.h

    def neighbors(self, value, component_axis, derivative_axis):
        """Tangential no-slip ghosts reflect at a half-cell wall; slip at top/sides.

        Normal velocities on solid interfaces are stored exactly as zero.
        Missing tangential locations inside a solid reflect the current value.
        The outlet has a homogeneous normal derivative for predictor velocities.
        """
        support = self.support[component_axis]
        minus = np.roll(value, 1, axis=derivative_axis)
        plus = np.roll(value, -1, axis=derivative_axis)
        sm = np.roll(support, 1, axis=derivative_axis)
        sp = np.roll(support, -1, axis=derivative_axis)
        minus = np.where(sm, minus, -value)
        plus = np.where(sp, plus, -value)
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[derivative_axis], hi[derivative_axis] = 0, -1
        lo, hi = tuple(lo), tuple(hi)
        # Inlet tangential velocities are zero; pipe inlet likewise.
        minus[lo] = value[lo] if derivative_axis == 2 else -value[lo]
        plus[hi] = value[hi]  # outflow, top slip and spanwise slip
        return minus, plus

    def at_component(self, velocity, source_axis, target_axis):
        if source_axis == target_axis:
            return velocity[source_axis]
        a = velocity[source_axis]
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[source_axis], high[source_axis] = slice(None, -1), slice(1, None)
        center = (a[tuple(low)] + a[tuple(high)]) / 2
        pad = [(0, 0)] * 3
        pad[target_axis] = (1, 1)
        padded = np.pad(center, pad, mode="edge")
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[target_axis], high[target_axis] = slice(None, -1), slice(1, None)
        return (padded[tuple(low)] + padded[tuple(high)]) / 2

