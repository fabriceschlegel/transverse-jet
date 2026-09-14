import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg


class Projection:
    """Exact discrete MAC projection: A = -D G, p=0 at outlet plane.

    All remaining boundaries have fixed normal velocity / Neumann correction.
    Pressure is a nonincremental projection estimate, not a high-order wall-pressure prediction.
    """

    def __init__(self, grid):
        self.grid = g = grid
        n = int(g.fluid.sum())
        diagonal = np.zeros(n)
        rows, cols, data = [], [], []
        for axis in range(3):
            a, b = [slice(None)] * 3, [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            ia, ib = g.ids[tuple(a)], g.ids[tuple(b)]
            valid = (ia >= 0) & (ib >= 0)
            ia, ib = ia[valid], ib[valid]
            np.add.at(diagonal, ia, 1 / g.h**2)
            np.add.at(diagonal, ib, 1 / g.h**2)
            rows.extend([ia, ib]); cols.extend([ib, ia])
            data.extend([np.full(ia.size, -1 / g.h**2)] * 2)
        outlet_ids = g.ids[-1][g.outlet]
        diagonal[outlet_ids] += 2 / g.h**2  # half-cell to pressure boundary
        rows.append(np.arange(n)); cols.append(np.arange(n)); data.append(diagonal)
        self.matrix = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        if np.any(diagonal == 0):
            raise ValueError("Isolated fluid cell in pressure domain")
        try:
            import pyamg
            self.preconditioner = pyamg.smoothed_aggregation_solver(self.matrix, symmetry="symmetric").aspreconditioner(cycle="V")
            self.backend = "CG + smoothed-aggregation multigrid"
        except ImportError:
            self.preconditioner = LinearOperator((n, n), matvec=lambda x: x / diagonal)
            self.backend = "CG + diagonal preconditioner"
        self.guess = np.zeros(n)
        self.iterations = 0

    def gradient(self, pressure):
        g = self.grid
        result = g.zeros()
        for axis, out in enumerate(result):
            interior = [slice(None)] * 3
            interior[axis] = slice(1, -1)
            out[tuple(interior)] = np.diff(pressure, axis=axis) / g.h
            out[~g.free[axis]] = 0
        result[0][-1] = -2 * pressure[-1] / g.h * g.outlet
        return result

    def apply(self, velocity, dt):
        g = self.grid
        rhs = -g.divergence(velocity)[g.fluid] / dt
        self.iterations = 0
        def callback(_):
            self.iterations += 1
        solution, info = cg(self.matrix, rhs, x0=self.guess,
                            rtol=g.config.pressure_rtol, atol=1e-12,
                            maxiter=g.config.pressure_maxiter,
                            M=self.preconditioner, callback=callback)
        if info != 0:
            raise RuntimeError(f"Pressure solve failed (CG info={info}); no unconverged step accepted")
        self.guess = solution
        p = np.zeros(g.shape)
        p[g.fluid] = solution
        for a, grad in zip(velocity, self.gradient(p)):
            a -= dt * grad
        return p

