import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg


class Projection:
    """Exact discrete MAC projection: A = -D G, p=0 at outlet plane.

    All remaining boundaries have fixed normal velocity / Neumann correction.
    Pressure is a nonincremental projection estimate, not a high-order wall-pressure prediction.
    """

    def __init__(self, grid, backend="pyamg"):
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
        self.backend_name = backend
        self.rank, self.mpi_size = 0, 1
        if backend == "pyamg":
            try:
                import pyamg
                self.preconditioner = pyamg.smoothed_aggregation_solver(
                    self.matrix, symmetry="symmetric").aspreconditioner(cycle="V")
                self.backend = "SciPy CG + PyAMG smoothed aggregation"
            except ImportError:
                self.preconditioner = LinearOperator((n, n), matvec=lambda x: x / diagonal)
                self.backend = "SciPy CG + diagonal preconditioner"
            self.guess = np.zeros(n)
        elif backend in {"petsc-gamg", "petsc-hypre"}:
            self._setup_petsc(backend, n)
        else:
            raise ValueError(f"Unknown pressure backend: {backend}")
        self.iterations = 0

    def _setup_petsc(self, backend, n):
        try:
            from petsc4py import PETSc
        except ImportError as exc:
            raise RuntimeError(
                "PETSc backend requires petsc4py built against an MPI-enabled PETSc") from exc
        self.PETSc = PETSc
        comm = PETSc.COMM_WORLD
        self.rank, self.mpi_size = comm.getRank(), comm.getSize()
        matrix = PETSc.Mat().createAIJ(size=(n, n), nnz=7, comm=comm)
        start, end = matrix.getOwnershipRange()
        source = self.matrix
        first, last = source.indptr[start], source.indptr[end]
        matrix.setValuesCSR(source.indptr[start:end + 1] - first,
                            source.indices[first:last], source.data[first:last])
        matrix.assemblyBegin(); matrix.assemblyEnd()
        self.petsc_matrix = matrix
        self.rhs = matrix.createVecLeft()
        self.solution = matrix.createVecRight()
        self.scatter, self.solution_all = PETSc.Scatter.toAll(self.solution)
        self.local_start, self.local_end = start, end

        ksp = PETSc.KSP().create(comm=comm)
        ksp.setOptionsPrefix("jet_pressure_")
        ksp.setOperators(matrix)
        ksp.setType(PETSc.KSP.Type.CG)
        pc = ksp.getPC()
        if backend == "petsc-gamg":
            pc.setType(PETSc.PC.Type.GAMG)
            label = "PETSc CG + GAMG"
        else:
            if not PETSc.Sys.hasExternalPackage("hypre"):
                raise RuntimeError("This PETSc build does not include hypre; use petsc-gamg")
            pc.setType(PETSc.PC.Type.HYPRE)
            pc.setHYPREType("boomeramg")
            label = "PETSc CG + hypre BoomerAMG"
        ksp.setTolerances(rtol=self.grid.config.pressure_rtol, atol=1e-12,
                          max_it=self.grid.config.pressure_maxiter)
        ksp.setInitialGuessNonzero(True)
        ksp.setFromOptions()
        ksp.setUp()
        self.ksp = ksp
        self.backend = f"{label} ({self.mpi_size} MPI ranks)"
        # PETSc owns the distributed matrix from here; release the replicated
        # SciPy copy after setup to reduce steady-state memory on every rank.
        self.matrix = None

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
        if self.backend_name == "pyamg":
            def callback(_):
                self.iterations += 1
            solution, info = cg(self.matrix, rhs, x0=self.guess,
                                rtol=g.config.pressure_rtol, atol=1e-12,
                                maxiter=g.config.pressure_maxiter,
                                M=self.preconditioner, callback=callback)
            if info != 0:
                raise RuntimeError(f"Pressure solve failed (CG info={info}); no unconverged step accepted")
            self.guess = solution
        else:
            local_rhs = self.rhs.getArray()
            local_rhs[:] = rhs[self.local_start:self.local_end]
            self.ksp.solve(self.rhs, self.solution)
            self.iterations = self.ksp.getIterationNumber()
            reason = self.ksp.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(
                    f"PETSc pressure solve failed (reason={reason}); no unconverged step accepted")
            self.scatter.scatter(self.solution, self.solution_all,
                                 addv=self.PETSc.InsertMode.INSERT_VALUES,
                                 mode=self.PETSc.ScatterMode.FORWARD)
            solution = self.solution_all.getArray(readonly=True).copy()
        p = np.zeros(g.shape)
        p[g.fluid] = solution
        for a, grad in zip(velocity, self.gradient(p)):
            a -= dt * grad
        return p
