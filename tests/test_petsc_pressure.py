import numpy as np
import pytest

from jetflow.config import Config
from jetflow.grid import Grid
from jetflow.pressure import Projection


def test_petsc_gamg_projection_matches_reference():
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("single-rank equivalence test")

    config = Config(x_min=-2, x_max=3, height=2, half_width=1, pipe_depth=1)
    grid = Grid(config)
    reference = Projection(grid, "pyamg")
    petsc = Projection(grid, "petsc-gamg")
    rng = np.random.default_rng(1402)
    initial = [rng.normal(size=shape.shape) for shape in grid.free]
    grid.boundary_conditions(initial)
    velocity_reference = [a.copy() for a in initial]
    velocity_petsc = [a.copy() for a in initial]

    pressure_reference = reference.apply(velocity_reference, 0.013)
    pressure_petsc = petsc.apply(velocity_petsc, 0.013)

    assert np.max(np.abs(grid.divergence(velocity_petsc)[grid.fluid])) < 1e-7
    np.testing.assert_allclose(pressure_petsc[grid.fluid],
                               pressure_reference[grid.fluid], atol=2e-8)
    for actual, expected in zip(velocity_petsc, velocity_reference):
        np.testing.assert_allclose(actual, expected, atol=2e-9)
