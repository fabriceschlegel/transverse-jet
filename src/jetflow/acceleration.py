"""Numba-parallel explicit stencils for the MAC-grid solver.

These kernels replace allocation-heavy NumPy expressions while preserving the
existing staggering, wall ghosts, inlet/outlet rules, and operation ordering.
The pressure projection is handled independently by the PyAMG or PETSc backend.
"""

import numpy as np
from numba import njit, prange


@njit(inline="always")
def _neighbor(value, support, i, j, k, axis, direction):
    current = value[i, j, k]
    if axis == 0:
        index, extent = i, value.shape[0]
    elif axis == 1:
        index, extent = j, value.shape[1]
    else:
        index, extent = k, value.shape[2]
    if direction < 0 and index == 0:
        # Spanwise outer boundary is slip; the other lower boundaries use the
        # same reflected/no-slip convention as Grid.neighbors.
        return current if axis == 2 else -current
    if direction > 0 and index == extent - 1:
        return current
    if axis == 0:
        ni = int(i + direction)
        return value[ni, j, k] if support[ni, j, k] else -current
    elif axis == 1:
        nj = int(j + direction)
        return value[i, nj, k] if support[i, nj, k] else -current
    nk = int(k + direction)
    return value[i, j, nk] if support[i, j, nk] else -current


@njit(parallel=True, cache=True)
def momentum_component(value, advector_x, advector_y, advector_z,
                       support, free, h, nu, skew_symmetric=False):
    result = np.zeros_like(value)
    for i in prange(value.shape[0]):
        for j in range(value.shape[1]):
            for k in range(value.shape[2]):
                if not free[i, j, k]:
                    continue
                rate = 0.0
                for axis in range(3):
                    minus = _neighbor(value, support, i, j, k, axis, -1)
                    plus = _neighbor(value, support, i, j, k, axis, 1)
                    if axis == 0:
                        advecting = advector_x
                    elif axis == 1:
                        advecting = advector_y
                    else:
                        advecting = advector_z
                    advector = advecting[i, j, k]
                    if skew_symmetric:
                        advector_minus = _neighbor(
                            advecting, support, i, j, k, axis, -1)
                        advector_plus = _neighbor(
                            advecting, support, i, j, k, axis, 1)
                        # Half advective and half conservative form. For a
                        # centered difference with closed/periodic boundaries,
                        # diag(u)D + D diag(u) is skew-adjoint and therefore
                        # contributes no semi-discrete kinetic-energy growth.
                        rate -= (advector * (plus - minus)
                                 + advector_plus * plus
                                 - advector_minus * minus) / (4.0 * h)
                    else:
                        rate -= advector * (plus - minus) / (2.0 * h)
                    rate += nu * (plus - 2.0 * value[i, j, k] + minus) / (h * h)
                result[i, j, k] = rate
    return result


@njit(parallel=True, cache=True)
def scalar_rhs_kernel(scalar, u, v, w, fluid, h, kappa):
    nx, ny, nz = scalar.shape
    result = np.zeros_like(scalar)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if not fluid[i, j, k]:
                    continue
                c = scalar[i, j, k]

                if i == 0:
                    vel = u[0, j, k]
                    fx0 = vel * (0.0 if vel >= 0.0 else c) - 2.0 * kappa * c / h
                elif fluid[i - 1, j, k]:
                    left = scalar[i - 1, j, k]
                    vel = u[i, j, k]
                    fx0 = vel * (left if vel >= 0.0 else c) - kappa * (c - left) / h
                else:
                    fx0 = 0.0
                if i == nx - 1:
                    vel = u[nx, j, k]
                    fx1 = vel * (c if vel >= 0.0 else 0.0)
                elif fluid[i + 1, j, k]:
                    right = scalar[i + 1, j, k]
                    vel = u[i + 1, j, k]
                    fx1 = vel * (c if vel >= 0.0 else right) - kappa * (right - c) / h
                else:
                    fx1 = 0.0

                if j == 0:
                    vel = v[i, 0, k]
                    fy0 = vel * (1.0 if vel >= 0.0 else c) - 2.0 * kappa * (c - 1.0) / h
                elif fluid[i, j - 1, k]:
                    low = scalar[i, j - 1, k]
                    vel = v[i, j, k]
                    fy0 = vel * (low if vel >= 0.0 else c) - kappa * (c - low) / h
                else:
                    fy0 = 0.0
                if j < ny - 1 and fluid[i, j + 1, k]:
                    high = scalar[i, j + 1, k]
                    vel = v[i, j + 1, k]
                    fy1 = vel * (c if vel >= 0.0 else high) - kappa * (high - c) / h
                else:
                    fy1 = 0.0

                if k > 0 and fluid[i, j, k - 1]:
                    back = scalar[i, j, k - 1]
                    vel = w[i, j, k]
                    fz0 = vel * (back if vel >= 0.0 else c) - kappa * (c - back) / h
                else:
                    fz0 = 0.0
                if k < nz - 1 and fluid[i, j, k + 1]:
                    front = scalar[i, j, k + 1]
                    vel = w[i, j, k + 1]
                    fz1 = vel * (c if vel >= 0.0 else front) - kappa * (front - c) / h
                else:
                    fz1 = 0.0

                result[i, j, k] = -(fx1 - fx0 + fy1 - fy0 + fz1 - fz0) / h
    return result
