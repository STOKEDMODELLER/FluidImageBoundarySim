"""
Lattice Boltzmann Method constants and utilities (D2Q9).
"""
from typing import List, Tuple
import numpy as np


class LatticeConstants:
    """Encapsulates D2Q9 lattice constants for Lattice Boltzmann Method."""

    # Speed of sound squared in lattice units for the standard D2Q9 lattice.
    CS2 = 1.0 / 3.0

    def __init__(self):
        (self._c, self._t, self._noslip,
         self._i1, self._i2, self._i3,
         self._y_mirror,
         self._cy_pos, self._cy_neg) = self._get_lattice_constants()

    @property
    def c(self) -> np.ndarray:
        """Lattice velocity vectors of shape (9, 2)."""
        return self._c

    @property
    def t(self) -> np.ndarray:
        """Lattice weights of shape (9,)."""
        return self._t

    @property
    def noslip(self) -> List[int]:
        """Opposite-velocity indices for full bounce-back (c -> -c)."""
        return self._noslip

    @property
    def i1(self) -> List[int]:
        """Indices of populations with c_x < 0 (unknown on the right wall)."""
        return self._i1

    @property
    def i2(self) -> List[int]:
        """Indices of populations with c_x = 0 (tangential to left/right walls)."""
        return self._i2

    @property
    def i3(self) -> List[int]:
        """Indices of populations with c_x > 0 (unknown on the left wall)."""
        return self._i3

    @property
    def y_mirror(self) -> List[int]:
        """Index map for free-slip in y: (cx, cy) -> (cx, -cy)."""
        return self._y_mirror

    @property
    def cy_pos(self) -> List[int]:
        """Indices of populations with c_y > 0 (unknown on the y=0 wall under free-slip)."""
        return self._cy_pos

    @property
    def cy_neg(self) -> List[int]:
        """Indices of populations with c_y < 0 (unknown on the y=ny-1 wall under free-slip)."""
        return self._cy_neg

    @property
    def q(self) -> int:
        """Number of lattice velocities (9 for D2Q9)."""
        return self._c.shape[0]

    def _get_lattice_constants(self):
        c = np.array([(x, y) for x in [0, -1, 1] for y in [0, -1, 1]])
        q = c.shape[0]

        t = (1.0 / 36.0) * np.ones(q)
        norm_ci = np.array([np.linalg.norm(ci) for ci in c])
        t[norm_ci < 1.1] = 1.0 / 9.0
        t[0] = 4.0 / 9.0

        noslip = [int(np.where((c == -c[i]).all(axis=1))[0][0]) for i in range(q)]
        y_mirror = [int(np.where((c == c[i] * np.array([1, -1])).all(axis=1))[0][0])
                    for i in range(q)]

        i1 = np.where(c[:, 0] < 0)[0].tolist()
        i2 = np.where(c[:, 0] == 0)[0].tolist()
        i3 = np.where(c[:, 0] > 0)[0].tolist()

        cy_pos = np.where(c[:, 1] > 0)[0].tolist()
        cy_neg = np.where(c[:, 1] < 0)[0].tolist()

        return c, t, noslip, i1, i2, i3, y_mirror, cy_pos, cy_neg

    @staticmethod
    def physical_to_lattice_velocity(u_phys: float, dx: float, dt: float) -> float:
        """
        Convert a physical velocity (m/s) to lattice units.

        u_LU = u_phys * (dt / dx). This is only meaningful when ``dx`` and ``dt``
        are real physical scales chosen by the user. The previous default of
        ``dx = (ny-1)/(ny-1) = 1`` made the conversion a no-op; callers must now
        supply real values or pass a lattice velocity directly.
        """
        if dx <= 0.0 or dt <= 0.0:
            raise ValueError(f"dx and dt must be positive (got dx={dx}, dt={dt})")
        return u_phys * (dt / dx)

    # Backward-compatibility alias. Prefer ``physical_to_lattice_velocity``.
    @staticmethod
    def mps_to_lu(flow_speed: float, dx: float, dt: float = 1.0) -> float:
        return LatticeConstants.physical_to_lattice_velocity(flow_speed, dx, dt)

    @staticmethod
    def compute_density(fin: np.ndarray) -> np.ndarray:
        """Macroscopic density rho(x, y) = sum_i f_i(x, y)."""
        return np.sum(fin, axis=0)

    def equilibrium_distribution(self, nx: int, ny: int, rho: np.ndarray,
                                 u: np.ndarray) -> np.ndarray:
        """Maxwell-Boltzmann equilibrium expanded to second order in Mach."""
        assert u.shape[0] == 2, "Velocity field must have shape (2, nx, ny)"

        cu = 3.0 * np.dot(self.c, u.transpose(1, 0, 2))
        usqr = (3.0 / 2.0) * (u[0] ** 2 + u[1] ** 2)

        feq = np.zeros((self.q, nx, ny))
        for i in range(self.q):
            feq[i, :, :] = rho * self.t[i] * (1.0 + cu[i] + 0.5 * cu[i] ** 2 - usqr)

        return feq
