"""
Validation and stability checking tools for LBM simulations.
"""
from typing import Dict, Any, Optional
import numpy as np


class ValidationTools:
    """Tools for validating LBM properties and checking stability."""

    @staticmethod
    def validate_lattice_boltzmann_properties(fin: np.ndarray, c: np.ndarray,
                                              t: np.ndarray,
                                              tolerance: float = 1e-10) -> Dict[str, Any]:
        from .lattice import LatticeConstants
        results: Dict[str, Any] = {}

        weight_sum = float(np.sum(t))
        results['weights_sum_to_one'] = abs(weight_sum - 1.0) < tolerance
        results['weight_sum'] = weight_sum

        rho = LatticeConstants.compute_density(fin)
        results['mass_conservation'] = bool(np.allclose(np.sum(fin, axis=0), rho, rtol=tolerance))

        momentum = np.zeros((2, fin.shape[1], fin.shape[2]))
        for i in range(fin.shape[0]):
            momentum[0] += fin[i] * c[i, 0]
            momentum[1] += fin[i] * c[i, 1]
        results['momentum_field'] = momentum
        results['max_momentum_x'] = float(np.max(np.abs(momentum[0])))
        results['max_momentum_y'] = float(np.max(np.abs(momentum[1])))

        expected_velocities = np.array([
            [0, 0], [0, -1], [0, 1], [-1, 0], [-1, -1],
            [-1, 1], [1, 0], [1, -1], [1, 1],
        ])
        results['lattice_structure_valid'] = bool(np.allclose(c, expected_velocities))
        return results

    @staticmethod
    def check_stability_conditions(u: np.ndarray, omega: float,
                                   Ma_max: float = 0.1,
                                   L_char: Optional[float] = None,
                                   U_ref: Optional[float] = None) -> Dict[str, Any]:
        """
        Args:
            u: velocity field (2, nx, ny).
            omega: BGK relaxation parameter.
            Ma_max: maximum Mach number for the stability check.
            L_char: characteristic length (in lattice cells). Caller should pass
                the actual obstacle length (diameter for cylinder, side for
                square, equivalent diameter from the mask, ...). If None we
                fall back to a coarse domain estimate which is *not* a true Re.
            U_ref: reference velocity to use for Re. Defaults to the maximum
                velocity magnitude in the field.
        """
        results: Dict[str, Any] = {}

        cs = 1.0 / np.sqrt(3.0)
        velocity_magnitude = np.sqrt(u[0] ** 2 + u[1] ** 2)
        max_mach = float(np.max(velocity_magnitude) / cs)
        results['max_mach_number'] = max_mach
        results['mach_stable'] = max_mach < Ma_max
        results['sound_speed'] = cs

        results['omega'] = omega
        results['omega_stable'] = bool(0.0 < omega < 2.0)

        max_vel = float(np.max(velocity_magnitude))
        if max_vel > 0.0:
            nu = (1.0 / omega - 0.5) / 3.0
            U = U_ref if U_ref is not None else max_vel
            if L_char is None:
                # No obstacle length supplied — use a domain proxy and flag it.
                L = float(np.sqrt(u.shape[1] * u.shape[2]) / 10.0)
                results['L_char_source'] = 'domain_proxy'
            else:
                L = float(L_char)
                results['L_char_source'] = 'obstacle'
            results['estimated_reynolds'] = U * L / nu
            results['kinematic_viscosity'] = nu
            results['L_char'] = L

        return results
