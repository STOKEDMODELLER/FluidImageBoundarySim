"""
Validation and stability checking tools for LBM simulations.
"""
import numpy as np
from typing import Dict, Any


class ValidationTools:
    """Tools for validating LBM properties and checking stability."""
    
    @staticmethod
    def validate_lattice_boltzmann_properties(fin: np.ndarray, c: np.ndarray, 
                                            t: np.ndarray, tolerance: float = 1e-10) -> Dict[str, Any]:
        """
        Validates key mathematical properties of the Lattice Boltzmann Method.
        
        Parameters:
        -----------
        fin : np.ndarray
            Distribution functions of shape (q, nx, ny).
        c : np.ndarray
            Lattice velocities of shape (q, 2).
        t : np.ndarray
            Lattice weights of shape (q,).
        tolerance : float
            Numerical tolerance for validation checks.
        
        Returns:
        --------
        dict
            Dictionary containing validation results and computed properties.
        """
        results = {}
        
        # Check if weights sum to 1
        weight_sum = np.sum(t)
        results['weights_sum_to_one'] = np.abs(weight_sum - 1.0) < tolerance
        results['weight_sum'] = weight_sum
        
        # Check mass conservation (sum of distributions should equal density)
        from .lattice import LatticeConstants
        rho = LatticeConstants.compute_density(fin)
        mass_conservation = np.allclose(np.sum(fin, axis=0), rho, rtol=tolerance)
        results['mass_conservation'] = mass_conservation
        
        # Check momentum conservation
        momentum = np.zeros((2, fin.shape[1], fin.shape[2]))
        for i in range(fin.shape[0]):
            momentum[0] += fin[i] * c[i, 0]
            momentum[1] += fin[i] * c[i, 1]
        
        results['momentum_field'] = momentum
        results['max_momentum_x'] = np.max(np.abs(momentum[0]))
        results['max_momentum_y'] = np.max(np.abs(momentum[1]))
        
        # Validate lattice structure (D2Q9)
        expected_velocities = np.array([
            [0, 0], [0, -1], [0, 1], [-1, 0], [-1, -1], 
            [-1, 1], [1, 0], [1, -1], [1, 1]
        ])
        lattice_structure_valid = np.allclose(c, expected_velocities)
        results['lattice_structure_valid'] = lattice_structure_valid
        
        return results

    @staticmethod
    def check_stability_conditions(u: np.ndarray, omega: float, Ma_max: float = 0.1) -> Dict[str, Any]:
        """
        Checks stability conditions for the Lattice Boltzmann simulation.
        
        Parameters:
        -----------
        u : np.ndarray
            Velocity field of shape (2, nx, ny).
        omega : float
            Relaxation parameter.
        Ma_max : float
            Maximum allowed Mach number for stability.
        
        Returns:
        --------
        dict
            Dictionary containing stability analysis results.
        """
        results = {}
        
        # Check Mach number (velocity should be much less than sound speed)
        cs = 1.0 / np.sqrt(3.0)  # Sound speed in lattice units for D2Q9
        velocity_magnitude = np.sqrt(u[0]**2 + u[1]**2)
        mach_number = velocity_magnitude / cs
        max_mach = np.max(mach_number)
        
        results['max_mach_number'] = max_mach
        results['mach_stable'] = max_mach < Ma_max
        results['sound_speed'] = cs
        
        # Check relaxation parameter stability (should be between 0 and 2)
        results['omega'] = omega
        results['omega_stable'] = 0.0 < omega < 2.0
        
        # Compute Reynolds number based on maximum velocity
        if np.max(velocity_magnitude) > 0:
            nu = (1.0/omega - 0.5) / 3.0  # Kinematic viscosity in lattice units
            # Estimate characteristic length
            L_char = np.sqrt(u.shape[1] * u.shape[2]) / 10.0  # Rough estimate
            Re_estimate = np.max(velocity_magnitude) * L_char / nu
            results['estimated_reynolds'] = Re_estimate
            results['kinematic_viscosity'] = nu
        
        return results