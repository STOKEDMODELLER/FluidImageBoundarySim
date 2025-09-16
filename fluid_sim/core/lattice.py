"""
Lattice Boltzmann Method constants and utilities.
"""
from typing import List, Tuple
import numpy as np


class LatticeConstants:
    """Encapsulates D2Q9 lattice constants for Lattice Boltzmann Method."""
    
    def __init__(self):
        """Initialize lattice constants for D2Q9 model."""
        self._c, self._t, self._noslip, self._i1, self._i2, self._i3 = self._get_lattice_constants()
    
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
        """Opposite velocity indices for bounce-back."""
        return self._noslip
    
    @property
    def i1(self) -> List[int]:
        """Indices for unknown velocities on the right wall."""
        return self._i1
    
    @property
    def i2(self) -> List[int]:
        """Indices for unknown velocities in the vertical middle."""
        return self._i2
    
    @property
    def i3(self) -> List[int]:
        """Indices for unknown velocities on the left wall."""
        return self._i3
    
    @property
    def q(self) -> int:
        """Number of lattice velocities (9 for D2Q9)."""
        return self._c.shape[0]
    
    def _get_lattice_constants(self) -> Tuple[np.ndarray, np.ndarray, List[int], List[int], List[int], List[int]]:
        """
        Returns the lattice constants for the D2Q9 Lattice Boltzmann Method.
        
        Returns:
        - c: np.ndarray of shape (q, 2) - Lattice velocity vectors
        - t: np.ndarray of shape (q,) - Lattice weights  
        - noslip: List of opposite velocity indices for bounce-back
        - i1, i2, i3: Boundary condition index arrays
        """
        c = np.array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) # Lattice velocities.
        q = c.shape[0]
        
        t = 1./36. * np.ones(q)  # Lattice weights.
        norm_ci = np.array([np.linalg.norm(ci) for ci in c])
        t[norm_ci < 1.1] = 1./9.
        t[0] = 4./9.
        
        noslip = [np.where((c == -c[i]).all(axis=1))[0][0] for i in range(q)]
        
        i1 = np.where(c[:, 0] < 0)[0].tolist()
        i2 = np.where(c[:, 0] == 0)[0].tolist()
        i3 = np.where(c[:, 0] > 0)[0].tolist()
        
        return c, t, noslip, i1, i2, i3
    
    @staticmethod
    def mps_to_lu(flow_speed: float, dx: float, dt: float = 1.0) -> float:
        """
        Converts a flow velocity from meters per second to lattice units.
        
        Args:
        - flow_speed: float - Flow velocity in meters per second
        - dx: float - Lattice spacing in meters
        - dt: float - Time step in seconds (default=1.0)
        
        Returns:
        - flow_speed_lu: float - Flow velocity in lattice units
        """
        return flow_speed * (dt / dx)
    
    @staticmethod
    def compute_density(fin: np.ndarray) -> np.ndarray:
        """
        Helper function to compute the density from the distribution function.
        
        Args:
        - fin: np.ndarray of shape (q, nx, ny) - Distribution functions
        
        Returns:
        - rho: np.ndarray of shape (nx, ny) - Computed density
        """
        return np.sum(fin, axis=0)
    
    def equilibrium_distribution(self, nx: int, ny: int, rho: np.ndarray, 
                               u: np.ndarray) -> np.ndarray:
        """
        Computes the equilibrium distribution function for the LBM.
        
        Args:
        - nx: int - Number of grid points in x direction
        - ny: int - Number of grid points in y direction
        - rho: np.ndarray of shape (nx, ny) - The density field
        - u: np.ndarray of shape (2, nx, ny) - The velocity field
        
        Returns:
        - feq: np.ndarray of shape (q, nx, ny) - Equilibrium distribution functions
        """
        # Validate inputs
        assert u.shape[0] == 2, "Velocity field must have shape (2, nx, ny)"
        
        # Compute dot product of lattice velocities with macroscopic velocity
        cu = 3.0 * np.dot(self.c, u.transpose(1, 0, 2))
        
        # Compute velocity magnitude squared
        usqr = 3.0/2.0 * (u[0]**2 + u[1]**2)
        
        # Initialize equilibrium distribution
        feq = np.zeros((self.q, nx, ny))
        
        # Compute equilibrium distribution for each lattice direction
        for i in range(self.q): 
            feq[i, :, :] = rho * self.t[i] * (1.0 + cu[i] + 0.5*cu[i]**2 - usqr)
        
        return feq