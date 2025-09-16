"""
Main LBM simulation class with improved structure and GUI support.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
from .lattice import LatticeConstants
from .validation import ValidationTools


class LBMSimulation:
    """Main Lattice Boltzmann Method simulation class."""
    
    def __init__(self, nx: int, ny: int, reynolds: float = 300.0, 
                 flow_speed: float = 0.05, omega: Optional[float] = None):
        """
        Initialize LBM simulation.
        
        Args:
            nx: Grid size in x direction
            ny: Grid size in y direction  
            reynolds: Reynolds number
            flow_speed: Physical flow speed in m/s
            omega: Relaxation parameter (auto-calculated if None)
        """
        self.nx = nx
        self.ny = ny
        self.reynolds = reynolds
        self.flow_speed = flow_speed
        self.ly = ny - 1.0
        
        # Initialize lattice constants
        self.lattice = LatticeConstants()
        self.validation = ValidationTools()
        
        # Calculate simulation parameters
        self.dx = self.ly / (ny - 1)
        self.uLB = LatticeConstants.mps_to_lu(flow_speed, self.dx, dt=1.0)
        
        # Calculate relaxation parameter if not provided
        if omega is None:
            nu_lb = self.uLB * (ny/9) / reynolds
            self.omega = 1.0 / (3.0 * nu_lb + 0.5)
        else:
            self.omega = omega
            
        # Initialize fields
        self.rho = 1.0
        self.fin = None
        self.u = None
        self.pressure = None
        self.obstacle = None
        
        # Simulation state
        self.time_step = 0
        self.is_running = False
        
    def setup_cylinder_obstacle(self, cx: float, cy: float, r: float, 
                               epsilon: float = 1e-4) -> None:
        """
        Setup cylinder obstacle and initial velocity field.
        
        Args:
            cx: x-coordinate of cylinder center
            cy: y-coordinate of cylinder center  
            r: cylinder radius
            epsilon: perturbation magnitude
        """
        # Create rectangular obstacle (can be modified for circle)
        self.obstacle = self._create_obstacle("rectangle", int(cx), int(cy), int(r))
        
        # Setup velocity inlet with perturbation
        vel = np.fromfunction(
            lambda d, x, y: (1-d) * self.uLB * (1.0 + epsilon * np.sin(y/self.ly * 2 * np.pi)), 
            (2, self.nx, self.ny)
        )
        
        # Create uniform density field
        rho_field = self.rho * np.ones((self.nx, self.ny))
        
        # Compute equilibrium distribution
        feq = self.lattice.equilibrium_distribution(self.nx, self.ny, rho_field, vel)
        self.fin = feq.copy()
        self.u = vel
        
    def setup_from_mask(self, mask_file: str, scale: float = 1.0) -> None:
        """
        Setup simulation from obstacle mask file.
        
        Args:
            mask_file: Path to PNG mask file
            scale: Scaling factor for mask
        """
        from ..utils.obstacles import load_mask
        self.obstacle = load_mask(mask_file, scale).transpose()
        
        # Setup initial velocity field
        vel = np.zeros((2, self.nx, self.ny))
        vel[0, :, :] = self.uLB  # Initial x-velocity
        
        # Create uniform density field
        rho_field = self.rho * np.ones((self.nx, self.ny))
        
        # Compute equilibrium distribution
        feq = self.lattice.equilibrium_distribution(self.nx, self.ny, rho_field, vel)
        self.fin = feq.copy()
        self.u = vel
        
    def step(self) -> Dict[str, Any]:
        """
        Perform one simulation step.
        
        Returns:
            Dictionary with simulation diagnostics
        """
        if self.fin is None:
            raise RuntimeError("Simulation not initialized. Call setup_* method first.")
            
        # Perform LBM step
        self.fin, self.u, rho, feq, fout, self.pressure = self._compute_fluid_flow()
        self.time_step += 1
        
        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics()
        
        return diagnostics
    
    def run(self, max_iterations: int, callback=None) -> None:
        """
        Run simulation for specified iterations.
        
        Args:
            max_iterations: Maximum number of iterations
            callback: Optional callback function called each step
        """
        self.is_running = True
        
        for i in range(max_iterations):
            if not self.is_running:
                break
                
            diagnostics = self.step()
            
            if callback:
                callback(self, diagnostics)
                
    def stop(self) -> None:
        """Stop the running simulation."""
        self.is_running = False
        
    def get_velocity_magnitude(self) -> np.ndarray:
        """Get velocity magnitude field."""
        if self.u is None:
            return np.zeros((self.nx, self.ny))
        return np.sqrt(self.u[0]**2 + self.u[1]**2)
    
    def get_pressure_field(self) -> np.ndarray:
        """Get pressure field."""
        if self.pressure is None:
            return np.zeros((self.nx, self.ny))
        return self.pressure
    
    def _create_obstacle(self, shape: str, cx: int, cy: int, r: int) -> np.ndarray:
        """Create geometric obstacle."""
        if shape == "circle":
            return np.fromfunction(lambda x,y: (x-cx)**2+(y-cy)**2<r**2, (self.nx, self.ny))
        elif shape == "rectangle":
            return np.fromfunction(
                lambda x,y: (x>cx-r/2) & (x<cx+r/2) & (y>cy-r/2) & (y<cy+r/2), 
                (self.nx, self.ny)
            )
        else:
            raise ValueError(f"Invalid obstacle shape: {shape}")
            
    def _compute_fluid_flow(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Core LBM computation step."""
        fin = self.fin
        
        # Set reflective boundary conditions
        fin[:, 0, :] = fin[:, 1, :]     # top boundary
        fin[:, -1, :] = fin[:, -2, :]   # bottom boundary
        
        # Right wall: outflow condition
        fin[self.lattice.i1, -1, :] = fin[self.lattice.i1, -2, :] 
        
        # Calculate macroscopic density and velocity
        rho = LatticeConstants.compute_density(fin)
        u = np.dot(self.lattice.c.transpose(), fin.transpose((1, 0, 2))) / rho

        # Left wall: apply inlet velocity boundary condition
        u[:, 0, :] = self.u[:, 0, :]
        
        # Left wall: compute density using Zou/He boundary condition
        denominator = 1.0 - u[0, 0, :]
        safe_denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        rho[0, :] = (1.0 / safe_denominator) * (
            LatticeConstants.compute_density(fin[self.lattice.i2, 0, :]) + 
            2.0 * LatticeConstants.compute_density(fin[self.lattice.i1, 0, :])
        )

        # Compute equilibrium distribution
        feq = self.lattice.equilibrium_distribution(self.nx, self.ny, rho, u)
        
        # Left wall: Zou/He boundary condition for unknown distributions
        fin[self.lattice.i3, 0, :] = fin[self.lattice.i1, 0, :] + feq[self.lattice.i3, 0, :] - fin[self.lattice.i1, 0, :]
        
        # Collision step (BGK approximation)
        fout = fin - self.omega * (fin - feq)
        
        # Apply bounce-back boundary condition at obstacles
        for i in range(self.lattice.q): 
            fout[i, self.obstacle] = fin[self.lattice.noslip[i], self.obstacle]
        
        # Streaming step
        for i in range(self.lattice.q): 
            fin[i, :, :] = np.roll(np.roll(fout[i, :, :], self.lattice.c[i, 0], axis=0), 
                                 self.lattice.c[i, 1], axis=1)
        
        # Calculate pressure
        rho = LatticeConstants.compute_density(fin)
        cs_squared = 1.0/3.0  # Speed of sound squared in lattice units
        pressure = rho * cs_squared
        
        return fin, u, rho, feq, fout, pressure
    
    def _calculate_diagnostics(self) -> Dict[str, Any]:
        """Calculate simulation diagnostics."""
        if self.u is None:
            return {}
            
        vel_mag = self.get_velocity_magnitude()
        max_vel = np.max(vel_mag)
        
        # Stability analysis
        stability = self.validation.check_stability_conditions(self.u, self.omega)
        
        return {
            'time_step': self.time_step,
            'max_velocity': max_vel,
            'max_pressure': np.max(self.pressure) if self.pressure is not None else 0,
            'min_pressure': np.min(self.pressure) if self.pressure is not None else 0,
            'mach_number': stability.get('max_mach_number', 0),
            'is_stable': stability.get('mach_stable', False) and stability.get('omega_stable', False),
            'reynolds_estimate': stability.get('estimated_reynolds', 0)
        }