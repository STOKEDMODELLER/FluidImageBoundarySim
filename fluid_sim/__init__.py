"""
FluidImageBoundarySim - A Lattice Boltzmann Method Fluid Simulation Package

This package provides a comprehensive fluid simulation framework using the
Lattice Boltzmann Method with GUI interface for parameter control and 
real-time visualization.
"""

__version__ = "2.0.0"
__author__ = "STOKEDMODELLER"
__description__ = "Lattice Boltzmann Method Fluid Simulation with GUI"

from .core import LBMSimulation, LatticeConstants
from .utils import ConfigManager

# Try to import GUI components (might not be available in all environments)
try:
    from .gui import SimulationGUI
    __all__ = ['LBMSimulation', 'LatticeConstants', 'SimulationGUI', 'ConfigManager']
except ImportError:
    # GUI not available (e.g., no tkinter)
    SimulationGUI = None
    __all__ = ['LBMSimulation', 'LatticeConstants', 'ConfigManager']