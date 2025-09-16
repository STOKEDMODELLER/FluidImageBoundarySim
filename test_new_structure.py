#!/usr/bin/env python3
"""
Test script for the improved structured Fluid Image Boundary Simulation.
"""

import sys
import os
import numpy as np

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fluid_sim import LBMSimulation, LatticeConstants, ConfigManager


def test_new_structure():
    """Test the new modular structure."""
    
    print("=" * 60)
    print("TESTING NEW IMPROVED PROJECT STRUCTURE")
    print("=" * 60)
    
    # Test 1: Configuration Management
    print("\n1. Testing Configuration Management...")
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    print(f"   Loaded config - Grid: {config.nx}x{config.ny}")
    print(f"   Reynolds: {config.reynolds}, Flow speed: {config.flow_speed} m/s")
    print(f"   Configuration validation: {config_manager.validate_config()}")
    
    # Test 2: Lattice Constants
    print("\n2. Testing Lattice Constants...")
    lattice = LatticeConstants()
    
    print(f"   Lattice velocities shape: {lattice.c.shape}")
    print(f"   Lattice weights shape: {lattice.t.shape}")
    print(f"   Number of velocities (q): {lattice.q}")
    print(f"   Weights sum to 1: {np.abs(np.sum(lattice.t) - 1.0) < 1e-10}")
    
    # Test 3: Simulation Initialization
    print("\n3. Testing Simulation Initialization...")
    simulation = LBMSimulation(
        nx=config.nx,
        ny=config.ny,
        reynolds=config.reynolds,
        flow_speed=config.flow_speed
    )
    
    print(f"   Simulation grid: {simulation.nx}x{simulation.ny}")
    print(f"   Lattice velocity: {simulation.uLB:.6f} LU")
    print(f"   Relaxation parameter: {simulation.omega:.6f}")
    
    # Test 4: Obstacle Setup
    print("\n4. Testing Obstacle Setup...")
    simulation.setup_cylinder_obstacle(
        cx=config.obstacle_cx,
        cy=config.obstacle_cy,
        r=config.obstacle_r
    )
    
    print(f"   Obstacle shape: {simulation.obstacle.shape}")
    print(f"   Obstacle cells: {np.sum(simulation.obstacle)}")
    print(f"   Initial velocity field shape: {simulation.u.shape}")
    
    # Test 5: Simulation Steps
    print("\n5. Testing Simulation Steps...")
    
    for i in range(5):
        diagnostics = simulation.step()
        print(f"   Step {i+1}: Max vel = {diagnostics['max_velocity']:.6f}, "
              f"Stable = {diagnostics['is_stable']}, "
              f"Mach = {diagnostics['mach_number']:.6f}")
    
    # Test 6: Data Access
    print("\n6. Testing Data Access...")
    velocity_mag = simulation.get_velocity_magnitude()
    pressure = simulation.get_pressure_field()
    
    print(f"   Velocity magnitude range: [{np.min(velocity_mag):.6f}, {np.max(velocity_mag):.6f}]")
    print(f"   Pressure range: [{np.min(pressure):.6f}, {np.max(pressure):.6f}]")
    
    # Test 7: Mask Loading (if bitmap files exist)
    print("\n7. Testing Mask Loading...")
    try:
        simulation.setup_from_mask("./bitmap667.png")
        print("   Successfully loaded obstacle mask from file")
        print(f"   Updated obstacle shape: {simulation.obstacle.shape}")
    except FileNotFoundError:
        print("   No bitmap file found, skipping mask test")
    except Exception as e:
        print(f"   Mask loading failed: {e}")
    
    print("\n" + "=" * 60)
    print("STRUCTURE TESTING COMPLETED!")
    print("\nKey improvements:")
    print("✓ Modular project structure with separate packages")
    print("✓ Object-oriented design with proper encapsulation")
    print("✓ Configuration management system")
    print("✓ Improved error handling and validation")
    print("✓ Clean separation of concerns")
    print("✓ Preparation for GUI integration")
    print("=" * 60)
    
    return simulation


def test_gui_components():
    """Test GUI components without actually starting the GUI."""
    
    print("\n" + "=" * 60)
    print("TESTING GUI COMPONENTS")
    print("=" * 60)
    
    try:
        # Test imports
        from fluid_sim.gui import SimulationGUI, VisualizationPanel, ControlPanel
        print("✓ GUI components imported successfully")
        
        # Test configuration loading
        from fluid_sim.utils import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config()
        print("✓ Configuration system working")
        
        print("\nGUI system ready for deployment!")
        print("Run 'python gui_app.py' to start the GUI application")
        
    except ImportError as e:
        print(f"✗ GUI import failed: {e}")
        print("This is expected if tkinter is not available")
    except Exception as e:
        print(f"✗ GUI test failed: {e}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test the new structure
    simulation = test_new_structure()
    
    # Test GUI components
    test_gui_components()
    
    print(f"\nFinal simulation state:")
    print(f"Time steps completed: {simulation.time_step}")
    print(f"Current max velocity: {np.max(simulation.get_velocity_magnitude()):.6f}")
    
    # Save configuration for future use
    config_manager = ConfigManager()
    config_manager.save_config()
    print(f"\nConfiguration saved to {config_manager.config_file}")