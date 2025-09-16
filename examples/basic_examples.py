#!/usr/bin/env python3
"""
Example demonstrating the new fluid simulation structure.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fluid_sim import LBMSimulation, ConfigManager
from fluid_sim.utils import create_dir_if_not_exists


def run_cylinder_flow_example():
    """Run a cylinder flow simulation example."""
    
    print("Running Cylinder Flow Example...")
    
    # Create output directory
    output_dir = "./examples_output/"
    create_dir_if_not_exists(output_dir)
    
    # Create simulation with custom parameters
    simulation = LBMSimulation(
        nx=200,
        ny=100, 
        reynolds=100.0,
        flow_speed=0.04
    )
    
    # Setup cylinder obstacle
    simulation.setup_cylinder_obstacle(
        cx=50,   # x-position of cylinder center
        cy=50,   # y-position of cylinder center  
        r=8      # cylinder radius
    )
    
    print(f"Simulation initialized:")
    print(f"  Grid size: {simulation.nx} x {simulation.ny}")
    print(f"  Reynolds number: {simulation.reynolds}")
    print(f"  Lattice velocity: {simulation.uLB:.6f}")
    print(f"  Relaxation parameter: {simulation.omega:.6f}")
    
    # Run simulation and save snapshots
    max_iterations = 500
    save_interval = 50
    
    velocity_history = []
    pressure_history = []
    
    for i in range(max_iterations):
        # Perform simulation step
        diagnostics = simulation.step()
        
        # Print progress
        if i % save_interval == 0:
            print(f"  Step {i:3d}: "
                  f"Max vel = {diagnostics['max_velocity']:.6f}, "
                  f"Stable = {'Yes' if diagnostics['is_stable'] else 'No'}, "
                  f"Mach = {diagnostics['mach_number']:.6f}")
            
            # Save visualization
            save_visualization(simulation, i, output_dir)
            
        # Store data for analysis
        velocity_history.append(diagnostics['max_velocity'])
        pressure_history.append(diagnostics.get('max_pressure', 0))
    
    # Plot convergence history
    plot_convergence_history(velocity_history, pressure_history, output_dir)
    
    print(f"\nExample completed! Results saved to {output_dir}")
    return simulation


def save_visualization(simulation, step, output_dir):
    """Save visualization of current simulation state."""
    
    # Get field data
    velocity_mag = simulation.get_velocity_magnitude()
    pressure = simulation.get_pressure_field()
    obstacle = simulation.obstacle
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot velocity magnitude
    vel_data = velocity_mag.T
    vel_masked = np.ma.masked_where(obstacle.T, vel_data)
    im1 = ax1.imshow(vel_masked, cmap='jet', origin='lower', aspect='auto')
    ax1.imshow(np.ma.masked_where(~obstacle.T, np.ones_like(vel_data)), 
               cmap='gray', origin='lower', aspect='auto', alpha=0.8)
    ax1.set_title(f'Velocity Magnitude - Step {step}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='Velocity Magnitude')
    
    # Plot pressure
    press_data = pressure.T
    press_masked = np.ma.masked_where(obstacle.T, press_data)
    im2 = ax2.imshow(press_masked, cmap='viridis', origin='lower', aspect='auto')
    ax2.imshow(np.ma.masked_where(~obstacle.T, np.ones_like(press_data)), 
               cmap='gray', origin='lower', aspect='auto', alpha=0.8)
    ax2.set_title(f'Pressure - Step {step}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Pressure')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}cylinder_flow_step_{step:04d}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence_history(velocity_history, pressure_history, output_dir):
    """Plot convergence history."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Velocity convergence
    ax1.plot(velocity_history)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Maximum Velocity')
    ax1.set_title('Velocity Convergence')
    ax1.grid(True)
    
    # Pressure convergence
    ax2.plot(pressure_history)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Maximum Pressure')
    ax2.set_title('Pressure Convergence')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}convergence_history.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_custom_mask_example():
    """Run example with custom obstacle mask."""
    
    print("\nRunning Custom Mask Example...")
    
    try:
        # Try to load existing mask
        simulation = LBMSimulation(nx=250, ny=120, reynolds=300.0)
        simulation.setup_from_mask("./bitmap667.png")
        
        print("  Successfully loaded custom obstacle mask")
        print(f"  Obstacle cells: {np.sum(simulation.obstacle)}")
        
        # Run a few steps
        for i in range(10):
            diagnostics = simulation.step()
            
        print(f"  Final max velocity: {diagnostics['max_velocity']:.6f}")
        
    except FileNotFoundError:
        print("  No custom mask file found, skipping this example")
    except Exception as e:
        print(f"  Custom mask example failed: {e}")


def configuration_example():
    """Demonstrate configuration management."""
    
    print("\nConfiguration Management Example...")
    
    # Create custom configuration
    config_manager = ConfigManager()
    
    # Load default configuration
    config = config_manager.load_config()
    print(f"  Default grid size: {config.nx} x {config.ny}")
    
    # Modify configuration
    config.nx = 150
    config.ny = 80
    config.reynolds = 200.0
    config.flow_speed = 0.06
    
    # Validate configuration
    is_valid = config_manager.validate_config(config)
    print(f"  Configuration valid: {is_valid}")
    
    # Save custom configuration
    config_manager.save_config(config, "examples_output/custom_config.json")
    print("  Custom configuration saved")


if __name__ == "__main__":
    print("=" * 60)
    print("FLUID SIMULATION EXAMPLES")
    print("=" * 60)
    
    # Run examples
    simulation = run_cylinder_flow_example()
    run_custom_mask_example()
    configuration_example()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETED!")
    print("Check the 'examples_output' directory for results.")
    print("\nTo run the GUI application:")
    print("  python gui_app.py")
    print("=" * 60)