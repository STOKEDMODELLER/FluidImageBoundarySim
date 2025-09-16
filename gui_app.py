#!/usr/bin/env python3
"""
Main entry point for the Fluid Image Boundary Simulation GUI application.
"""

import sys
import os

# Add the parent directory to the path so we can import fluid_sim
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fluid_sim.gui import SimulationGUI


def main():
    """Main entry point for the GUI application."""
    try:
        app = SimulationGUI()
        app.run()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Application closed")


if __name__ == "__main__":
    main()