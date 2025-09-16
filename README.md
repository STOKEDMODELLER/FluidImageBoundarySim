# Fluid Image Boundary Simulation v2.0.0 - GUI Edition

A comprehensive Lattice Boltzmann Method (LBM) fluid simulation package with an intuitive GUI interface for real-time visualization and parameter control.

## 🚀 Major Improvements in v2.0.0

### New GUI Interface
- **Interactive Parameter Control**: Real-time adjustment of simulation parameters
- **Multi-view Visualization**: Separate tabs for velocity, pressure, and combined views
- **Live Simulation Monitoring**: Real-time diagnostics and stability analysis
- **Configuration Management**: Save and load simulation configurations
- **Obstacle Mask Loading**: Import custom obstacle geometries from PNG files

### Improved Project Structure
- **Modular Design**: Clean separation of core simulation, GUI, and utilities
- **Object-Oriented Architecture**: Proper encapsulation and reusable components
- **Configuration System**: JSON-based configuration management
- **Enhanced Error Handling**: Robust validation and error reporting
- **Package Installation**: Proper Python package with setup.py

### Enhanced Simulation Features
- **Stability Analysis**: Real-time Mach number and Reynolds number monitoring
- **Mathematical Validation**: Built-in checks for mass and momentum conservation
- **Flexible Obstacle Creation**: Support for circles, rectangles, and custom masks
- **Improved Boundary Conditions**: Enhanced Zou-He implementation with numerical stability

## 📁 Project Structure

```
FluidImageBoundarySim/
├── fluid_sim/                 # Main package
│   ├── core/                  # Core simulation logic
│   │   ├── lattice.py         # Lattice constants and methods
│   │   ├── simulation.py      # Main LBM simulation class
│   │   └── validation.py      # Validation and stability tools
│   ├── gui/                   # GUI interface
│   │   ├── main_window.py     # Main application window
│   │   ├── visualization.py   # Visualization panels
│   │   └── controls.py        # Parameter control panels
│   ├── utils/                 # Utility functions
│   │   ├── config.py          # Configuration management
│   │   ├── obstacles.py       # Obstacle creation tools
│   │   └── file_utils.py      # File operations
│   └── config/                # Configuration files
├── examples/                  # Example scripts
├── tests/                     # Test files
├── docs/                      # Documentation
├── gui_app.py                 # GUI application entry point
├── test_new_structure.py      # Structure validation tests
├── simulation_config.json     # Default configuration
├── requirements.txt           # Python dependencies
└── setup.py                   # Package installation script
```

## 🛠️ Installation

### Option 1: Direct Installation
```bash
# Clone the repository
git clone https://github.com/STOKEDMODELLER/FluidImageBoundarySim.git
cd FluidImageBoundarySim

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Option 2: Development Installation
```bash
# Clone and install in development mode
git clone https://github.com/STOKEDMODELLER/FluidImageBoundarySim.git
cd FluidImageBoundarySim
pip install -e .[dev]
```

## 🎮 Usage

### GUI Application
Launch the interactive GUI application:
```bash
python gui_app.py
```

### Command Line Interface
```python
from fluid_sim import LBMSimulation, ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config()

# Create simulation
sim = LBMSimulation(
    nx=config.nx, 
    ny=config.ny,
    reynolds=config.reynolds,
    flow_speed=config.flow_speed
)

# Setup obstacle
sim.setup_cylinder_obstacle(
    cx=config.obstacle_cx,
    cy=config.obstacle_cy, 
    r=config.obstacle_r
)

# Run simulation
for i in range(1000):
    diagnostics = sim.step()
    if i % 100 == 0:
        print(f"Step {i}: Max velocity = {diagnostics['max_velocity']:.4f}")
```

### Custom Obstacle from Image
```python
# Load obstacle from PNG file
sim.setup_from_mask("path/to/obstacle_mask.png")

# Or create geometric obstacles
from fluid_sim.utils import create_obstacle
obstacle = create_obstacle("circle", nx=250, ny=120, cx=60, cy=60, r=20)
```

## 🎯 GUI Features

### Real-time Controls
- **Grid Parameters**: Adjust simulation domain size
- **Physical Parameters**: Modify Reynolds number and flow speed
- **Obstacle Parameters**: Change obstacle position and size
- **Simulation Controls**: Start/stop, single step, reset

### Visualization Modes
- **Velocity Tab**: Real-time velocity magnitude visualization
- **Pressure Tab**: Pressure field visualization  
- **Combined Tab**: Side-by-side velocity and pressure plots

### Advanced Features
- **Configuration Management**: Save/load simulation setups
- **Stability Monitoring**: Real-time stability analysis
- **Export Capabilities**: Save visualization images
- **Diagnostics Panel**: Live simulation metrics

## 📊 Configuration Management

### JSON Configuration Format
```json
{
  "nx": 250,
  "ny": 120,
  "reynolds": 300.0,
  "flow_speed": 0.05,
  "max_iterations": 5000,
  "omega": null,
  "obstacle_cx": 62.5,
  "obstacle_cy": 60.0,
  "obstacle_r": 13.3,
  "colormap": "jet",
  "dpi": 100
}
```

### Configuration Usage
```python
from fluid_sim.utils import ConfigManager

# Load existing configuration
config_manager = ConfigManager()
config = config_manager.load_config("my_config.json")

# Modify parameters
config_manager.update_config(reynolds=500.0, flow_speed=0.08)

# Save configuration
config_manager.save_config(config, "updated_config.json")
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Test new structure
python test_new_structure.py

# Test original functionality (legacy)
python test_improvements.py
```

## 📋 Mathematical Foundation

The simulation implements the **D2Q9 Lattice Boltzmann Method** with:

### Core Equations
- **Equilibrium Distribution**: Maxwell-Boltzmann expanded to second order
- **Collision Operator**: BGK approximation with single relaxation time
- **Streaming Step**: Advection of distribution functions
- **Boundary Conditions**: Zou-He implementation for velocity/pressure boundaries

### Key Physical Properties
- **Sound Speed**: cs = 1/√3 in lattice units
- **Kinematic Viscosity**: ν = (1/ω - 0.5)/3
- **Pressure**: P = ρ × cs² (proper equation of state)
- **Stability Condition**: Mach number < 0.1 for numerical stability

## 🔧 Requirements

### Core Dependencies
- Python >= 3.8
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- pillow >= 8.0.0
- scipy >= 1.6.0

### Optional Dependencies
- opencv-python >= 4.5.0 (for video output)
- tkinter (usually included with Python)

## 📚 Examples

### Basic Cylinder Flow
```python
from fluid_sim import LBMSimulation

# Create simulation
sim = LBMSimulation(nx=200, ny=100, reynolds=100)

# Setup cylinder obstacle  
sim.setup_cylinder_obstacle(cx=50, cy=50, r=10)

# Run simulation
for i in range(1000):
    diagnostics = sim.step()
    
    # Check stability
    if not diagnostics['is_stable']:
        print(f"Warning: Simulation unstable at step {i}")
```

### Custom Configuration
```python
from fluid_sim.utils import ConfigManager, SimulationConfig

# Create custom configuration
config = SimulationConfig(
    nx=300,
    ny=150, 
    reynolds=500.0,
    flow_speed=0.1
)

# Validate and save
config_manager = ConfigManager()
if config_manager.validate_config(config):
    config_manager.save_config(config, "custom_config.json")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**STOKEDMODELLER** - Fluid dynamics simulation enthusiast

## 🙏 Acknowledgments

- Lattice Boltzmann Method community for theoretical foundations
- NumPy and SciPy developers for numerical computing tools
- Matplotlib team for visualization capabilities
- tkinter developers for GUI framework