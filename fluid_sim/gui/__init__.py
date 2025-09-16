"""
GUI module - Contains the main GUI application and visualization components.
"""

# Try to import GUI components, but gracefully handle missing tkinter
try:
    from .main_window import SimulationGUI
    from .visualization import VisualizationPanel
    from .controls import ControlPanel
    __all__ = ['SimulationGUI', 'VisualizationPanel', 'ControlPanel']
except ImportError:
    # tkinter not available
    SimulationGUI = None
    VisualizationPanel = None
    ControlPanel = None
    __all__ = []