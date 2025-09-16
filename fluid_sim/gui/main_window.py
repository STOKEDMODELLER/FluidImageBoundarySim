"""
Main GUI window for the fluid simulation application.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from typing import Optional

from ..core import LBMSimulation
from ..utils import ConfigManager
from .visualization import VisualizationPanel
from .controls import ControlPanel


class SimulationGUI:
    """Main GUI application for fluid simulation."""
    
    def __init__(self):
        """Initialize the GUI application."""
        self.root = tk.Tk()
        self.root.title("Fluid Image Boundary Simulation - Lattice Boltzmann Method")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        self.simulation: Optional[LBMSimulation] = None
        self.simulation_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Setup GUI
        self._setup_gui()
        self._setup_menu()
        
        # Initialize simulation
        self._initialize_simulation()
        
    def _setup_gui(self):
        """Setup the main GUI layout."""
        # Create main frames
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create control panel on the left
        self.control_panel = ControlPanel(
            self.main_frame, 
            self.config,
            on_parameter_change=self._on_parameter_change,
            on_start_stop=self._on_start_stop,
            on_reset=self._on_reset,
            on_step=self._on_step,
            on_load_obstacle=self._on_load_obstacle
        )
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Create visualization panel on the right
        self.visualization_panel = VisualizationPanel(
            self.main_frame,
            width=800,
            height=600
        )
        self.visualization_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Configuration...", command=self._load_config)
        file_menu.add_command(label="Save Configuration...", command=self._save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Load Obstacle Mask...", command=self._on_load_obstacle)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Simulation menu
        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="Start/Stop", command=self._on_start_stop)
        sim_menu.add_command(label="Step", command=self._on_step)
        sim_menu.add_command(label="Reset", command=self._on_reset)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        
    def _initialize_simulation(self):
        """Initialize the simulation with current configuration."""
        try:
            self.simulation = LBMSimulation(
                nx=self.config.nx,
                ny=self.config.ny,
                reynolds=self.config.reynolds,
                flow_speed=self.config.flow_speed,
                omega=self.config.omega
            )
            
            # Setup initial obstacle
            self.simulation.setup_cylinder_obstacle(
                cx=self.config.obstacle_cx,
                cy=self.config.obstacle_cy,
                r=self.config.obstacle_r
            )
            
            # Update visualization
            self._update_visualization()
            self.status_var.set("Simulation initialized")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize simulation: {str(e)}")
            self.status_var.set("Error: Failed to initialize")
    
    def _on_parameter_change(self, parameter: str, value):
        """Handle parameter changes from control panel."""
        try:
            # Update configuration
            setattr(self.config, parameter, value)
            
            # Reinitialize simulation if necessary
            if parameter in ['nx', 'ny', 'reynolds', 'flow_speed', 'omega']:
                self._initialize_simulation()
            elif parameter in ['obstacle_cx', 'obstacle_cy', 'obstacle_r']:
                if self.simulation:
                    self.simulation.setup_cylinder_obstacle(
                        cx=self.config.obstacle_cx,
                        cy=self.config.obstacle_cy,
                        r=self.config.obstacle_r
                    )
                    self._update_visualization()
                    
            self.status_var.set(f"Parameter {parameter} updated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update parameter {parameter}: {str(e)}")
    
    def _on_start_stop(self):
        """Handle start/stop button."""
        if not self.is_running:
            self._start_simulation()
        else:
            self._stop_simulation()
    
    def _start_simulation(self):
        """Start the simulation in a separate thread."""
        if not self.simulation:
            messagebox.showerror("Error", "No simulation initialized")
            return
            
        self.is_running = True
        self.control_panel.set_running(True)
        self.status_var.set("Simulation running...")
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def _stop_simulation(self):
        """Stop the running simulation."""
        self.is_running = False
        if self.simulation:
            self.simulation.stop()
        self.control_panel.set_running(False)
        self.status_var.set("Simulation stopped")
    
    def _run_simulation(self):
        """Run the simulation loop."""
        try:
            while self.is_running and self.simulation:
                # Perform simulation step
                diagnostics = self.simulation.step()
                
                # Update GUI in main thread
                self.root.after(0, self._update_gui_with_diagnostics, diagnostics)
                
                # Control simulation speed
                time.sleep(0.01)  # 10ms delay
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Simulation error: {str(e)}"))
            self.root.after(0, self._stop_simulation)
    
    def _update_gui_with_diagnostics(self, diagnostics):
        """Update GUI with simulation diagnostics."""
        # Update control panel
        self.control_panel.update_diagnostics(diagnostics)
        
        # Update visualization
        self._update_visualization()
        
        # Update status
        if diagnostics:
            status = f"Step: {diagnostics.get('time_step', 0)}, "
            status += f"Max Vel: {diagnostics.get('max_velocity', 0):.4f}, "
            status += f"Stable: {'Yes' if diagnostics.get('is_stable', False) else 'No'}"
            self.status_var.set(status)
    
    def _on_step(self):
        """Perform a single simulation step."""
        if not self.simulation:
            messagebox.showwarning("Warning", "No simulation initialized")
            return
            
        try:
            diagnostics = self.simulation.step()
            self._update_gui_with_diagnostics(diagnostics)
            
        except Exception as e:
            messagebox.showerror("Error", f"Step failed: {str(e)}")
    
    def _on_reset(self):
        """Reset the simulation."""
        self._stop_simulation()
        self._initialize_simulation()
    
    def _on_load_obstacle(self):
        """Load obstacle mask from file."""
        filename = filedialog.askopenfilename(
            title="Load Obstacle Mask",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if self.simulation:
                    self.simulation.setup_from_mask(filename)
                    self._update_visualization()
                    self.status_var.set(f"Loaded obstacle mask: {filename}")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load obstacle mask: {str(e)}")
    
    def _update_visualization(self):
        """Update the visualization panel."""
        if self.simulation:
            velocity_mag = self.simulation.get_velocity_magnitude()
            pressure = self.simulation.get_pressure_field()
            obstacle = self.simulation.obstacle
            
            self.visualization_panel.update_fields(velocity_mag, pressure, obstacle)
    
    def _load_config(self):
        """Load configuration from file."""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.config = self.config_manager.load_config(filename)
                self.control_panel.update_config(self.config)
                self._initialize_simulation()
                self.status_var.set(f"Configuration loaded: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
    
    def _save_config(self):
        """Save configuration to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.config_manager.save_config(self.config, filename)
                self.status_var.set(f"Configuration saved: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def _show_about(self):
        """Show about dialog."""
        about_text = """
Fluid Image Boundary Simulation v2.0.0

A Lattice Boltzmann Method fluid simulation with GUI interface.

Features:
• Real-time fluid simulation visualization
• Interactive parameter control
• Obstacle mask loading
• Configuration management
• Stability analysis

Author: STOKEDMODELLER
        """
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()
    
    def destroy(self):
        """Clean up and destroy the application."""
        self._stop_simulation()
        self.root.destroy()