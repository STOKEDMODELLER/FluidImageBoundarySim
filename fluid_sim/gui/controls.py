"""
Control panel for simulation parameters and controls.
"""
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Dict, Any


class ControlPanel(ttk.Frame):
    """Control panel for simulation parameters."""
    
    def __init__(self, parent, config, 
                 on_parameter_change: Callable[[str, Any], None],
                 on_start_stop: Callable[[], None],
                 on_reset: Callable[[], None],
                 on_step: Callable[[], None],
                 on_load_obstacle: Callable[[], None]):
        """
        Initialize control panel.
        
        Args:
            parent: Parent widget
            config: Configuration object
            on_parameter_change: Callback for parameter changes
            on_start_stop: Callback for start/stop button
            on_reset: Callback for reset button
            on_step: Callback for step button
            on_load_obstacle: Callback for load obstacle button
        """
        super().__init__(parent, width=300)
        
        self.config = config
        self.on_parameter_change = on_parameter_change
        self.on_start_stop = on_start_stop
        self.on_reset = on_reset
        self.on_step = on_step
        self.on_load_obstacle = on_load_obstacle
        
        # Track variables
        self.variables = {}
        self.is_running = False
        
        # Setup GUI
        self._setup_gui()
        
    def _setup_gui(self):
        """Setup the control panel GUI."""
        # Main container with scrollbar
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create control sections
        self._create_simulation_controls(scrollable_frame)
        self._create_grid_parameters(scrollable_frame)
        self._create_physical_parameters(scrollable_frame)
        self._create_simulation_parameters(scrollable_frame)
        self._create_obstacle_parameters(scrollable_frame)
        self._create_diagnostics_section(scrollable_frame)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    def _create_simulation_controls(self, parent):
        """Create simulation control buttons."""
        controls_frame = ttk.LabelFrame(parent, text="Simulation Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Start/Stop button
        self.start_stop_var = tk.StringVar(value="Start")\n        self.start_stop_btn = ttk.Button(\n            controls_frame, \n            textvariable=self.start_stop_var,\n            command=self.on_start_stop\n        )\n        self.start_stop_btn.pack(fill=tk.X, pady=2)\n        \n        # Step button\n        self.step_btn = ttk.Button(\n            controls_frame, \n            text=\"Single Step\",\n            command=self.on_step\n        )\n        self.step_btn.pack(fill=tk.X, pady=2)\n        \n        # Reset button\n        reset_btn = ttk.Button(\n            controls_frame, \n            text=\"Reset\",\n            command=self.on_reset\n        )\n        reset_btn.pack(fill=tk.X, pady=2)\n        \n        # Load obstacle button\n        load_btn = ttk.Button(\n            controls_frame, \n            text=\"Load Obstacle Mask\",\n            command=self.on_load_obstacle\n        )\n        load_btn.pack(fill=tk.X, pady=2)\n        \n    def _create_grid_parameters(self, parent):\n        \"\"\"Create grid parameter controls.\"\"\"\n        grid_frame = ttk.LabelFrame(parent, text=\"Grid Parameters\", padding=10)\n        grid_frame.pack(fill=tk.X, padx=5, pady=5)\n        \n        # Grid size X\n        self._create_parameter_control(\n            grid_frame, \"nx\", \"Grid Size X:\", \n            self.config.nx, 50, 500, int\n        )\n        \n        # Grid size Y\n        self._create_parameter_control(\n            grid_frame, \"ny\", \"Grid Size Y:\", \n            self.config.ny, 50, 300, int\n        )\n        \n    def _create_physical_parameters(self, parent):\n        \"\"\"Create physical parameter controls.\"\"\"\n        phys_frame = ttk.LabelFrame(parent, text=\"Physical Parameters\", padding=10)\n        phys_frame.pack(fill=tk.X, padx=5, pady=5)\n        \n        # Reynolds number\n        self._create_parameter_control(\n            phys_frame, \"reynolds\", \"Reynolds Number:\", \n            self.config.reynolds, 10.0, 1000.0, float\n        )\n        \n        # Flow speed\n        self._create_parameter_control(\n            phys_frame, \"flow_speed\", \"Flow Speed (m/s):\", \n            self.config.flow_speed, 0.01, 0.2, float\n        )\n        \n    def _create_simulation_parameters(self, parent):\n        \"\"\"Create simulation parameter controls.\"\"\"\n        sim_frame = ttk.LabelFrame(parent, text=\"Simulation Parameters\", padding=10)\n        sim_frame.pack(fill=tk.X, padx=5, pady=5)\n        \n        # Max iterations\n        self._create_parameter_control(\n            sim_frame, \"max_iterations\", \"Max Iterations:\", \n            self.config.max_iterations, 100, 50000, int\n        )\n        \n        # Omega (optional)\n        omega_frame = ttk.Frame(sim_frame)\n        omega_frame.pack(fill=tk.X, pady=2)\n        \n        ttk.Label(omega_frame, text=\"Omega (auto if empty):\").pack(anchor=tk.W)\n        \n        self.variables[\"omega\"] = tk.StringVar(\n            value=str(self.config.omega) if self.config.omega else \"\"\n        )\n        omega_entry = ttk.Entry(omega_frame, textvariable=self.variables[\"omega\"])\n        omega_entry.pack(fill=tk.X)\n        omega_entry.bind('<Return>', lambda e: self._on_omega_change())\n        omega_entry.bind('<FocusOut>', lambda e: self._on_omega_change())\n        \n    def _create_obstacle_parameters(self, parent):\n        \"\"\"Create obstacle parameter controls.\"\"\"\n        obs_frame = ttk.LabelFrame(parent, text=\"Obstacle Parameters\", padding=10)\n        obs_frame.pack(fill=tk.X, padx=5, pady=5)\n        \n        # Obstacle center X\n        self._create_parameter_control(\n            obs_frame, \"obstacle_cx\", \"Center X:\", \n            self.config.obstacle_cx, 10.0, 240.0, float\n        )\n        \n        # Obstacle center Y\n        self._create_parameter_control(\n            obs_frame, \"obstacle_cy\", \"Center Y:\", \n            self.config.obstacle_cy, 10.0, 110.0, float\n        )\n        \n        # Obstacle radius\n        self._create_parameter_control(\n            obs_frame, \"obstacle_r\", \"Radius:\", \n            self.config.obstacle_r, 5.0, 50.0, float\n        )\n        \n    def _create_diagnostics_section(self, parent):\n        \"\"\"Create diagnostics display section.\"\"\"\n        diag_frame = ttk.LabelFrame(parent, text=\"Diagnostics\", padding=10)\n        diag_frame.pack(fill=tk.X, padx=5, pady=5)\n        \n        # Diagnostics labels\n        self.diag_labels = {}\n        \n        diag_items = [\n            (\"time_step\", \"Time Step:\"),\n            (\"max_velocity\", \"Max Velocity:\"),\n            (\"max_pressure\", \"Max Pressure:\"),\n            (\"mach_number\", \"Mach Number:\"),\n            (\"is_stable\", \"Stable:\"),\n            (\"reynolds_estimate\", \"Est. Reynolds:\")\n        ]\n        \n        for key, label in diag_items:\n            frame = ttk.Frame(diag_frame)\n            frame.pack(fill=tk.X, pady=1)\n            \n            ttk.Label(frame, text=label).pack(side=tk.LEFT)\n            self.diag_labels[key] = ttk.Label(frame, text=\"N/A\")\n            self.diag_labels[key].pack(side=tk.RIGHT)\n            \n    def _create_parameter_control(self, parent, param_name: str, label: str, \n                                 initial_value, min_val, max_val, value_type):\n        \"\"\"Create a parameter control widget.\"\"\"\n        frame = ttk.Frame(parent)\n        frame.pack(fill=tk.X, pady=2)\n        \n        ttk.Label(frame, text=label).pack(anchor=tk.W)\n        \n        # Create variable\n        if value_type == int:\n            self.variables[param_name] = tk.IntVar(value=initial_value)\n        else:\n            self.variables[param_name] = tk.DoubleVar(value=initial_value)\n            \n        # Create scale\n        scale = ttk.Scale(\n            frame, \n            from_=min_val, \n            to=max_val,\n            variable=self.variables[param_name],\n            command=lambda v, p=param_name: self._on_scale_change(p, v)\n        )\n        scale.pack(fill=tk.X)\n        \n        # Create entry\n        entry = ttk.Entry(frame, textvariable=self.variables[param_name], width=10)\n        entry.pack()\n        entry.bind('<Return>', lambda e, p=param_name: self._on_entry_change(p))\n        entry.bind('<FocusOut>', lambda e, p=param_name: self._on_entry_change(p))\n        \n    def _on_scale_change(self, param_name: str, value: str):\n        \"\"\"Handle scale value changes.\"\"\"\n        try:\n            if param_name in [\"nx\", \"ny\", \"max_iterations\"]:\n                val = int(float(value))\n            else:\n                val = float(value)\n            self.on_parameter_change(param_name, val)\n        except ValueError:\n            pass\n            \n    def _on_entry_change(self, param_name: str):\n        \"\"\"Handle entry value changes.\"\"\"\n        try:\n            var = self.variables[param_name]\n            if param_name in [\"nx\", \"ny\", \"max_iterations\"]:\n                val = var.get()\n            else:\n                val = var.get()\n            self.on_parameter_change(param_name, val)\n        except (ValueError, tk.TclError):\n            pass\n            \n    def _on_omega_change(self):\n        \"\"\"Handle omega parameter change.\"\"\"\n        try:\n            omega_str = self.variables[\"omega\"].get().strip()\n            if omega_str:\n                omega_val = float(omega_str)\n                if 0 < omega_val < 2:\n                    self.on_parameter_change(\"omega\", omega_val)\n                else:\n                    tk.messagebox.showwarning(\"Warning\", \"Omega must be between 0 and 2\")\n            else:\n                self.on_parameter_change(\"omega\", None)\n        except ValueError:\n            tk.messagebox.showerror(\"Error\", \"Invalid omega value\")\n            \n    def set_running(self, running: bool):\n        \"\"\"Update GUI state based on simulation running status.\"\"\"\n        self.is_running = running\n        \n        if running:\n            self.start_stop_var.set(\"Stop\")\n            self.step_btn.config(state=tk.DISABLED)\n        else:\n            self.start_stop_var.set(\"Start\")\n            self.step_btn.config(state=tk.NORMAL)\n            \n    def update_diagnostics(self, diagnostics: Dict[str, Any]):\n        \"\"\"Update diagnostics display.\"\"\"\n        for key, label in self.diag_labels.items():\n            if key in diagnostics:\n                value = diagnostics[key]\n                if isinstance(value, float):\n                    if key in [\"max_velocity\", \"max_pressure\", \"mach_number\"]:\n                        text = f\"{value:.6f}\"\n                    else:\n                        text = f\"{value:.2f}\"\n                elif isinstance(value, bool):\n                    text = \"Yes\" if value else \"No\"\n                else:\n                    text = str(value)\n                label.config(text=text)\n                \n    def update_config(self, config):\n        \"\"\"Update control panel with new configuration.\"\"\"\n        self.config = config\n        \n        # Update all variables\n        for param_name, var in self.variables.items():\n            if hasattr(config, param_name):\n                value = getattr(config, param_name)\n                if param_name == \"omega\":\n                    var.set(str(value) if value is not None else \"\")\n                else:\n                    var.set(value)