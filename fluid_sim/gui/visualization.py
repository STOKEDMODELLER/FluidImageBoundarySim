"""
Visualization panel for displaying simulation results.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from typing import Optional


class VisualizationPanel(ttk.Frame):
    """Panel for visualizing simulation results."""
    
    def __init__(self, parent, width: int = 800, height: int = 600):
        """
        Initialize visualization panel.
        
        Args:
            parent: Parent widget
            width: Panel width
            height: Panel height
        """
        super().__init__(parent)
        
        self.width = width
        self.height = height
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create visualization tabs
        self._create_velocity_tab()
        self._create_pressure_tab()
        self._create_combined_tab()
        
        # Current data
        self.velocity_data: Optional[np.ndarray] = None
        self.pressure_data: Optional[np.ndarray] = None
        self.obstacle_data: Optional[np.ndarray] = None
        
    def _create_velocity_tab(self):
        """Create velocity visualization tab."""
        self.velocity_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.velocity_frame, text="Velocity")
        
        # Create matplotlib figure
        self.velocity_fig = Figure(figsize=(8, 6), dpi=100)
        self.velocity_ax = self.velocity_fig.add_subplot(111)
        self.velocity_ax.set_title("Velocity Magnitude")
        self.velocity_ax.set_xlabel("X")
        self.velocity_ax.set_ylabel("Y")
        
        # Create canvas
        self.velocity_canvas = FigureCanvasTkAgg(self.velocity_fig, self.velocity_frame)
        self.velocity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plot
        self.velocity_im = None
        self.velocity_cbar = None
        
    def _create_pressure_tab(self):
        """Create pressure visualization tab."""
        self.pressure_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pressure_frame, text="Pressure")
        
        # Create matplotlib figure
        self.pressure_fig = Figure(figsize=(8, 6), dpi=100)
        self.pressure_ax = self.pressure_fig.add_subplot(111)
        self.pressure_ax.set_title("Pressure")
        self.pressure_ax.set_xlabel("X")
        self.pressure_ax.set_ylabel("Y")
        
        # Create canvas
        self.pressure_canvas = FigureCanvasTkAgg(self.pressure_fig, self.pressure_frame)
        self.pressure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plot
        self.pressure_im = None
        self.pressure_cbar = None
        
    def _create_combined_tab(self):
        """Create combined visualization tab."""
        self.combined_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.combined_frame, text="Combined")
        
        # Create matplotlib figure with subplots
        self.combined_fig = Figure(figsize=(8, 8), dpi=100)
        self.combined_ax1 = self.combined_fig.add_subplot(211)
        self.combined_ax2 = self.combined_fig.add_subplot(212)
        
        self.combined_ax1.set_title("Velocity Magnitude")
        self.combined_ax1.set_xlabel("X")
        self.combined_ax1.set_ylabel("Y")
        
        self.combined_ax2.set_title("Pressure")
        self.combined_ax2.set_xlabel("X")
        self.combined_ax2.set_ylabel("Y")
        
        self.combined_fig.tight_layout()
        
        # Create canvas
        self.combined_canvas = FigureCanvasTkAgg(self.combined_fig, self.combined_frame)
        self.combined_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plots
        self.combined_im1 = None
        self.combined_im2 = None
        self.combined_cbar1 = None
        self.combined_cbar2 = None
        
    def update_fields(self, velocity: Optional[np.ndarray] = None, 
                     pressure: Optional[np.ndarray] = None, 
                     obstacle: Optional[np.ndarray] = None):
        """
        Update visualization with new field data.
        
        Args:
            velocity: Velocity magnitude field
            pressure: Pressure field
            obstacle: Obstacle mask
        """
        if velocity is not None:
            self.velocity_data = velocity
        if pressure is not None:
            self.pressure_data = pressure
        if obstacle is not None:
            self.obstacle_data = obstacle
            
        # Update visualizations
        self._update_velocity_plot()
        self._update_pressure_plot()
        self._update_combined_plot()
        
    def _update_velocity_plot(self):
        """Update velocity visualization."""
        if self.velocity_data is None:
            return
            
        # Clear previous plot
        self.velocity_ax.clear()
        self.velocity_ax.set_title("Velocity Magnitude")
        self.velocity_ax.set_xlabel("X")
        self.velocity_ax.set_ylabel("Y")
        
        # Create velocity plot with obstacle overlay
        data_to_plot = self.velocity_data.T
        
        # Mask obstacles if available
        if self.obstacle_data is not None:
            masked_data = np.ma.masked_where(self.obstacle_data.T, data_to_plot)
            self.velocity_im = self.velocity_ax.imshow(
                masked_data, cmap='jet', origin='lower', aspect='auto'
            )
            
            # Show obstacles in gray
            obstacle_overlay = np.ma.masked_where(~self.obstacle_data.T, np.ones_like(data_to_plot))
            self.velocity_ax.imshow(
                obstacle_overlay, cmap='gray', origin='lower', aspect='auto', alpha=0.8
            )
        else:
            self.velocity_im = self.velocity_ax.imshow(
                data_to_plot, cmap='jet', origin='lower', aspect='auto'
            )
        
        # Add colorbar
        if self.velocity_cbar:
            self.velocity_cbar.remove()
        self.velocity_cbar = self.velocity_fig.colorbar(self.velocity_im, ax=self.velocity_ax)
        self.velocity_cbar.set_label('Velocity Magnitude')
        
        self.velocity_canvas.draw()
        
    def _update_pressure_plot(self):
        """Update pressure visualization."""
        if self.pressure_data is None:
            return
            
        # Clear previous plot
        self.pressure_ax.clear()
        self.pressure_ax.set_title("Pressure")
        self.pressure_ax.set_xlabel("X")
        self.pressure_ax.set_ylabel("Y")
        
        # Create pressure plot with obstacle overlay
        data_to_plot = self.pressure_data.T
        
        # Mask obstacles if available
        if self.obstacle_data is not None:
            masked_data = np.ma.masked_where(self.obstacle_data.T, data_to_plot)
            self.pressure_im = self.pressure_ax.imshow(
                masked_data, cmap='viridis', origin='lower', aspect='auto'
            )
            
            # Show obstacles in gray
            obstacle_overlay = np.ma.masked_where(~self.obstacle_data.T, np.ones_like(data_to_plot))
            self.pressure_ax.imshow(
                obstacle_overlay, cmap='gray', origin='lower', aspect='auto', alpha=0.8
            )
        else:
            self.pressure_im = self.pressure_ax.imshow(
                data_to_plot, cmap='viridis', origin='lower', aspect='auto'
            )
        
        # Add colorbar
        if self.pressure_cbar:
            self.pressure_cbar.remove()
        self.pressure_cbar = self.pressure_fig.colorbar(self.pressure_im, ax=self.pressure_ax)
        self.pressure_cbar.set_label('Pressure')
        
        self.pressure_canvas.draw()
        
    def _update_combined_plot(self):
        """Update combined visualization."""
        if self.velocity_data is None or self.pressure_data is None:
            return
            
        # Clear previous plots
        self.combined_ax1.clear()
        self.combined_ax2.clear()
        
        self.combined_ax1.set_title("Velocity Magnitude")
        self.combined_ax1.set_xlabel("X")
        self.combined_ax1.set_ylabel("Y")
        
        self.combined_ax2.set_title("Pressure")
        self.combined_ax2.set_xlabel("X")
        self.combined_ax2.set_ylabel("Y")
        
        # Plot velocity
        vel_data = self.velocity_data.T
        if self.obstacle_data is not None:
            vel_masked = np.ma.masked_where(self.obstacle_data.T, vel_data)
            self.combined_im1 = self.combined_ax1.imshow(
                vel_masked, cmap='jet', origin='lower', aspect='auto'
            )
            obstacle_overlay = np.ma.masked_where(~self.obstacle_data.T, np.ones_like(vel_data))
            self.combined_ax1.imshow(
                obstacle_overlay, cmap='gray', origin='lower', aspect='auto', alpha=0.8
            )
        else:
            self.combined_im1 = self.combined_ax1.imshow(
                vel_data, cmap='jet', origin='lower', aspect='auto'
            )
        
        # Plot pressure
        press_data = self.pressure_data.T
        if self.obstacle_data is not None:
            press_masked = np.ma.masked_where(self.obstacle_data.T, press_data)
            self.combined_im2 = self.combined_ax2.imshow(
                press_masked, cmap='viridis', origin='lower', aspect='auto'
            )
            obstacle_overlay = np.ma.masked_where(~self.obstacle_data.T, np.ones_like(press_data))
            self.combined_ax2.imshow(
                obstacle_overlay, cmap='gray', origin='lower', aspect='auto', alpha=0.8
            )
        else:
            self.combined_im2 = self.combined_ax2.imshow(
                press_data, cmap='viridis', origin='lower', aspect='auto'
            )
        
        # Add colorbars
        if self.combined_cbar1:
            self.combined_cbar1.remove()
        if self.combined_cbar2:
            self.combined_cbar2.remove()
            
        self.combined_cbar1 = self.combined_fig.colorbar(self.combined_im1, ax=self.combined_ax1)
        self.combined_cbar1.set_label('Velocity Magnitude')
        
        self.combined_cbar2 = self.combined_fig.colorbar(self.combined_im2, ax=self.combined_ax2)
        self.combined_cbar2.set_label('Pressure')
        
        self.combined_fig.tight_layout()
        self.combined_canvas.draw()
        
    def save_current_view(self, filename: str):
        """
        Save current visualization to file.
        
        Args:
            filename: Output filename
        """
        current_tab = self.notebook.select()
        tab_text = self.notebook.tab(current_tab, "text")
        
        if tab_text == "Velocity":
            self.velocity_fig.savefig(filename, dpi=150, bbox_inches='tight')
        elif tab_text == "Pressure":
            self.pressure_fig.savefig(filename, dpi=150, bbox_inches='tight')
        elif tab_text == "Combined":
            self.combined_fig.savefig(filename, dpi=150, bbox_inches='tight')