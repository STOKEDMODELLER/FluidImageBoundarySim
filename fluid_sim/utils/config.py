"""
Configuration management for the fluid simulation.
"""
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class SimulationConfig:
    """Configuration parameters for fluid simulation."""
    
    # Grid parameters
    nx: int = 250
    ny: int = 120
    
    # Physical parameters
    reynolds: float = 300.0
    flow_speed: float = 0.05  # m/s
    
    # Simulation parameters
    max_iterations: int = 5000
    omega: Optional[float] = None
    
    # Obstacle parameters
    obstacle_cx: float = 62.5  # nx/4
    obstacle_cy: float = 60.0  # ny/2
    obstacle_r: float = 13.3   # ny/9
    
    # Visualization parameters
    save_interval: int = 50
    output_dir: str = "./output/"
    colormap: str = "jet"
    dpi: int = 100
    
    # GUI parameters
    window_width: int = 1200
    window_height: int = 800
    update_interval: int = 100  # ms


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self, config_file: str = "simulation_config.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = SimulationConfig()
        
    def load_config(self, config_file: Optional[str] = None) -> SimulationConfig:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Optional path to config file
            
        Returns:
            Loaded configuration object
        """
        file_path = config_file or self.config_file
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Update config with loaded values
                for key, value in config_dict.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        
                print(f"Configuration loaded from {file_path}")
                
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config from {file_path}: {e}")
                print("Using default configuration.")
        else:
            print(f"Config file {file_path} not found. Using default configuration.")
            
        return self.config
    
    def save_config(self, config: Optional[SimulationConfig] = None, 
                   config_file: Optional[str] = None) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration object to save (uses self.config if None)
            config_file: Optional path to config file
        """
        config_obj = config or self.config
        file_path = config_file or self.config_file
        
        try:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(file_path)
            if dir_path:  # Only create directory if there is a directory path
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(asdict(config_obj), f, indent=2)
                
            print(f"Configuration saved to {file_path}")
            
        except IOError as e:
            print(f"Error saving config to {file_path}: {e}")
    
    def create_default_config(self, file_path: str) -> None:
        """
        Create a default configuration file.
        
        Args:
            file_path: Path where to save the default config
        """
        default_config = SimulationConfig()
        self.save_config(default_config, file_path)
    
    def validate_config(self, config: Optional[SimulationConfig] = None) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate (uses self.config if None)
            
        Returns:
            True if configuration is valid
        """
        config_obj = config or self.config
        
        # Check grid parameters
        if config_obj.nx <= 0 or config_obj.ny <= 0:
            print("Error: Grid dimensions must be positive")
            return False
            
        # Check physical parameters
        if config_obj.reynolds <= 0:
            print("Error: Reynolds number must be positive")
            return False
            
        if config_obj.flow_speed <= 0:
            print("Error: Flow speed must be positive")
            return False
            
        # Check simulation parameters
        if config_obj.max_iterations <= 0:
            print("Error: Maximum iterations must be positive")
            return False
            
        if config_obj.omega is not None and not (0 < config_obj.omega < 2):
            print("Error: Omega must be between 0 and 2")
            return False
            
        # Check obstacle parameters
        if not (0 <= config_obj.obstacle_cx <= config_obj.nx):
            print("Error: Obstacle x-center must be within grid")
            return False
            
        if not (0 <= config_obj.obstacle_cy <= config_obj.ny):
            print("Error: Obstacle y-center must be within grid")
            return False
            
        if config_obj.obstacle_r <= 0:
            print("Error: Obstacle radius must be positive")
            return False
            
        return True
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return asdict(self.config)
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = SimulationConfig()