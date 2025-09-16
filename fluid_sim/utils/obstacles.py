"""
Obstacle creation and manipulation tools.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure, rotate
from typing import Union


class ObstacleTools:
    """Tools for creating and manipulating obstacles."""
    
    @staticmethod
    def create_circle(nx: int, ny: int, cx: int, cy: int, r: int) -> np.ndarray:
        """Create circular obstacle."""
        return np.fromfunction(lambda x,y: (x-cx)**2+(y-cy)**2<r**2, (nx,ny))
    
    @staticmethod
    def create_rectangle(nx: int, ny: int, cx: int, cy: int, l: int, w: int) -> np.ndarray:
        """Create rectangular obstacle."""
        return np.fromfunction(lambda x,y: (x>cx-l/2) & (x<cx+l/2) & (y>cy-w/2) & (y<cy+w/2), (nx,ny))
    
    @staticmethod
    def create_square(nx: int, ny: int, cx: int, cy: int, size: int) -> np.ndarray:
        """Create square obstacle."""
        return ObstacleTools.create_rectangle(nx, ny, cx, cy, size, size)
    
    @staticmethod
    def smooth_corners(obstacle_mask: np.ndarray, erosion_size: int = 2) -> np.ndarray:
        """
        Smooths the corners of an obstacle mask by applying an erosion operation.
        
        Parameters:
        -----------
        obstacle_mask : numpy.ndarray
            Binary mask representing obstacles.
        erosion_size : int, optional
            Size of the erosion structuring element (default is 2).
        
        Returns:
        --------
        numpy.ndarray
            Binary mask with smoothed corners.
        """
        selem = generate_binary_structure(2, 2)
        selem[erosion_size-1:, erosion_size-1:] = False
        smoothed_mask = binary_erosion(obstacle_mask, selem)
        return smoothed_mask
    
    @staticmethod
    def rotate_obstacle(obstacle_mask: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotates the True section of an obstacle mask by a given angle in degrees.
        
        Parameters:
        -----------
        obstacle_mask : numpy.ndarray
            Binary mask representing obstacles.
        angle : float
            Angle of rotation in degrees (counterclockwise).
        
        Returns:
        --------
        numpy.ndarray
            Rotated binary mask representing obstacles.
        """
        indices = np.argwhere(obstacle_mask)
        
        if len(indices) == 0:
            return obstacle_mask
        
        centroid = np.mean(indices, axis=0)
        indices_centered = indices - centroid
        
        radians = np.deg2rad(angle)
        rotation_matrix = np.array([[np.cos(radians), -np.sin(radians)],
                                    [np.sin(radians), np.cos(radians)]])
        indices_rotated = np.dot(indices_centered, rotation_matrix).astype(int)
        indices_uncentered = indices_rotated + centroid
        
        rotated_mask = np.zeros_like(obstacle_mask)
        indices_uncentered = indices_uncentered.astype(int)
        indices_uncentered[:, 0] = np.clip(indices_uncentered[:, 0], 0, rotated_mask.shape[0]-1)
        indices_uncentered[:, 1] = np.clip(indices_uncentered[:, 1], 0, rotated_mask.shape[1]-1)
        rotated_mask[indices_uncentered[:, 0], indices_uncentered[:, 1]] = True
        
        return rotated_mask


def create_obstacle(shape: str, nx: int, ny: int, cx: int, cy: int, r: int, 
                   l: int = 0, w: int = 0) -> np.ndarray:
    """
    Creates an obstacle in the form of a numpy ndarray based on the specified shape.
    
    Parameters:
    -----------
    shape : str
        String representing the shape of the obstacle.
        Possible values: "circle", "rectangle", "square".
    nx : int
        Grid size in the x-direction.
    ny : int
        Grid size in the y-direction.
    cx : int
        x-coordinate of the center of the obstacle.
    cy : int
        y-coordinate of the center of the obstacle.
    r : int
        Radius of the obstacle (for "circle" shape).
    l : int
        Length of the obstacle (for "rectangle" shape).
    w : int
        Width of the obstacle (for "rectangle" and "square" shapes).
    
    Returns:
    --------
    numpy.ndarray
        Numpy array representing the obstacle.
    """
    if shape == "circle":
        return ObstacleTools.create_circle(nx, ny, cx, cy, r)
    elif shape == "rectangle":
        return ObstacleTools.create_rectangle(nx, ny, cx, cy, l, w)
    elif shape == "square":
        return ObstacleTools.create_square(nx, ny, cx, cy, r)
    else:
        raise ValueError(f"Invalid obstacle shape: {shape}")


def load_mask(filename: str, scale: float = 1.0) -> np.ndarray:
    """
    Loads a binary mask from a PNG file and scales it by the given factor.
    
    Parameters:
    -----------
    filename : str
        Name of the PNG file to load.
    scale : float, optional
        Scaling factor for the mask (default is 1.0).
    
    Returns:
    --------
    numpy.ndarray
        Numpy array representing the binary mask.
    """
    im = Image.open(filename).convert("L")
    
    size = tuple(int(dim * scale) for dim in im.size)
    im = im.resize(size)
    
    mask = np.array(im)
    mask = (mask > 0).astype(int)
    
    return np.logical_not(mask)