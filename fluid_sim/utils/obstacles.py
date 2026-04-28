"""
Obstacle creation and manipulation tools.

Mask convention: True == obstacle (solid), False == fluid. ``load_mask``
inverts the input image so that opaque (non-zero) pixels become solid.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import (binary_opening, generate_binary_structure,
                           rotate as ndi_rotate)


class ObstacleTools:
    """Tools for creating and manipulating obstacles. True == obstacle."""

    @staticmethod
    def create_circle(nx: int, ny: int, cx: int, cy: int, r: int) -> np.ndarray:
        return np.fromfunction(lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 < r ** 2, (nx, ny))

    @staticmethod
    def create_rectangle(nx: int, ny: int, cx: int, cy: int, l: int, w: int) -> np.ndarray:
        return np.fromfunction(
            lambda x, y: (x > cx - l / 2) & (x < cx + l / 2) & (y > cy - w / 2) & (y < cy + w / 2),
            (nx, ny),
        )

    @staticmethod
    def create_square(nx: int, ny: int, cx: int, cy: int, size: int) -> np.ndarray:
        return ObstacleTools.create_rectangle(nx, ny, cx, cy, size, size)

    @staticmethod
    def smooth_corners(obstacle_mask: np.ndarray, size: int = 2) -> np.ndarray:
        """
        Symmetric morphological opening to chip off single-pixel corners.

        Replaces the previous implementation which zeroed an asymmetric corner
        of the structuring element and biased the mask toward the upper-left.
        """
        size = max(int(size), 1)
        selem = generate_binary_structure(2, 1)
        if size > 1:
            # Iterate the 4-connected element ``size`` times to enlarge it.
            iters = size - 1
            return binary_opening(obstacle_mask, structure=selem, iterations=1 + iters)
        return binary_opening(obstacle_mask, structure=selem)

    @staticmethod
    def rotate_obstacle(obstacle_mask: np.ndarray, angle: float) -> np.ndarray:
        """Rotate the mask by ``angle`` (degrees, CCW) without leaving holes."""
        if not np.any(obstacle_mask):
            return obstacle_mask
        rotated = ndi_rotate(obstacle_mask.astype(float), angle, order=0,
                             reshape=False, mode='constant', cval=0.0)
        return rotated > 0.5


def create_obstacle(shape: str, nx: int, ny: int, cx: int, cy: int, r: int,
                    l: int = 0, w: int = 0) -> np.ndarray:
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
    Load a binary obstacle mask from an image file.

    Convention: opaque pixels (non-zero greyscale) become **obstacle** (True);
    transparent / black pixels become **fluid** (False). The returned array has
    shape ``(width, height)`` (PIL convention) — callers that work in
    ``(nx, ny)`` typically transpose.
    """
    im = Image.open(filename).convert("L")
    size = tuple(int(dim * scale) for dim in im.size)
    im = im.resize(size)

    arr = np.array(im)
    fluid = (arr > 0)
    return np.logical_not(fluid)
