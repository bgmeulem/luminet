"""Spatial operations

Simple convenience functions for polar coordinates that are often used in this package.
"""

import numpy as np
from scipy.spatial.distance import pdist
from typing import Tuple

def polar_to_cartesian(
        th: float | np.ndarray, 
        radius: float | np.ndarray, 
        rotation : float = 0) -> float | np.ndarray:
    """Convert polar to cartesian coordinates.
    
    Args:
        th (float | np.ndarray): angle in radians
        radius (float | np.ndarray): radius
        rotation (float): Amount of radians to rotate the result. Useful when messing with the orientation of polar plots.

    Returns:
        Tuple[float | :class:`~np.ndarray`]: x and y coordinates
    """
    x = radius * np.cos(th + rotation)
    y = - radius * np.sin(th + rotation)
    return x, y


def cartesian_to_polar(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """Convert cartesian to polar coordinates.

    Args:
        x (float | :class:`~np.ndarray`): x coordinate(s)
        y (float | :class:`~np.ndarray`): y coordinate(s)

    Returns:
        Tuple[float | :class:`~np.ndarray`]: angle in radians and radius
    """
    R = np.hypot(x, y)
    th = np.arctan2(y, x) % (2*np.pi)
    return th, R

def polar_cartesian_distance(p1: Tuple[float], p2: Tuple[float]) -> float:
    """Calculate cartesian distance between two polar coordinates
    
    Args:
        p1 (Tuple[float]): Two-tuple containing the angle and radius of point 1.
        p1 (Tuple[float]): Two-tuple containing the angle and radius of point 2.

    Returns:
        float: Cartesian distance between the two points.

    """
    return pdist((p1, p2))[0]