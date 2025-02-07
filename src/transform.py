"""Convert cartesian to polar coordinates and back."""

import numpy as np

def polar_to_cartesian(th, radius, rotation=0):
    """Convert polar to cartesian coordinates.
    
    Args:
        th (float | np.ndarray): angle in radians
        radius (float | np.ndarray): radius

    Returns:
        Tuple[float | np.ndarray]: x and y coordinates
    """
    x = radius * np.cos(th + rotation)
    y = - radius * np.sin(th + rotation)
    return x, y


def cartesian_to_polar(x, y):
    """Convert cartesian to polar coordinates.

    Args:
        x (float | np.ndarray): x coordinate
        y (float | np.ndarray): y coordinate

    Returns:
        Tuple[float | np.ndarray]: angle in radians and radius
    """
    R = np.hypot(x, y)
    th = np.arctan2(y, x) % (2*np.pi)
    return th, R
