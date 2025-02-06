import numpy as np

def polar_to_cartesian(th, radius, rotation=0):
    x = radius * np.cos(th + rotation)
    y = - radius * np.sin(th + rotation)
    return x, y


def cartesian_to_polar(x, y):
    R = np.hypot(x, y)
    th = np.arctan2(y, x) % (2*np.pi)
    return th, R

def get_angle_around(p1, p2):
    """
    Calculates the angle of p2 around p1

    :param p1: coordinate 1 in format [x, y]
    :param p2:  coordinate 2 in format [x, y]
    :return: angle in radians
    """
    cx, cy = p1

    p2_ = np.subtract(p2, p1)
    angle_center, _ = cartesian_to_polar(cx, cy)
    # rotate p2_ counter-clockwise until the vector to the isoradial center is aligned with negative y-axis
    theta = np.pi - angle_center if angle_center > np.pi else angle_center
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    p2_ = np.dot(rot, p2_)
    angle_target, _ = cartesian_to_polar(p2[0], p2[1])
    angle_target_around_center, _ = cartesian_to_polar(p2_[0], p2_[1])

    return angle_target_around_center
