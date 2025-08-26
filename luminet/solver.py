"""Scipy solvers.

This module provides light wrappers to scipy solvers.
"""

from functools import partial
from typing import Callable, Dict

import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp1d


def improve_solutions(
    func: Callable,
    x: np.ndarray,
    y: np.ndarray,
    kwargs: Dict,
) -> np.ndarray:
    """Find the root of a function.

    Uses brentq to find the root of a function.

    Args:
        func (Callable): function to find the root of
        x (np.ndarray): x values
        y (np.ndarray): y values
        kwargs (Dict): keyword arguments for the function

    Returns:
        float: Root of the function i.e. where :math:`y = 0`
    """
    assert len(x) == len(y) == 2, "x and y must have length 2"
    assert np.sign(y[0]) != np.sign(y[1]), "No sign change in y"

    x = opt.brentq(partial(func, **kwargs), x[0], x[1])
    # x = opt.ridder(partial(func, **kwargs), x[0], x[1])
    # x = opt.bisect(partial(func, **kwargs), x[0], x[1])
    return x


def root_2d(
    func: Callable,
    x0: np.ndarray,
    args: tuple = (),
    method: str = 'hybr',
    tol: float = 1e-6,
):
    """
    Find the root of a function of two variables.

    :meta private:
    """
    res = opt.root(func, x0, args=args, method=method, tol=tol)
    if not res.success:
        return np.array([np.nan, np.nan])
    return res.x

def interpolator(x, y):
    return interp1d(x, y, kind="cubic")
