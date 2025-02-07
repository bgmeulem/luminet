from functools import partial
from typing import Callable, Dict

import numpy as np
import scipy.optimize as opt


def improve_solutions(
    func: Callable,
    x: np.ndarray,
    y: np.ndarray,
    kwargs: Dict,
) -> np.ndarray:
    """
    Find the root of a function.
    """
    assert len(x) == len(y) == 2, "x and y must have length 2"
    assert np.sign(y[0]) != np.sign(y[1]), "No sign change in y"

    x = opt.brentq(partial(func, **kwargs), x[0], x[1])
    # x = opt.ridder(partial(func, **kwargs), x[0], x[1])
    # x = opt.bisect(partial(func, **kwargs), x[0], x[1])
    return x


def fmin_2d(
    func: Callable,
    x0: np.ndarray,
    args: tuple,
    xtol: float = 1e-6,
    ftol: float = 1e-6,
    maxiter: int = 200,
):
    """
    Find the minimum of a function of two variables.
    """
    res = opt.fmin(func, x0, args=args, xtol=xtol, ftol=ftol, maxiter=maxiter)
    return res.xopt
