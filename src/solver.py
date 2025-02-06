import numpy as np
from typing import Callable, Dict
from functools import partial
import scipy.optimize as opt

def improve_solutions(
        func: Callable,
        kwargs: Dict,
        x: np.ndarray,
        y: np.ndarray,
) -> np.ndarray:
    """
    Find the root of a function.
    """
    assert len(x) == len(y) == 2, "x and y must have length 2"
    assert np.sign(y[0]) != np.sign(y[1]), "No sign change in y"

    x = opt.brentq(partial(func, **kwargs), x[0], x[1])
    # x = opt.ridder(partial(func, **kwargs), x[0], x[1])
    # x = opt.bisect(partial(func, **kwargs), x[0], x[1])
    return x, None