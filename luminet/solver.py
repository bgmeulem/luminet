import numpy as np
from typing import Callable
from typing import Dict

def midpoint_method(
        func: Callable, 
        x: np.ndarray, 
        y: np.ndarray, 
        kwargs: Dict
        ) -> np.ndarray:
    assert len(x) == len(y) == 2, "x and y must have length 2"
    assert np.sign(y[0]) != np.sign(y[1]), "No sign change in y"
    inbetween_x = np.mean(x)
    inbetween_solution = func(periastron=inbetween_x, **kwargs)
    if np.sign(y[0]) == np.sign(inbetween_solution):
        x[0], y[0] = inbetween_x, inbetween_solution
    else:
        x[1], y[1] = inbetween_x, inbetween_solution
    return x


def improve_solutions_midpoint(
        func: Callable, 
        kwargs: Dict, 
        x: np.ndarray, 
        y: np.ndarray, 
        iterations: int
        ) -> np.ndarray:
    """
    To increase precision.
    Recalculate each solution in :arg:`solutions` using the provided :arg:`func`.
    Achieves an improved solution be re-evaluating the provided :arg:`func` at a new
    :arg:`x`, inbetween two pre-existing values for :arg:`x` where the sign of :arg:`y` changes.
    Does this :arg:`iterations` times
    """
    assert len(x) == len(y) == 2, "x and y must have length 2"
    for _ in range(iterations):
        x = midpoint_method(
            func=func, 
            kwargs=kwargs, 
            x=x, 
            y=y)
    return x

