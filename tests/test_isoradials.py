from src.isoradial import Isoradial
import numpy as np
import pytest

@pytest.mark.parametrize("mass", [1., 2.])
@pytest.mark.parametrize("inclination", [0, 45, 90, 135])
@pytest.mark.parametrize("radius", [6., 20, 60.])
def test_isoradials(mass, inclination, radius) -> None:
    N_ISORADIALS=20
    radii = np.linspace(6, 60, N_ISORADIALS)
    ir = Isoradial(radius=radius*mass, incl=inclination, bh_mass=mass, order=0).calculate()
    ir_ghost = Isoradial(radius=radius*mass, incl=inclination, bh_mass=mass, order=0).calculate()

    return None