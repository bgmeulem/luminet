from luminet.isoradial import Isoradial
import numpy as np
import pytest

@pytest.mark.parametrize("mass", [1., 2.])
@pytest.mark.parametrize("incl", [0, 45, 90, 135])
@pytest.mark.parametrize("radius", [6., 20, 60.])
def test_isoradials(mass, incl, radius) -> None:
    N_ISORADIALS=20
    radii = np.linspace(6, 60, N_ISORADIALS)
    ir = Isoradial(radius=radius*mass, incl=incl, bh_mass=mass, order=0).calculate()
    ir_ghost = Isoradial(radius=radius*mass, incl=incl, bh_mass=mass, order=0).calculate()

    return None