from src.black_hole import *
import numpy as np
import pytest

@pytest.mark.parametrize("mass", [1., 2.])
def test_varying_mass(mass):
    """
    Test if black hole can be created with a mass other than 1
    """

    for incl in (0, 45, 90, 135):
        bh = BlackHole(inclination=incl, mass=mass)
        radii = np.linspace(6*mass, 60*mass, 10)
        bh.calc_isoradials(direct_r=radii, ghost_r=radii)  # calculate some isoradials, should be quick enough
        for isoradial in bh.isoradials:
            assert not any(np.isnan(isoradial.radii_b)), "Isoradials contain nan values"
            assert not any(np.isnan(isoradial.angles)), "Isoradials contain nan values"
    return None