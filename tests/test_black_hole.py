from luminet.black_hole import BlackHole
import numpy as np
import pytest

@pytest.mark.parametrize("mass", [1., 2.])
@pytest.mark.parametrize("incl", np.linspace(np.pi/3, np.pi/2, 3))
def test_varying_mass(mass, incl):
    """
    Test if black hole can be created with a mass other than 1
    """

    bh = BlackHole(incl=incl, mass=mass)
    radii = np.linspace(6*mass, 60*mass, 10)
    bh.calc_isoradials(direct_r=radii, ghost_r=radii)  # calculate some isoradials, should be quick enough
    for isoradial in bh.isoradials:
        assert not any(np.isnan(isoradial.impact_parameters)), "Isoradials contain nan values"
        assert not any(np.isnan(isoradial.angles)), "Isoradials contain nan values"
    return None


if __name__ == "__main__":
    test_varying_mass(1., np.pi/3)
 