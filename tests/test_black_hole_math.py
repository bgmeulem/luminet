import numpy as np
import pytest

from luminet import black_hole_math as bhmath

N_inclines = 2
N_angles = 2
N_radii = 2


@pytest.mark.parametrize("mass", [1.0, 2.5, 4.0])
@pytest.mark.parametrize(
    "incl", [np.random.randint(0, np.pi) for _ in range(N_inclines)]
)
@pytest.mark.parametrize(
    "angle", [np.random.randint(0.0, 2 * np.pi) for _ in range(N_angles)]
)
@pytest.mark.parametrize(
    "radius", [np.random.randint(6.0, 60.0) for _ in range(N_radii)]
)
@pytest.mark.parametrize(
    "order", [0.0, 1.0, 2.0]
)  # test potential higher orders as well
class TestParametrized:

    def test_calc_periastron(self, mass, incl, angle, radius, order):
        """
        Test the method for calculating the impact parameter with varying input parameters
        """
        bhmath.solve_for_perigee(
            radius=radius * mass,
            incl=incl,
            alpha=angle,
            bh_mass=mass,
            order=order,
        )

    def test_get_b_from_periastron(self, mass, incl, angle, radius, order):
        b = bhmath.solve_for_impact_parameter(
            radius=radius * mass,
            incl=incl,
            alpha=angle,
            order=order,
            bh_mass=mass,
        )
        assert (not np.isnan(b)) and (b is not None), (
            f"Calculating impact parameter failed. with"
            f"M={mass},    incl={incl}, alpha={angle},  R={radius}, "
            f"order={order}"
        )
        return None