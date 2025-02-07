
from src.black_hole import *
import pytest

@pytest.mark.parametrize("mass", [1., 2.])
@pytest.mark.parametrize("incl", np.linspace(np.pi/3, np.pi/2, 4))
def test_isoredshifts(mass, incl):
    """
    Test if black hole can be created with a mass other than 1

    Attention:
        Only inclinations close to pi/2 have large redshifts
    
    # TODO: calc min and max redshifts for a given mass
    """

    bh = BlackHole(inclination=incl, mass=mass)
    try:
        zs = [-.1, 0, .2]
        bh.plot_isoredshifts(zs, c="white")
        plt.show()
    except AssertionError as e:
        raise ValueError("Failed for mass={}, incl={}".format(mass, incl)) from e
    return None

if __name__ == "__main__":
    test_isoredshifts(1., np.pi/2-0.01)
    print("Everything passed")