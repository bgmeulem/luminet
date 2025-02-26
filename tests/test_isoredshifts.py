
from luminet.black_hole import BlackHole
import numpy as np
import matplotlib.pyplot as plt
import pytest

@pytest.mark.parametrize("incl", np.linspace(np.pi/3, np.pi/2, 3))
def test_isoredshifts(incl):
    """
    Attention:
        Only inclinations close to pi/2 have large redshifts
    
    # TODO: calc min and max redshifts for a given mass
    """

    bh = BlackHole(incl=incl, mass=1.)
    try:
        zs = [-.1, 0, .2]
        ax = bh.plot_isoredshifts(zs, c="white")
        # plt.show()  # Uncomment to see the plot
    except AssertionError as e:
        raise ValueError("Failed for incl={}".format(incl)) from e
    return None

if __name__ == "__main__":
    test_isoredshifts(np.pi/2-0.1)