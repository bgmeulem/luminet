import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src import black_hole


def test_bh_isoradial_coverage():
    """Plot a black hole by plotting a full range of isoradials"""
    M = 1.0
    incl = 85 * np.pi / 180
    outer_accretion_disk_edge = 40 * M
    bh = black_hole.BlackHole(
        inclination=incl, mass=M, outer_edge=outer_accretion_disk_edge
    )

    ax = bh.plot()
    t_end = time.time()
    plt.savefig("isoradials.pdf", dpi=200)
    plt.show()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../")))
    test_bh_isoradial_coverage()
