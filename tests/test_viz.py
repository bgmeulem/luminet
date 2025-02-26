import os
import time

import matplotlib.pyplot as plt
import numpy as np

from luminet import black_hole


def test_bh_isoradial_coverage():
    """Plot a black hole by plotting a full range of isoradials"""
    M = 1.0
    incl = 85 * np.pi / 180
    outer_accretion_disk_edge = 40 * M
    bh = black_hole.BlackHole(
        incl=incl, mass=M, outer_edge=outer_accretion_disk_edge
    )
    t_start = time.time()
    ax = bh.plot()
    t_end = time.time()
    print(f"Time to calc and plot: {t_end - t_start:.2f} s")
    # plt.show()

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../")))
    test_bh_isoradial_coverage()
