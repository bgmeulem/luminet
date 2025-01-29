import os
import configparser
from .. import transform
from .. import isoradial
import pandas as pd
import numpy as np
from .. import viz
import matplotlib.pyplot as plt

def test_plot_points():
    test_path = os.path.abspath(os.path.join(__file__.rstrip('.py'), '../../../photons_M=1._incl=85.csv'))
    assert os.path.exists(test_path)
    photons = pd.read_csv(test_path)
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('parameters.ini')
    settings = {}
    for i, section in enumerate(config.sections()):
        settings[section] = {key: eval(val) for key, val in config[section].items()}

    def mask(tri):
        x, y = np.mean(tri.x[tri.triangles], axis=1), np.mean(tri.y[tri.triangles], axis=1)
        a, b = transform.cartesian_to_polar(x, y)
        return np.less_equal(b, inner_isoradial.get_b_from_angle(a))

    fig, ax = plt.subplots(
        subplot_kw={"projection": "polar"}
    )
    inner_isoradial = isoradial.Isoradial(
        incl=85*np.pi/180,
        radius=6.,
        mass=1.,
        params=settings["isoradial_angular_parameters"])
    ax = viz.plot_points(
        photons, 
        ax=ax, 
        levels=100, 
        mask= mask,
)
    plt.show()


if __name__ == "__main__":
    test_plot_points()
