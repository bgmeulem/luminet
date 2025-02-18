import configparser, os
from functools import partial
from multiprocessing import Pool
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import black_hole_math as bhmath
from . import transform
from .isoradial import Isoradial
from .isoredshift import Isoredshift


class BlackHole:
    def __init__(self, mass=1.0, inclination=1.5, acc=1.0, outer_edge=None):
        """Initialise black hole with mass and accretion rate

        Args:
            mass (float): Mass of the black hole in natural units :math:`G = c = 1`
            inclination (float): Inclination of the observer's plane in radians
            acc (float): Accretion rate in natural units
        """
        self.incl = inclination
        self.M = mass
        self.acc = acc  # accretion rate, in natural units
        self.critical_b = 3 * np.sqrt(3) * self.M
        self.settings = {}  # All settings: see below
        self.ir_parameters = {}
        self.angular_properties = {}
        self.iz_solver_params = {}
        self._read_parameters()

        self.isoradial_template = partial(
            Isoradial, incl=self.incl, bh_mass=self.M, params=self.ir_parameters
        )
        self.disk_outer_edge = outer_edge if outer_edge is not None else 30.0 * self.M
        self.disk_inner_edge = 6.0 * self.M
        self.disk_apparent_outer_edge = self._calc_outer_isoradial()
        self.disk_apparent_inner_edge = self._calc_inner_isoradial()
        self.disk_apparent_inner_edge_ghost = self._calc_inner_isoradial(order=1)
        self.disk_apparant_outer_edge_ghost = self._calc_outer_isoradial(order=1)

        self.isoradials = []
        self.isoredshifts = []

    def _read_parameters(self):
        pardir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        config = configparser.ConfigParser(inline_comment_prefixes="#")
        config.read(os.path.join(pardir, "parameters.ini"))
        for i, section in enumerate(config.sections()):
            self.settings[section] = {
                key: eval(val) for key, val in config[section].items()
            }
        self.ir_parameters = self.settings["isoradial_angular_parameters"]
        self.iz_solver_params = self.settings["isoredshift_solver_parameters"]

    def plot_photon_sphere(self, ax=None, color="grey", n_steps=100):
        """Plot the photon sphere, defined as a sphere with radius 3 * sqrt(3) * M"""
        if ax is None:
            _, ax = plt.subplots()

        a = np.linspace(0, 2 * np.pi, n_steps)
        b = [self.critical_b] * n_steps
        x, y = transform.polar_to_cartesian(b, a, rotation=-np.pi / 2)
        ax.plot(x, y, color=color)
        return ax

    def _calc_inner_isoradial(self, order=0):
        """Calculate the isoradial that defines the inner edge of the accretion disk"""
        ir = self.isoradial_template(radius=self.disk_inner_edge, order=order)
        ir.calculate()
        return ir

    def _calc_outer_isoradial(self, order=0):
        """Calculate the isoradial that defines the outer edge of the accretion disk"""
        ir = self.isoradial_template(radius=self.disk_outer_edge, order=order)
        ir.calculate()
        return ir

    def _calc_apparent_outer_edge(self, angle):
        return self.disk_apparent_outer_edge.get_b_from_angle(angle)

    def _calc_apparent_inner_edge(self, angle):
        """Get the apparent inner edge of the accretion disk at some angle"""
        return self.disk_apparent_inner_edge.get_b_from_angle(angle)

    def _get_fig_ax(self, polar=True):
        if polar:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            ax.set_theta_zero_location("S")  # theta=0 at the bottom
        else:
            fig, ax = plt.subplots()
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.grid()
        plt.axis("off")  # command for hiding the axis.
        # Remove padding between the figure and the axes
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return fig, ax

    def calc_isoredshifts(self, redshifts=None, n_isoradials=100):
        redshifts = redshifts or []
        for redshift in redshifts:
            iz = Isoredshift(
                inclination=self.incl,
                redshift=redshift,
                bh_mass=self.M,
                solver_parameters=self.iz_solver_params,
                from_isoradials=[self.isoradial_template(r) for r in np.linspace(6, 30, n_isoradials)],
            )
            self.isoredshifts.append(iz)
        return self.isoredshifts

    def calc_isoradials(self, direct_r: List[int | float], ghost_r: List[int | float]):
        # calc ghost images
        with Pool() as pool:
            isoradials = pool.starmap(
                Isoradial,
                [
                    (
                        r,
                        self.incl,
                        self.M,
                        1,
                        self.acc,
                        self.ir_parameters,
                    )
                    for r in ghost_r
                ],
            )
        self.isoradials.extend(isoradials)

        with Pool() as pool:
            isoradials = pool.starmap(
                Isoradial,
                [
                    (
                        r,
                        self.incl,
                        self.M,
                        0,
                        self.acc,
                        self.ir_parameters,
                    )
                    for r in direct_r
                ],
            )
        self.isoradials.extend(isoradials)

        self.isoradials.sort(key=lambda x: (1 - x.order, x.radius))

    def plot_isoradials(
        self,
        direct_r: List[int | float],
        ghost_r: List[int | float] = None,
        color_by="flux",
        **kwargs,
    ):
        """Given an array of radii for the direct image and/or ghost image, plots the corresponding
        isoradials.

        """

        ghost_r = ghost_r if ghost_r is not None else []
        self.calc_isoradials(direct_r, ghost_r)
        _, ax = self._get_fig_ax()

        if color_by == "redshift":
            if not "cmap" in kwargs:
                kwargs["cmap"] = "RdBu_r"
            zs = [ir.redshift_factors for ir in self.isoradials]
            mx = np.max([np.max(z) for z in zs])
            norm = (-mx, mx)
        elif color_by == "flux":
            if not "cmap" in kwargs:
                kwargs["cmap"] = "Greys_r"
            zs = [
                bhmath.flux_observed(ir.radius, self.acc, self.M, ir.redshift_factors)
                for ir in self.isoradials
            ]
            mx = np.max([np.max(z) for z in zs])
            norm = (0, mx)
        
        for z, ir in zip(zs, self.isoradials):
            if ir.radius in direct_r and ir.order == 0:
                ax = ir.plot(ax, z=z, norm=norm, zorder= ir.radius, **kwargs)
            elif ir.radius in ghost_r and ir.order == 1:
                ax = ir.plot(ax, z=z, norm=norm, zorder= -ir.radius, **kwargs)

        biggest_ir = sorted(self.isoradials, key=lambda x: x.radius)[-1]
        ax.set_ylim((0, 1.1*max(biggest_ir.radii_b)))
        return ax

    def plot(self, n_isoradials=100, **kwargs):
        """Plot the black hole

        This is a wrapper method to plto the black hole.
        It simply calls the :py:meth:`plot_isoradials` method with a dense range of isoradials.

        Args:
            n_isoradials (int): number of isoradials to plot

        Returns:
            :py:class:`~matplotlib.axes.Axes`: The axis with the isoradials plotted.
        """

        radii = np.linspace(self.disk_inner_edge, self.disk_outer_edge, n_isoradials)
        ax = self.plot_isoradials(direct_r=radii, ghost_r=radii, color_by="flux", **kwargs)
        return ax

    def plot_isoredshifts(self, redshifts=None, **kwargs):
        _, ax = self._get_fig_ax()
        self.calc_isoredshifts(redshifts=redshifts)
        for isoredshift in self.isoredshifts:
            ax = isoredshift.plot(ax, **kwargs)
        return ax

    def sample_photons(self, n_points=1000):
        """
        Samples points on the accretion disk. This sampling is not done uniformly, but a bias is added towards the
        center of the accretion disk, as the observed flux is exponentially bigger here and this needs the most
        precision.
        """
        n_points = int(n_points)
        min_radius_ = self.disk_inner_edge
        max_radius_ = self.disk_outer_edge
        with Pool() as p:
            photons = p.starmap(
                sample_photon,
                [
                    (min_radius_, max_radius_, self.incl, self.M, 0)
                    for _ in range(n_points)
                ],
            )
        with Pool() as p:
            ghost_photons = p.starmap(
                sample_photon,
                [
                    (min_radius_, max_radius_, self.incl, self.M, 1)
                    for _ in range(n_points)
                ],
            )

        df = pd.DataFrame(photons)
        df["z_factor"] = bhmath.redshift_factor(
            df["radius"], df["alpha"], self.incl, self.M, df["impact_parameter"]
        )
        df["flux_o"] = bhmath.flux_observed(
            df["radius"], self.acc, self.M, df["z_factor"]
        )

        df_ghost = pd.DataFrame(ghost_photons)
        df_ghost["z_factor"] = bhmath.redshift_factor(
            df_ghost["radius"],
            df_ghost["alpha"],
            self.incl,
            self.M,
            df_ghost["impact_parameter"],
        )
        df_ghost["flux_o"] = bhmath.flux_observed(
            df_ghost["radius"], self.acc, self.M, df_ghost["z_factor"]
        )

        self.photons = df
        self.ghost_photons = df_ghost

    def plot_isoredshifts_from_points(self, levels=None, extension="png"):
        # TODO add ghost image

        if levels is None:
            levels = [
                -0.2,
                -0.15,
                -0.1,
                -0.05,
                0.0,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.5,
                0.75,
            ]

        _fig, _ax = self._get_fig_ax()
        points = pd.read_csv(f"points_incl={int(round(self.incl * 180 / np.pi))}.csv")
        br = self._calc_inner_isoradial()
        color_map = plt.get_cmap("RdBu_r")

        # points1 = addBlackRing(self, points1)
        levels_ = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75]
        _ax.tricontour(
            points["X"],
            points["Y"],
            [e for e in points["z_factor"]],
            cmap=color_map,
            norm=plt.Normalize(0, 2),
            levels=[e + 1 for e in levels_],
            nchunk=2,
            linewidths=2,
        )
        _ax.fill_between(br.X, br.Y, color="black", zorder=2)
        plt.show()
        _fig.savefig(
            f"Plots/Isoredshifts_incl={str(int(180 * self.incl / np.pi)).zfill(3)}.{extension}",
            facecolor="black",
            dpi=300,
        )
        return _fig, _ax


def sample_photon(min_r, max_r, incl, bh_mass, n):
    """Sample a random photon from the accretion disk

    Attention:
        Photons are not sampled uniformly on the accretion disk, but biased towards the center.
        Black holes have more flux delta towards the center, and thus we need more precision there.
        This makes the triangulation with hollow mask in the center also very happy.

    Args:
        min_r: minimum radius of the accretion disk
        max_r: maximum radius of the accretion disk
        incl: inclination of the observer wrt the disk
        bh_mass: mass of the black hole
        n: order of the isoradial
    """
    alpha = np.random.random() * 2 * np.pi

    # Bias sampling towards circle center (even sampling would be sqrt(random))
    r = min_r + (max_r - min_r) * np.random.random()
    b = bhmath.calc_impact_parameter(r, incl, alpha, bh_mass, n)
    assert (
        b is not np.nan
    ), f"b is nan for r={r}, alpha={alpha}, incl={incl}, M={bh_mass}, n={n}"
    # f_o = flux_observed(r, acc_r, bh_mass, redshift_factor_)
    return {
        "radius": r,
        "alpha": alpha,
        "impact_parameter": b,
    }
