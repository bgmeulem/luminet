"""Black hole class for calculating and visualizing a Swarzschild black hole."""

import configparser
import os
from functools import partial
from multiprocessing import Pool
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from luminet import black_hole_math as bhmath
from luminet.isoradial import Isoradial
from luminet.isoredshift import Isoredshift


class BlackHole:
    """Black hole class for calculating and visualizing a Swarzschild black hole.
    """

    def __init__(self, mass=1.0, incl=1.5, acc=1.0, outer_edge=None):
        """
        Args:
            mass (float): Mass of the black hole in natural units :math:`G = c = 1`
            incl (float): Inclination of the observer's plane in radians
            acc (float): Accretion rate in natural units
        """
        self.incl = incl
        """float: Inclination angle of the observer"""
        self.mass = mass
        """float: Mass of the black hole"""
        self.acc = acc  # accretion rate, in natural units
        """float: Accretion rate of the black hole"""
        self.critical_b = 3 * np.sqrt(3) * self.mass
        r"""float: critical impact parameter for the photon sphere :math:`3 \sqrt{3} M`"""
        self.settings = {}  # All settings: see below
        self.ir_parameters = {}
        self.ir_parameters = {"angular_precision": 200}

        self.isoradial_template = partial(
            Isoradial,
            incl=self.incl,
            bh_mass=self.mass,
            params=self.ir_parameters,
        )
        """isoradial_template (partial): partial function to create an isoradial with some radius and order."""

        self.disk_outer_edge = (
            outer_edge if outer_edge is not None else 30.0 * self.mass
        )
        """float: outer edge of the accretion disk. Default is :math:`30 M`."""
        self.disk_inner_edge = 6.0 * self.mass
        """float: inner edge of the accretion disk i.e. :math:`6 M`."""
        self.disk_apparent_outer_edge = self._calc_outer_isoradial()
        """:py:class:`Isoradial`: isoradial that defines the outer edge of the accretion disk."""
        self.disk_apparent_inner_edge = self._calc_inner_isoradial()
        """:py:class:`Isoradial`: isoradial that defines the inner edge of the accretion disk."""
        self.disk_apparent_inner_edge_ghost = self._calc_inner_isoradial(order=1)
        """:py:class:`Isoradial`: isoradial that defines the inner edge of the ghost image."""
        self.disk_apparent_outer_edge_ghost = self._calc_outer_isoradial(order=1)
        """:py:class:`Isoradial`: isoradial that defines the outer edge of the ghost image."""

        self.isoradials = []
        """List[Isoradial]: list of calculated isoradials"""
        self.isoredshifts = []
        """List[Isoredshift]: list of calculated isoredshifts"""

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
        """Calculate isoredshifts for a list of redshift values

        This method creates an array of :py:class:`Isoradials` whose coordinates will be lazily computed.
        These no-coordinate isoradials are used by the :py:class:`Isoredshift` to calculate the locations
        of redshift values along these isoradials.

        Args:
            redshifts (List[float]): list of redshift values
            n_isoradials (int): number of isoradials to use for each isoredshift calculation.

        Returns:
            List[Isoredshift]: list of calculated isoredshifts
        """

        redshifts = redshifts or []
        for redshift in redshifts:
            iz = Isoredshift(
                incl=self.incl,
                redshift=redshift,
                bh_mass=self.mass,
                from_isoradials=[
                    self.isoradial_template(r) for r in np.linspace(6, 30, n_isoradials)
                ],
            )
            self.isoredshifts.append(iz)
        return self.isoredshifts

    def calc_isoradials(
        self, direct_r: List[int | float], ghost_r: List[int | float]
    ) -> List[Isoradial]:
        """Calculate isoradials for a list of radii for the direct image and/or ghost image.

        These calculations are parallellized using the :py:class:`multiprocessing.Pool` class.

        Args:
            direct_r (List[int | float]): list of radii for the direct image
            ghost_r (List[int | float]): list of radii for the ghost image

        Returns:
            List[:py:class:`Isoradial`]: list of calculated isoradials
        """
        # calc ghost images
        with Pool() as pool:
            isoradials = pool.starmap(
                Isoradial,
                [
                    (
                        r,
                        self.incl,
                        self.mass,
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
                        self.mass,
                        0,
                        self.acc,
                        self.ir_parameters,
                    )
                    for r in direct_r
                ],
            )
        self.isoradials.extend(isoradials)
        self.isoradials.sort(key=lambda x: (1 - x.order, x.radius))
        return self.isoradials

    def plot_isoradials(
        self,
        direct_r: List[int | float],
        ghost_r: List[int | float] = None,
        color_by="flux",
        **kwargs,
    ) -> plt.Axes:
        """Plot multiple isoradials.

        This method can be used to plot one or more isoradials.
        If the radii are close to each other, the isoradials will be plotted on top of each other,
        essentially visualizing the entire black hole.

        Args:
            direct_r (List[int | float]): list of radii for the direct image
            ghost_r (List[int | float]): list of radii for the ghost image
            color (str): color scheme for the isoradials. Default is 'flux'.
            **kwargs: additional keyword arguments for the :py:meth:`Isoradial.plot` method.

        Returns:
            :py:class:`~matplotlib.axes.Axes`: The axis with the isoradials plotted.
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
                bhmath.calc_flux_observed(
                    ir.radius, self.acc, self.mass, ir.redshift_factors
                )
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
        """Plot isoredshifts for a list of redshift values

        Args:
            redshifts (List[float]): list of redshift values
            **kwargs: additional keyword arguments for the :py:meth:`Isoredshift.plot` method.

        Returns:
            :py:class:`~matplotlib.axes.Axes`: The axis with the isoredshifts plotted.
        """
        _, ax = self._get_fig_ax()
        self.calc_isoredshifts(redshifts=redshifts)
        for isoredshift in self.isoredshifts:
            ax = isoredshift.plot(ax, **kwargs)
        return ax

    def sample_photons(self, n_points=1000):
        r"""Sample points on the accretion disk.

        Sampling is parallellized using the :py:class:`multiprocessing.Pool` class.

        Each photon has the following properties:

        - ``radius``: radius of the photon on the accretion disk :math:`r`
        - ``alpha``: angle of the photon on the accretion disk :math:`\alpha`
        - ``impact_parameter``: impact parameter of the photon :math:`b`
        - ``z_factor``: redshift factor of the photon :math:`1+z`
        - ``flux_o``: observed flux of the photon :math:`F_o`

        Attention:
            Sampling is not done uniformly, but biased towards the
            center of the accretion disk, as this is where most of the luminosity comes from.
        """
        n_points = int(n_points)
        min_radius_ = self.disk_inner_edge
        max_radius_ = self.disk_outer_edge
        with Pool() as p:
            photons = p.starmap(
                sample_photon,
                [
                    (min_radius_, max_radius_, self.incl, self.mass, 0)
                    for _ in range(n_points)
                ],
            )
        with Pool() as p:
            ghost_photons = p.starmap(
                sample_photon,
                [
                    (min_radius_, max_radius_, self.incl, self.mass, 1)
                    for _ in range(n_points)
                ],
            )

        df = pd.DataFrame(photons)
        df["z_factor"] = bhmath.calc_redshift_factor(
            df["radius"],
            df["alpha"],
            self.incl,
            self.mass,
            df["impact_parameter"],
        )
        df["flux_o"] = bhmath.calc_flux_observed(
            df["radius"], self.acc, self.mass, df["z_factor"]
        )

        df_ghost = pd.DataFrame(ghost_photons)
        df_ghost["z_factor"] = bhmath.calc_redshift_factor(
            df_ghost["radius"],
            df_ghost["alpha"],
            self.incl,
            self.mass,
            df_ghost["impact_parameter"],
        )
        df_ghost["flux_o"] = bhmath.calc_flux_observed(
            df_ghost["radius"], self.acc, self.mass, df_ghost["z_factor"]
        )

        self.photons = df
        self.ghost_photons = df_ghost


def sample_photon(min_r, max_r, incl, bh_mass, n):
    r"""Sample a random photon from the accretion disk

    Each photon has the following properties:

    - ``radius``: radius of the photon on the accretion disk :math:`r`
    - ``alpha``: angle of the photon on the accretion disk :math:`\alpha`
    - ``impact_parameter``: impact parameter of the photon :math:`b`
    - ``z_factor``: redshift factor of the photon :math:`1+z`
    - ``flux_o``: observed flux of the photon :math:`F_o`

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
    b = bhmath.solve_for_impact_parameter(r, incl, alpha, bh_mass, n)
    assert (
        b is not np.nan
    ), f"b is nan for r={r}, alpha={alpha}, incl={incl}, M={bh_mass}, n={n}"
    # f_o = flux_observed(r, acc_r, bh_mass, redshift_factor_)
    return {
        "radius": r,
        "alpha": alpha,
        "impact_parameter": b,
    }
