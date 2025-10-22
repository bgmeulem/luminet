"""Black hole class for calculating and visualizing a Swarzschild black hole."""

from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from luminet import black_hole_math as bhmath
from luminet.isoradial import Isoradial
from luminet.isoredshift import Isoredshift


class BlackHole:
    """Black hole class for calculating and visualizing a Swarzschild black hole.
    """

    def __init__(
            self, 
            mass=1.0, 
            incl=1.4, 
            acc=1.0, 
            outer_edge=None,
            angular_resolution=200,
            radial_resolution=200
    ):
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
        self.max_flux = self._calc_max_flux()
        """float: Maximum flux of the black hole, as emitted by the isoriadial R ~ 9.55. See :cite:t:`Luminet_1979`"""
        self.critical_b = 3 * np.sqrt(3) * self.mass
        r"""float: critical impact parameter for the photon sphere :math:`3 \sqrt{3} M`"""
        self.angular_resolution = angular_resolution
        """int: Angular resolution to use when calculating or plotting the black hole or related properties. Default is 200."""
        self.radial_resolution = radial_resolution
        """int: Radial resolution to use when calculating or plotting the black hole or related properties. Default is 200."""


        self.isoradial_template = partial(
            Isoradial,
            incl=self.incl,
            bh_mass=self.mass,
            angular_resolution=self.angular_resolution,
        )
        """callable: partial function to create an isoradial with some radius and order."""

        self.disk_outer_edge = (
            outer_edge if outer_edge is not None else 30.0 * self.mass
        )
        """float: outer edge of the accretion disk. Default is :math:`30 M`."""
        self.disk_inner_edge = 6.0 * self.mass
        """float: inner edge of the accretion disk i.e. :math:`6 M`."""
        self.disk_apparent_outer_edge = self._calc_outer_isoradial()
        """Isoradial: isoradial that defines the outer edge of the accretion disk."""
        self.disk_apparent_inner_edge = self._calc_inner_isoradial()
        """Isoradial: isoradial that defines the inner edge of the accretion disk."""
        self.disk_apparent_inner_edge_ghost = self._calc_inner_isoradial(order=1)
        """Isoradial: isoradial that defines the inner edge of the ghost image."""
        self.disk_apparent_outer_edge_ghost = self._calc_outer_isoradial(order=1)
        """Isoradial: isoradial that defines the outer edge of the ghost image."""

        self.isoradials = []
        """List[Isoradial]: list of calculated isoradials"""
        self.isoredshifts = []
        """List[Isoredshift]: list of calculated isoredshifts"""

    def _calc_max_flux(self):
        r"""Get the maximum intrinsic flux emitted by the black hole
        
        Max flux happens at radius ~ 9.55, which yields a max flux of:
        :math:`\frac{3M\dot{M}}{8\pi}*1.146*10^{-4}`.

        See also:
            Please refer to Eq15 in :cite:t:`Luminet_1979` for more info on this magic number.
        """
        return 3 * self.mass * self.acc * 1.146e-4 / (8 * np.pi)

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

    def _get_fig_ax(self, polar=True) -> Tuple[Figure, Axes]:
        """Fetch a figure set up for plotting black holes and associated attributes.

        This figure has the following properties:

        - Polar coordinates
        - Black background
        - No axes
        - Scaled between :math:`R = [0, max]`, where max is the largest :math:`b` of the largest isoradial

        This function should be called after all necessary isoradials to plot something have already been calculated (for proper scaling).
        """
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
        sorted_ir_direct = sorted(
            [ir for ir in self.isoradials if ir.order == 0], 
            key=lambda x: x.radius)
        sorted_ir_ghost = sorted(
            [ir for ir in self.isoradials if ir.order == 1], 
            key=lambda x: x.radius)
        if sorted_ir_direct: biggest_ir = sorted_ir_direct[-1]
        elif sorted_ir_ghost: biggest_ir = sorted_ir_ghost[-1]
        else: raise ValueError("Can't fetch the figure, as no isoradials have been calculated yet.")
        ax.set_ylim((0, 1.1*max(biggest_ir.impact_parameters)))
        return fig, ax
    
    def _is_ir_calculated(self, radius, order):
        return radius in [ir.radius for ir in self.isoradials if ir.order == order]

    def _calc_isoredshift(self, redshift, order=0):
        """Calculate a single isoredshift

        Args:
            redshift (int|float): The redshift value for which to calculate the `Isoredshift` for.
            order (int, optional): The order of the image associated with this `Isoredshift`.
        """
        direct_irs = [ir for ir in self.isoradials if ir.order == order]
        with Pool() as p:
            solutions = p.starmap(
                _call_calc_redshift_locations,
                [(ir, redshift) for ir in direct_irs]
            )
        angle_pairs, b_pairs = zip(*solutions)

        iz = Isoredshift(
            redshift=redshift, 
            angles=angle_pairs, 
            impact_parameters=b_pairs, 
            ir_radii=[ir.radius for ir in direct_irs]
        )

        self.isoredshifts.append(iz)
    
    def calc_isoredshifts(self, redshifts=None, order=0):
        """Calculate isoredshifts for a list of redshift values

        This method creates an array of `Isoradials` whose coordinates will be lazily computed.
        These no-coordinate isoradials are used by the `Isoredshift` to calculate the locations
        of redshift values along these isoradials.

        Args:
            redshifts (List[float]): list of redshift values

        Returns:
            List[:class:`~luminet.isoredshift.Isoredshift`]: list of calculated isoredshifts
        """
        # Don't recalculate isoredshifts that have already been calculated
        redshifts = [z for z in redshifts if z not in [irz.redshift for irz in self.isoredshifts]]

        radii = np.linspace(self.disk_inner_edge, self.disk_outer_edge, self.radial_resolution)
        if order == 0: self.calc_isoradials(direct_r=radii, ghost_r=[])
        elif order == 1: self.calc_isoradials(direct_r=[], ghost_r=radii)
        else: raise ValueError("Orders other than 0 (direct) or 1 (ghost) are not supported yet.")

        for z in redshifts: self._calc_isoredshift(z, order=order)

    def calc_isoradials(
        self, direct_r: List[int | float], ghost_r: List[int | float]
    ) -> List[Isoradial]:
        """Calculate isoradials for a list of radii for the direct image and/or ghost image.

        These calculations are parallellized using the :py:class:`multiprocessing.Pool` class.

        Args:
            direct_r (List[int | float]): list of radii for the direct image
            ghost_r (List[int | float]): list of radii for the ghost image

        Returns:
            List[:class:`~luminet.isoradial.Isoradial`]: list of calculated isoradials
        """
        # Filter out isoradials that have already been calculated:
        direct_r = [r for r in direct_r if not self._is_ir_calculated(r, order=0)]
        ghost_r = [r for r in ghost_r if not self._is_ir_calculated(r, order=1)]

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
                        self.angular_resolution,
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
                        self.angular_resolution,
                    )
                    for r in direct_r
                ],
            )
        self.isoradials.extend(isoradials)
        self.isoradials.sort(key=lambda x: (1 - x.order, x.radius))

    def plot_isoradials(
        self,
        direct_r: List[int | float],
        ghost_r: List[int | float] | None = None,
        color_by="flux",
        ax: Axes | None = None,
        **kwargs,
    ) -> Axes:
        """Plot multiple isoradials.

        This method can be used to plot one or more isoradials.
        If the radii are close to each other, the isoradials will be plotted on top of each other,
        essentially visualizing the entire black hole.

        Args:
            direct_r (List[int | float]): list of radii for the direct image
            ghost_r (List[int | float]): list of radii for the ghost image
            color (str): color scheme for the isoradials. Default is 'flux'.
            kwargs (optional): additional keyword arguments for the :meth:`luminet.isoradial.Isoradial.plot` method.
            ax (:class:`~matplotlib.axes.Axes`, optional): Axes object to plot on.
                Useful for when you want to plot multiple things one a single canvas.


        Example::

            from luminet.black_hole import BlackHole

            direct_irs = [6, 10, 15, 20]
            ghost_irs = [6, 20, 50, 100]  # ghost_r can go to infinity
            ax = bh.plot_isoradials(direct_irs, ghost_irs, lw=1, colors='white')

        .. image:: /../_static/_images/isoradials.png
           :align: center

        Returns:
            :py:class:`~matplotlib.axes.Axes`: The plotted isoradials.
        """

        ghost_r = ghost_r if ghost_r is not None else []
        self.calc_isoradials(direct_r, ghost_r)
        if ax is None: _, ax = self._get_fig_ax()

        if color_by == "redshift":
            if not "cmap" in kwargs:
                kwargs["cmap"] = "RdBu_r"
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

        return ax

    def plot(self, **kwargs) -> Axes:
        """Plot the black hole

        This is a wrapper method to plot the black hole.
        It simply calls the :meth:`~luminet.black_hole.BlackHole.plot_isoradials` method with a dense range of isoradials,
        as specified in :attr:`radial_resolution`

        Example::

            from luminet.black_hole import BlackHole

            bh = BlackHole()
            bh.plot()

        .. image:: /../_static/_images/bh.png
           :align: center

        Returns:
            :class:`~matplotlib.axes.Axes`: The axis with the isoradials plotted.
        """

        radii = np.linspace(self.disk_inner_edge, self.disk_outer_edge, self.radial_resolution)
        ax = self.plot_isoradials(direct_r=radii, ghost_r=radii, color_by="flux", **kwargs)
        return ax

    def plot_isoredshifts(self, redshifts=None, order=0, ax=None, **kwargs) -> Axes:
        """Plot isoredshifts for a list of redshift values

        Args:
            redshifts (List[float]): list of redshift values
            kwargs (optional): additional keyword arguments for the :meth:`luminet.isoredshift.Isoredshift.plot` method.
            order (int): The order of the image to plot siofluxlines for. Default is :math:`0`.
            ax (:class:`~matplotlib.axes.Axes`, optional): Axes object to plot on.
                Useful for when you want to plot multiple things one a single canvas.

        Example::

            from luminet.black_hole import BlackHole

            bh = BlackHole()
            redshifts = [-.2, -.1, 0., .1, .2, .3, .4]
            ax = bh.plot_isoredshifts(redshifts, c='white')
            ax = bh.disk_apparent_inner_edge.plot(ax=ax, c='white')

        .. image:: /../_static/_images/isoredshifts.png
           :align: center

        Returns:
            :py:class:`~matplotlib.axes.Axes`: The plotted isoredshifts.
        """
        self.calc_isoredshifts(redshifts=redshifts, order=order)
        if ax is None: fig, ax = self._get_fig_ax()
        for isoredshift in self.isoredshifts:
            ax = isoredshift.plot(ax, **kwargs)
        return ax
    
    def plot_isofluxlines(self, mask_inner=True, mask_outer=True, normalize=True, order=0, ax=None, **kwargs) -> Axes:
        """Plot lines of equal flux.

        Args:
            normalize (bool): Whether to normalize the fluxlines by the maximum flux or not. Defaults to True.
            mask_inner (bool): 
                Whether to place a mask over the apparent inner edge, where the direct image produces no flux. 
                Useful to mitigate matplotlib tricontour artifacts. Default is ``True``
            mask_outer (bool): 
                Whether to place a mask over the apparent outer edge, where we are not capturing photons from.
                Useful to mitigate matplotlib tricontour artifacts. Default is ``True``.
            order (int): The order of the image to plot siofluxlines for. Default is :math:`0`.
            ax (:class:`~matplotlib.axes.Axes`, optional): Axes object to plot on.
                Useful for when you want to plot multiple things one a single canvas.
            kwargs (optional): Other keyword arguments to pass to :py:func:`~matplotlib.pyplot.tricontour`.

        Hint:
            Normalizing the isofluxlines makes it easier to define specific levels.

        Hint:
            Levels in logspace tend to produce nicer results than linearly increasing levels.

        Example::

            from luminet.black_hole import BlackHole

            bh = BlackHole(incl=1.4, radial_resolution=200)
            levels = [.05, .1, .15, .2, .25, .3, .6, .9, 1.2, 1.5, 1.8, 2.1]
            ax = bh.plot_isofluxlines(colors='white', levels=levels, linewidths=1)

        .. image:: /../_static/_images/isofluxlines.png
           :align: center

        Returns:
            :class:`matplotlib.axes.Axes`: The plotted isofluxlines.
        """
        radii = np.linspace(self.disk_inner_edge, self.disk_outer_edge, self.radial_resolution)
        if order == 0: self.calc_isoradials(direct_r=radii, ghost_r=[])
        elif order == 1: self.calc_isoradials(direct_r=[], ghost_r=radii)
        else: raise ValueError("Orders other than 0 (direct) or 1 (ghost) are not supported yet.")

        irs = [ir for ir in self.isoradials if ir.order == order]
        a = np.array([float(angle) for ir in irs for angle in ir.angles])
        b = np.array([float(r) for ir in irs for r in ir.impact_parameters])
        zs = np.array([
            flux 
            for ir in irs 
            for flux in bhmath.calc_flux_observed(
                r=ir.radius, 
                acc=self.acc, 
                bh_mass=self.mass, 
                redshift_factor=ir.redshift_factors
            )
            ])
        if normalize: zs /= self.max_flux

        if ax is None: fig, ax = self._get_fig_ax()
        contour = plt.tricontour(
            a, b, zs, 
            **kwargs
            )
        if mask_inner:
            ax.fill_between(
                self.disk_apparent_inner_edge.angles,
                0, 
                self.disk_apparent_inner_edge.impact_parameters,
                color='k',
                zorder=len(contour.levels) + 1
                )
        if mask_outer:
            max_r = ax.get_ylim()[-1]
            ax.fill_between(
                self.disk_apparent_outer_edge.angles,
                self.disk_apparent_outer_edge.impact_parameters,
                max_r, 
                color='k',
                zorder=len(contour.levels) + 1
                )
        return ax

    def sample_photons(self, n_points=1000) -> Tuple[pd.DataFrame]:
        r"""Sample points on the accretion disk.

        Photons are appended as class-level attributes.
        Each photon is a :class:`pandas.series.Series` with the following properties:

        - ``radius``: radius of the photon on the accretion disk :math:`r`
        - ``alpha``: angle of the photon on the accretion disk :math:`\alpha`
        - ``impact_parameter``: impact parameter of the photon :math:`b`
        - ``z_factor``: redshift factor of the photon :math:`1+z`
        - ``flux_o``: observed flux of the photon :math:`F_o`

        Args:
            n_points (int): Amount of photons to sample.

        Attention:
            Sampling is not done uniformly, but biased towards the
            center of the accretion disk, as this is where most of the luminosity comes from.

        Returns:
            Tuple[:class:`~pandas.dataframe.DataFrame`]:
                Dataframes containing photons for both direct and ghost image.
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

        return self.photons, self.ghost_photons


def sample_photon(min_r, max_r, incl, bh_mass, n):
    r"""Sample a random photon from the accretion disk

    Each photon is a dictionary with the following properties:

    - ``radius``: radius of the photon on the accretion disk :math:`r`
    - ``alpha``: angle of the photon on the accretion disk :math:`\alpha`
    - ``impact_parameter``: impact parameter of the photon :math:`b`
    - ``z_factor``: redshift factor of the photon :math:`1+z`

    This function is used in :meth:`~luminet.black_hole.BlackHole.sample_photons` to sample
    photons on the accretion disk of a black hole in a parallellized manner.

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

    Returns:
        Dict: Dictionary containing all basic properties of a single photon from the accretion disk.
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

def _call_calc_redshift_locations(ir, redshift):
    """Helper function for multiprocessing"""
    return ir.interpolate_redshift_locations(redshift)
