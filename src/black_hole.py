import configparser
from functools import partial
from multiprocessing import Pool
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import black_hole_math as bhmath
from . import transform
from .isoradial import Isoradial


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
            Isoradial, incl=self.incl, mass=self.M, params=self.ir_parameters
        )
        self.disk_outer_edge = outer_edge if outer_edge is not None else 30.0 * self.M
        self.disk_inner_edge = 6.0 * self.M
        self.disk_apparent_outer_edge = self.calc_outer_isoradial()
        self.disk_apparent_inner_edge = self.calc_inner_isoradial()
        self.disk_apparent_inner_edge_ghost = self.calc_inner_isoradial(order=1)
        self.disk_apparant_outer_edge_ghost = self.calc_outer_isoradial(order=1)

        self.isoradials = []
        self.isoredshifts = []

    def _read_parameters(self):
        config = configparser.ConfigParser(inline_comment_prefixes="#")
        config.read("parameters.ini")
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

    def calc_inner_isoradial(self, order=0):
        """Calculate the isoradial that defines the inner edge of the accretion disk"""
        ir = self.isoradial_template(radius=self.disk_inner_edge, order=order)
        ir.calculate()
        return ir

    def calc_outer_isoradial(self, order=0):
        """Calculate the isoradial that defines the outer edge of the accretion disk"""
        ir = self.isoradial_template(radius=self.disk_outer_edge, order=order)
        ir.calculate()
        return ir

    def calc_apparent_outer_edge(self, angle):
        return self.disk_apparent_outer_edge.get_b_from_angle(angle)

    def calc_apparent_inner_edge(self, angle):
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

    def plot_apparent_inner_edge(self, ax, n_steps=100, **kwargs):
        """Plot the apparent inner edge of the black hole

        This edge is bounded at the top by the ghost image, wrapping close to the photon sphere,
        and bounded at the bottom by the inner edge of the accretion disk.
        """
        if ax is None:
            _, ax = plt.subplots()

        bottom_quart = np.linspace(0, np.pi / 2, n_steps // 4)
        quart_ell = [bhmath.ellipse(5.2, a, self.incl) for a in bottom_quart]
        b = quart_ell.copy()
        b.extend([5.2] * (n_steps // 2))
        b.extend(quart_ell[::-1])
        x, y = transform.polar_to_cartesian(
            b, np.linspace(0, 2 * np.pi, n_steps), rotation=-np.pi / 2
        )
        ax.plot(x, y, **kwargs)
        return ax

    def calc_isoredshifts(self, redshifts=None):

        def get_dirty_isoradials(__bh):
            # an array of quick and dirty isoradials for the initial guesses of redshifts
            isoradials = []  # for initial guesses
            for radius in np.linspace(
                __bh.disk_inner_edge,
                __bh.disk_outer_edge,
                __bh.iz_solver_params["initial_radial_precision"],
            ):
                isoradial = Isoradial(radius, __bh.t, __bh.M, params=__bh.ir_parameters)
                isoradials.append(isoradial)
            return isoradials

        redshifts = redshifts or []
        dirty_isoradials = get_dirty_isoradials(self)
        for redshift in redshifts:
            dirty_ir_copy = dirty_isoradials.copy()
            # spawn an isoredshift instance and calc coordinates based on dirty isoradials
            iz = Isoredshift(
                inclination=self.incl,
                redshift=redshift,
                bh_mass=self.M,
                solver_parameters=self.iz_solver_params,
                from_isoradials=dirty_ir_copy,
            )
            # iteratively improve coordinates and closing tip of isoredshift
            iz.improve()
            self.isoredshifts[redshift] = iz
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
        color="flux",
        **kwargs,
    ):
        """Given an array of radii for the direct image and/or ghost image, plots the corresponding
        isoradials.

        """

        ghost_r = ghost_r if ghost_r is not None else []
        self.calc_isoradials(direct_r, ghost_r)
        _, ax = self._get_fig_ax()

        if color == "redshift":
            cmap = "RdBu_r"
            zs = [ir.redshift_factors for ir in self.isoradials]
            mx = np.max([np.max(z) for z in zs])
            norm = (-mx, mx)
        elif color == "flux":
            cmap = "Greys_r"
            zs = [
                bhmath.flux_observed(ir.radius, self.acc, self.M, ir.redshift_factors)
                for ir in self.isoradials
            ]
            mx = np.max([np.max(z) for z in zs])
            norm = (0, mx)
        else:
            raise ValueError(
                f"Color {color} not recognized. Options are 'flux' or 'redshift'"
            )

        for z, ir in zip(zs, self.isoradials):
            ax = ir.plot(ax, z=z, cmap=cmap, norm=norm, **kwargs)

        biggest_ir = sorted(self.isoradials, key=lambda x: x.radius)[-1]
        ax.set_ylim((0, max(biggest_ir.radii_b)))
        return ax

    def plot(self, n_isoradials=100):
        """Plot a black hole by plotting a full range of isoradials"""

        radii = np.linspace(self.disk_inner_edge, self.disk_outer_edge, n_isoradials)
        ax = self.plot_isoradials(direct_r=radii, ghost_r=radii, color="flux")
        return ax

    def write_frames(self, func, direct_r=None, ghost_r=None, step_size=5):
        """
        Given some function that produces  fig and ax, this method sets increasing values for the inclination,
        plots said function and write it out as a frame.
        """
        if ghost_r is None:
            ghost_r = [6, 10, 20, 30]
        if direct_r is None:
            direct_r = [6, 10, 20, 30]
        steps = np.linspace(0, 180, 1 + (0 - 180) // step_size)
        for a in tqdm(steps, position=0, desc="Writing frames"):
            self.incl = a
            fig_, ax_ = func(direct_r, ghost_r, ax_lim=self.plot_params["ax_lim"])
            name = self.plot_params["title"].replace(" ", "_")
            name = name.replace("°", "")
            fig_.savefig(
                "movie/" + name, dpi=300, facecolor=self.plot_params["face_color"]
            )
            plt.close()  # to not destroy your RAM

    def plot_isoredshifts(self, redshifts=None, plot_core=False):
        _fig, _ax = self._get_fig_ax()  # make new figure

        bh.calc_isoredshifts(redshifts=redshifts).values()

        for redshift, irz in self.isoredshifts.items():
            r_w_s, r_wo_s = irz.split_co_on_solutions()
            if len(r_w_s.keys()):
                split_index = irz.split_co_on_jump()
                if split_index is not None:
                    plt.plot(
                        irz.y[:split_index],
                        [-e for e in irz.x][:split_index],
                    )
                    plt.plot(
                        irz.y[split_index + 1 :],
                        [-e for e in irz.x][split_index + 1 :],
                    )
                else:
                    plt.plot(
                        irz.y,
                        [-e for e in irz.x],
                    )  # todo: why do i need to flip x

        if plot_core:
            _ax = self.plot_apparent_inner_edge(_ax, linestyle="-")
        plt.suptitle("Isoredshift lines for M={}".format(self.M))
        plt.show()
        return _fig, _ax

    def sample_points(self, n_points=1000):
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
        br = self.calc_inner_isoradial()
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
