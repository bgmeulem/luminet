import configparser
from typing import Dict, List

import numpy as np

from .viz import colorline

from . import black_hole_math as bhmath
from . import transform


class Isoradial:
    def __init__(
        self,
        radius: float,
        incl: float,
        mass: float,
        order: int = 0,
        acc: float | None = None,
        params: Dict | None = None,
    ):
        assert (
            radius >= 6.0 * mass
        ), """
        Radius should be at least 6 times the mass of the black hole. 
        No orbits are stable below this radius for Swarzschild black holes.
        """
        self.bh_mass = mass  # mass of the black hole containing this isoradial
        self.t = incl  # inclination of observer's plane
        self.radius = radius
        self.order = order
        self.acc = acc
        self.params = params if params is not None else {}
        self.find_redshift_params = {
            "force_redshift_solution": False,  # force finding a redshift solution on the isoradial
            "max_force_iter": 5,  # only make this amount of iterations when forcing finding a solution
        }
        
        self.radii_b = []
        self.angles = []
        self.cartesian_co = self.X, self.Y = [], []
        self.redshift_factors = None

        self.calculate()

    def calculate_coordinates(self):
        """Calculates the angles (alpha) and radii (b) of the photons emitted at radius self.radius as they would appear
        on the observer's photographic plate. Also saves the corresponding values for the impact parameters (P).

        Args:

        Returns:
            tuple: Tuple containing the angles (alpha) and radii (b) for the image on the observer's photographic plate
        """

        angles = []
        impact_parameters = []
        t = np.linspace(0, 2 * np.pi, self.params["angular_precision"])
        for alpha in t:
            b = bhmath.calc_impact_parameter(
                radius=self.radius,
                incl=self.t,
                alpha=alpha,
                bh_mass=self.bh_mass,
                n=self.order,
            )
            assert (
                b > 0
            ), "Impact parameter should be positive, but it wasnt for: R={}, alpha={}, incl={}".format(
                self.radius, alpha, self.t
            )
            if b is np.nan:
                impact_parameters.append(bhmath.ellipse(self.radius, alpha, self.t))
            else:
                impact_parameters.append(b)
            angles.append(alpha)

        # flip image if necessary
        self.angles = np.array(angles)
        self.radii_b = np.array(impact_parameters)
        self.X, self.Y = transform.polar_to_cartesian(
            self.radii_b, self.angles, rotation=-np.pi / 2
        )
        self.cartesian_co = self.X, self.Y
        return angles, impact_parameters

    def calc_redshift_factors(self):
        """Calculates the redshift factor (1 + z) over the line of the isoradial"""
        redshift_factors = [
            bhmath.redshift_factor(
                radius=self.radius, angle=angle, incl=self.t, bh_mass=self.bh_mass, b=b_
            )
            for b_, angle in zip(self.radii_b, self.angles)
        ]
        self.redshift_factors = np.array(redshift_factors)
        return redshift_factors

    def calculate(self):
        self.calculate_coordinates()
        self.calc_redshift_factors()

    def find_angle(self, z) -> List[int]:
        """Returns angle at which the isoradial redshift equals some value z
        Args:
            z: The redshift value z. Do not confuse with redshift factor 1 + z"""
        indices = np.where(
            np.diff(np.sign([redshift - z - 1 for redshift in self.redshift_factors]))
        )[0]
        return [self.angles[i] for i in indices if len(indices)]

    def get_b_from_angle(self, angle: float):
        angles_array = np.array(self.angles)
        radii_b_array = np.array(self.radii_b)

        def find_closest_radii(ang):
            indices = np.argmin(np.abs(angles_array - ang), axis=-1)
            return radii_b_array[indices]

        if isinstance(angle, (float, int)):
            return find_closest_radii(angle)
        else:
            angle = np.asarray(angle)
            return np.vectorize(find_closest_radii, otypes=[float])(angle)

    def plot(self, ax, z, cmap, norm, **kwargs):
        """
        
        Args:
            ax: The axis on which the isoradial should be plotted
            z: The color values to be used
            cmap (str): The colormap to be used
            norm (tuple): The normalization to be used
            **kwargs: Additional arguments to be passed to the colorline function
        """
        ax = colorline(
            ax,
            self.angles,
            self.radii_b,
            z=z,
            cmap=cmap,
            norm=norm,
            **kwargs
        )  

        return ax

    def calc_between(self, ind):
        """
        Calculates the impact parameter and redshift factor at the
        isoradial angle between place ind and ind + 1

        Args:
            ind: the index denoting the location at which the middle point should be calculated. The impact parameter,
            redshift factor, b (observer plane) and alpha (observer/BH coordinate system) will be calculated on the
            isoradial between location ind and ind + 1

        Returns:
            None: Nothing. Updates the isoradial.
        """
        mid_angle = 0.5 * (self.angles[ind] + self.angles[ind + 1])
        b_ = bhmath.calc_impact_parameter(
            self.radius, self.t, mid_angle, self.bh_mass
        )
        z_ = bhmath.redshift_factor(self.radius, mid_angle, self.t, self.bh_mass, b_)
        self.radii_b.insert(ind + 1, b_)
        self.angles.insert(ind + 1, mid_angle)
        self.redshift_factors.insert(ind + 1, z_)

    def force_intersection(self, redshift):
        # TODO: improve this method, currently does not seem to work
        """
        If you know a redshift should exist on the isoradial, use this function to calculate the isoradial until
        it finds it. Useful for when the redshift you're looking for equals (or is close to) the maximum
        redshift along some isoradial line.

        Only works if the redshift can be found within the isoradial begin and end angle.
        """

        if len(self.angles) == 2:
            self.calc_between(0)
        diff = [redshift + 1 - z_ for z_ in self.redshift_factors]
        cross = np.where(np.diff(np.sign(diff)))[0]
        if len(cross):
            return diff  # intersection is found

        it = 0
        while len(cross) == 0 and it < self.find_redshift_params["max_force_iter"]:
            # calc derivatives
            delta = [
                e - b
                for b, e in zip(self.redshift_factors[:-1], self.redshift_factors[1:])
            ]
            # where does the redshift go back up/down before it reaches the redshift we want to find
            initial_guess_indices = np.where(np.diff(np.sign(delta)))[0]
            new_ind = initial_guess_indices[0]  # initialize the initial guess.
            self.calc_between(new_ind)  # insert more accurate solution
            diff = [
                redshift + 1 - z_ for z_ in self.redshift_factors
            ]  # calc new interval
            cross = np.where(np.diff(np.sign(diff)))[0]
            it += 1
            # plt.plot(self.angles, [redshift + 1 - z_ for z_ in self.redshift_factors])
            # plt.axvline(0)
            # plt.show()
        return diff

    def calc_redshift_location_on_ir(self, redshift, cartesian=False):
        """
        Calculates which location on the isoradial has some redshift value (not redshift factor)
        Doest this by means of a midpoint method, with midpoint_steps steps (defined in parameters.ini).
        The (b, alpha, z) coordinates of the isoradial are calculated closer and closer to the desired z.
        It does not matter all that much how high the isoradial resolution is, since midpoint_steps is
        much more important to find an accurate location.
        """

        diff = [redshift + 1 - z_ for z_ in self.redshift_factors]
        # if self.find_redshift_params['force_redshift_solution']:
        #     pass  # TODO, force_intersection does not always seem to work
        #     diff = self.force_intersection(redshift)
        initial_guess_indices = np.where(np.diff(np.sign(diff)))[0]

        angle_solutions = []
        b_solutions = []
        if len(initial_guess_indices):
            for s in range(
                len(initial_guess_indices)
            ):  # generally, two solutions exists on a single isoradial
                new_ind = initial_guess_indices[s]  # initialize the initial guess.
                for _ in range(self.solver_params["midpoint_iterations"]):
                    self.calc_between(new_ind)  # insert more accurate solution
                    diff_ = [
                        redshift + 1 - z_
                        for z_ in self.redshift_factors[new_ind : new_ind + 3]
                    ]  # calc new interval
                    start = np.where(np.diff(np.sign(diff_)))[
                        0
                    ]  # returns index where the sign changes
                    new_ind += start[
                        0
                    ]  # index of new redshift solution in refined isoradial
                # append average values of final interval
                angle_solutions.append(
                    0.5 * (self.angles[new_ind] + self.angles[new_ind + 1])
                )
                b_solutions.append(
                    0.5 * (self.radii_b[new_ind] + self.radii_b[new_ind + 1])
                )
                # update the initial guess indices, as the indexing has changed due to inserted solutions
                initial_guess_indices = [
                    e + self.solver_params["midpoint_iterations"]
                    for e in initial_guess_indices
                ]
            if cartesian:
                return transform.polar_to_cartesian(b_solutions, angle_solutions)
        return angle_solutions, b_solutions

    def plot_redshift(self, fig=None, ax=None, show=True):
        """
        Plots the redshift values along the isoradial line in function of the angle<
        """
        fig_ = fig if fig else plt.figure()
        ax_ = ax if ax else fig_.add_subplot()
        ax_.plot(self.angles, [z - 1 for z in self.redshift_factors])
        plt.title("Redshift values for isoradial\nR={} | M = {}".format(20, M))
        ax_.set_xlim([0, 2 * np.pi])
        if show:
            plt.show()
