from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipj, ellipk, ellipkinc

from .solver import improve_solutions

plt.style.use("fivethirtyeight")
colors = plt.rcParams["axes.prop_cycle"].by_key()[
    "color"
]  # six fivethirtyeight themed colors


def calc_q(periastron: float, bh_mass: float) -> float:
    """
    Convert Periastron distance P to the variable Q (easier to work with)
    """
    if periastron < 2.0 * bh_mass:
        return np.nan
    return np.sqrt((periastron - 2.0 * bh_mass) * (periastron + 6.0 * bh_mass))


def calc_b_from_periastron(periastron: float, bh_mass: float) -> float:
    r"""Get impact parameter b from Periastron distance P


    .. math::

        b = \sqrt{\frac{P^3}{P - 2M}}

    Args:
        periastron (float): Periastron distance
        bh_mass (float): Black hole mass

    Attention:
        The paper has a typo here. The fracture on the right hand side equals :math:`b^2`, not :math:`b`.
        You can verify this by filling in :math:`u_2` in Equation 3.
        Only this way do the limits :math:`P -> 3M` and :math:`P >> M` hold true,
        as well as the value for :math:`b_c`

    Returns:
        float: Impact parameter :math:`b`
    """
    if periastron <= 2.0 * bh_mass:
        return np.nan
    return np.sqrt(periastron**3 / (periastron - 2.0 * bh_mass))


def k(periastron: float, bh_mass: float) -> float:
    r"""Calculate the modulus of the elliptic integral

    The modulus is defined as :math:`k = \sqrt{\frac{Q - P + 6M}{2q}}`

    Args:
        periastron (float): Periastron distance
        bh_mass (float): Black hole mass

    Returns:
        float: Modulus of the elliptic integral
    """
    q = calc_q(periastron, bh_mass)
    if q is np.nan:
        return np.nan
    # WARNING: Paper has an error here. There should be brackets around the numerator.
    return np.sqrt((q - periastron + 6 * bh_mass) / (2 * q))


def k_squared(periastron: float, bh_mass: float):
    """Calculate the squared modulus of elliptic integral"""
    q = calc_q(periastron, bh_mass)
    if q is np.nan:
        return np.nan
    # WARNING: Paper has an error here. There should be brackets around the numerator.
    return (q - periastron + 6 * bh_mass) / (2 * q)


def zeta_inf(periastron: float, bh_mass: float) -> float:
    """
    Calculate Zeta_inf for elliptic integral F(Zeta_inf, k)
    """
    q = calc_q(periastron, bh_mass)
    if q is np.nan:
        return np.nan
    arg = (q - periastron + 2 * bh_mass) / (q - periastron + 6 * bh_mass)
    z_inf = np.arcsin(np.sqrt(arg))
    return z_inf


def zeta_r(periastron: float, r: float, bh_mass: float) -> float:
    """
    Calculate the elliptic integral argument Zeta_r for a given value of P and r
    """
    q = calc_q(periastron, bh_mass)
    if q is np.nan:
        return np.nan
    a = (q - periastron + 2 * bh_mass + (4 * bh_mass * periastron) / r) / (
        q - periastron + (6 * bh_mass)
    )
    s = np.arcsin(np.sqrt(a))
    return s


def cos_gamma(_a: float, incl: float) -> float:
    r"""
    Calculate :math:`\cos(\gamma)` from alpha and inclination
    """
    return np.cos(_a) / np.sqrt(np.cos(_a) ** 2 + 1 / (np.tan(incl) ** 2))


def cos_alpha(phi: float, incl: float) -> float:
    """Returns cos(angle) alpha in observer frame given angles phi (black hole frame) and
    inclination (black hole frame)"""
    return (
        np.cos(phi) * np.cos(incl) / np.sqrt((1 - np.sin(incl) ** 2 * np.cos(phi) ** 2))
    )


def alpha(phi: float, incl: float):
    """Returns observer coordinate of photon given phi (BHF) and inclination (BHF)"""
    return np.arccos(cos_alpha(phi, incl))


def calc_sn(
    periastron: float,
    angle: float,
    bh_mass: float,
    incl: float,
    order: int = 0,
) -> float:
    r"""Calculate the elliptic integral in equation 13.

    For direct images, this is:

    ..math::

        \text{sn} \left( \frac{\gamma}{2 \sqrt{P/Q}} + F(\zeta_{\infty}, k) \right)

    For higher order images, this is:

    .. math::

        \text{sn} \left( \frac{\gamma - 2n\pi}{2 \sqrt{P/Q}} - F(\zeta_{\infty}, k) + 2K(k) \right)

    """
    q = calc_q(periastron, bh_mass)
    if q is np.nan:
        return np.nan
    z_inf = zeta_inf(periastron, bh_mass)
    m = k_squared(periastron, bh_mass)  # mpmath takes m = k² as argument.
    ell_inf = ellipkinc(z_inf, m)  # Elliptic integral F(zeta_inf, k)
    g = np.arccos(cos_gamma(angle, incl))

    if order == 0:  # higher order image
        ellips_arg = g / (2.0 * np.sqrt(periastron / q)) + ell_inf
    elif order > 0:  # direct image
        ell_k = ellipk(m)  # calculate complete elliptic integral of mod m = k²
        ellips_arg = (
            (g - 2.0 * order * np.pi) / (2.0 * np.sqrt(periastron / q))
            - ell_inf
            + 2.0 * ell_k
        )
    else:
        raise NotImplementedError(
            "Only 0 and positive integers are allowed for the image order."
        )

    sn, _, _, _ = ellipj(ellips_arg, m)
    return sn


def calc_radius(
    periastron: float,
    ir_angle: float,
    bh_mass: float,
    incl: float,
    order: int = 0,
) -> float:
    """Calculate the radius of origing of a trajectory.

    Args:
        periastron (float): Periastron distance. This is directly related to the observer coordinate frame :math:`b`
        ir_angle (float): Angle of the observer/bh coordinate frame.
        bh_mass (float): Black hole mass
        incl (float): Inclination of the black hole
        order (int): Order of the image. Default is :math:`0` (direct image).

    See also:
        :py:meth:`calc_impact_parameter` to convert periastron distance to impact parameter :math:`b` (observer frame).

    Attention:
        This is not the equation used to solve for the periastron value.
        This is the equation to calculate the radius of the trajectory.
        For the equation that is optimized in order to convert between black hole and observer frame,
        see :py:meth:`eq13_optimizer`.

    Returns:
        float: Radius of the trajectory
    """
    sn = calc_sn(periastron, ir_angle, bh_mass, incl, order)
    q = calc_q(periastron, bh_mass)

    term1 = -(q - periastron + 2.0 * bh_mass)
    term2 = (q - periastron + 6.0 * bh_mass) * sn * sn

    return 4.0 * bh_mass * periastron / (term1 + term2)


def eq13_optimizer(
    periastron: float,
    ir_radius: float,
    ir_angle: float,
    bh_mass: float,
    incl: float,
    order: int = 0,
) -> float:
    r"""Cost function for the optimization of the periastron value.

    This function is optimized to find the periastron value that solves equation 13:

    .. math::

        4 M P - r (Q - P + 2 M) + r (Q - P + 6 M) \text{sn}^2 \left( \frac{\gamma}{2 \sqrt{P/Q}} + F(\zeta_{\infty}, k) \right) = 0

    When the above equation is zero, the radius is correct.

    See also:
        :py:meth:`calc_periastron` to calculate the radius of the trajectory from a periastron value.
    """
    q = calc_q(periastron, bh_mass)
    if q is np.nan:
        return np.nan
    sn = calc_sn(periastron, ir_angle, bh_mass, incl, order)
    term1 = -(q - periastron + 2.0 * bh_mass)
    term2 = (q - periastron + 6.0 * bh_mass) * sn * sn
    zero_opt = 4.0 * bh_mass * periastron - ir_radius * (term1 + term2)
    return zero_opt


def calc_periastron(
    radius: float,
    incl: float,
    alpha: float,
    bh_mass: float,
    order: int = 0,
) -> float:
    r"""Solve eq13 for a periastron value.

    This periastron can be converted to an impact parameter b, yielding the observer frame coordinates (b, alpha).

    Does this by generating range of periastron values, and using a midpoint method
    to iteratively improve which periastron value solves equation 13:

    .. math::

        4 M P - r (Q - P + 2 M) + r (Q - P + 6 M) \text{sn}^2 \left( \frac{\gamma}{2 \sqrt{P/Q}} + F(\zeta_{\infty}, k) \right) = 0

    Args:
        radius (float): radius on the accretion disk (BH frame)
        incl (float): inclination of the black hole
        alpha: angle along the accretion disk (BH frame and observer frame)
        bh_mass (float): mass of the black hole
        midpoint_iterations (int): amount of midpoint iterations to do when searching a periastron value solving eq13
        plot_inbetween (bool): plot
    """

    if radius <= 3 * bh_mass:
        return np.nan

    # Get an initial range for the possible periastron: must span the solution
    min_periastron = (
        3.0 * bh_mass + order * 1e-5
    )  # higher order images go to inf for P -> 3M
    periastron_initial_guess = np.linspace(
        min_periastron,
        radius,  # Periastron cannot be bigger than the radius by definition.
        2,
    )

    # Check if the solution is in the initial range
    y = np.array(
        [
            eq13_optimizer(periastron_guess, radius, alpha, bh_mass, incl, order)
            for periastron_guess in periastron_initial_guess
        ]
    )
    assert not any(np.isnan(y)), "Initial guess contains nan values"

    # If the solution is not in the initial range, return nan
    if np.sign(y[0]) == np.sign(y[1]):
        return np.nan

    kwargs_eq13 = {
        "ir_radius": radius,
        "ir_angle": alpha,
        "bh_mass": bh_mass,
        "incl": incl,
        "order": order,
    }
    periastron = improve_solutions(
        func=eq13_optimizer,
        x=periastron_initial_guess,
        y=y,
        kwargs=kwargs_eq13,
    )
    return periastron


def calc_impact_parameter(
    radius,
    incl,
    alpha,
    bh_mass,
    order=0,
) -> float:
    """Calculate observer coordinates of a BH frame photon.

    Given a value for r (BH frame) and alpha (BH/observer frame), calculate the perigee.
    This perigee value is then converted to an impact parameter b, yielding the observer frame coordinates (b, alpha).

    This method iteratively improves an initial guess of two perigee limits using :py:mod:`solver`.

    Attention:
        Photons that originate from close to the black hole, and the front of the accretion disk, have orbits whose
        perigee is below :math:`3M` (and thus would be absorbed by the black hole), but still make it to the camera in the observer frame.
        These photons are not absorbed by the black hole, since they simply never actually travel the part of their orbit that lies below :math:`3M`

    Args:
        radius (float): radius on the accretion disk (BH frame)
        incl (float): inclination of the black hole
        alpha: angle along the accretion disk (BH frame and observer frame)
        bh_mass (float): mass of the black hole
        n (int): Order of the image. Default is :math:`0` (direct image).
        midpoint_iterations (int): amount of midpoint iterations to do when searching a periastron value solving eq13
    """
    # alpha_obs is flipped alpha/bh if n is odd
    if order % 2 == 1:
        alpha = (alpha + np.pi) % (2 * np.pi)

    periastron_solution = calc_periastron(radius, incl, alpha, bh_mass, order)

    if order == 0 and ((alpha < np.pi / 2) or (alpha > 3 * np.pi / 2)):
        # Photons with small R in the lower half of the image originate from photon orbits that
        # have a perigee < 3M. However, these photons are not absorbed by the black hole and do in fact reach the camera,
        # since they never actually travel this forbidden part of their orbit.
        # --> Return the newtonian limit i.e. just an ellipse, like the rings of saturn that are visible in front of saturn.
        return ellipse(radius, alpha, incl)
    # Photons that have no perigee and are not due to the exception described above are simply absorbed
    if periastron_solution is np.nan:
        return np.nan
    b = calc_b_from_periastron(periastron_solution, bh_mass)
    return b


def phi_inf(periastron, M):
    q = calc_q(periastron, M)
    ksq = (q - periastron + 6.0 * M) / (2.0 * q)
    z_inf = zeta_inf(periastron, M)
    phi = 2.0 * (np.sqrt(periastron / q)) * (ellipk(ksq) - ellipkinc(z_inf, ksq))
    return phi


def mu(periastron, bh_mass):
    return float(2 * phi_inf(periastron, bh_mass) - np.pi)


def ellipse(r, a, incl) -> float:
    """Equation of an ellipse, reusing the definition of cos_gamma.
    This equation can be used for calculations in the Newtonian limit (large P = b, small a)
    or to visualize the equatorial plane."""
    a = (a + np.pi / 2) % (
        2 * np.pi
    )  # rotate 90 degrees for consistency with rest of the code
    major_axis = r
    minor_axis = abs(major_axis * np.cos(incl))
    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
    return minor_axis / np.sqrt((1 - (eccentricity * np.cos(a)) ** 2))


def flux_intrinsic(r, acc, bh_mass):
    r_ = r / bh_mass
    log_arg = ((np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3))) / (
        (np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3))
    )
    f = (
        (3.0 * bh_mass * acc / (8 * np.pi))
        * (1 / ((r_ - 3) * r**2.5))
        * (np.sqrt(r_) - np.sqrt(6) + 3**-0.5 * np.log10(log_arg))
    )
    return f


def flux_observed(r, acc, bh_mass, redshift_factor):
    flux_intr = flux_intrinsic(r, acc, bh_mass)
    flux_observed = flux_intr / redshift_factor**4
    return flux_observed


def redshift_factor(radius, angle, incl, bh_mass, b):
    r"""
    Calculate the gravitational redshift factor (ignoring cosmological redshift):

    .. math::

        1 + z = (1 - \Omega b \cos(\eta)) \left( -g_{tt} - 2 \Omega g_{t\phi} - \Omega^2 g_{\phi\phi} \right)^{-1/2}

    Attention:
        :cite:`Luminet_1979` does not have the correct equation for the redshift factor.
        The correct formula is given above.
    """
    # gff = (radius * np.sin(incl) * np.sin(angle)) ** 2
    # gtt = - (1 - (2. * M) / radius)
    z_factor = (
        1.0 + np.sqrt(bh_mass / (radius**3)) * b * np.sin(incl) * np.sin(angle)
    ) * (1 - 3.0 * bh_mass / radius) ** -0.5
    return z_factor


if __name__ == "__main__":
    M = 1
    print(calc_periastron(3, 10, 0, 1))

    # writeFramesEq13(5, solver_params=solver_params)
