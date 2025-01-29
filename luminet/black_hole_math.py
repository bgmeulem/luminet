import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipj, ellipk, ellipkinc
from typing import List
from .solver import improve_solutions_midpoint

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors


def calc_q(periastron: float, bh_mass: float) -> float:
    """
    Convert Periastron distance P to the variable Q (easier to work with)
    """
    if periastron < 2. * bh_mass: return np.nan
    return np.sqrt((periastron - 2. * bh_mass) * (periastron + 6. * bh_mass))


def calc_b_from_periastron(periastron: float, bh_mass: float) -> float:
    """
    Get impact parameter b from Periastron distance P

    Args:
        periastron (float): Periastron distance
        bh_mass (float): Black hole mass

    Attention:
        The paper has a typo here. The fracture on the right hand side equals b², not b.
        You can verify this by filling in u_2 in equation 3, and you'll see. 
        Only this way do the limits P -> 3M and P >> M hold true,
        as well as the value for b_c

    Returns:
        float: Impact parameter :math:`b`
    """
    if periastron <= 2. * bh_mass: return np.nan
    return np.sqrt(periastron ** 3 / (periastron - 2. * bh_mass))


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
    if q is np.nan: return np.nan
    # WARNING: Paper has an error here. There should be brackets around the numerator.
    return np.sqrt((q - periastron + 6 * bh_mass) / (2 * q))


def k_squared(periastron: float, bh_mass: float):
    """Calculate the squared modulus of elliptic integral"""
    q = calc_q(periastron, bh_mass)
    if q is np.nan: return np.nan
    # WARNING: Paper has an error here. There should be brackets around the numerator.
    return (q - periastron + 6 * bh_mass) / (2 * q)


def zeta_inf(periastron: float, bh_mass: float) -> float:
    """
    Calculate Zeta_inf for elliptic integral F(Zeta_inf, k)
    """
    q = calc_q(periastron, bh_mass)
    if q is np.nan: return np.nan
    arg = (q - periastron + 2 * bh_mass) / (q - periastron + 6 * bh_mass)
    z_inf = np.arcsin(np.sqrt(arg))
    return z_inf


def zeta_r(periastron: float, r: float, bh_mass: float) -> float:
    """
    Calculate the elliptic integral argument Zeta_r for a given value of P and r
    """
    q = calc_q(periastron, bh_mass)
    if q is np.nan: return np.nan
    a = (q - periastron + 2 * bh_mass + (4 * bh_mass * periastron) / r) / (q - periastron + (6 * bh_mass))
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
    return np.cos(phi) * np.cos(incl) / np.sqrt((1 - np.sin(incl) ** 2 * np.cos(phi) ** 2))


def alpha(phi: float, incl: float):
    """Returns observer coordinate of photon given phi (BHF) and inclination (BHF)"""
    return np.arccos(cos_alpha(phi, incl))


def filter_periastrons(periastron: List, bh_mass: float, tol: float = 10e-3) -> List[float]:
    """
    Removes instances where P == 2*M
    returns indices where this was the case
    """
    return [e for e in periastron if abs(e - 2. * bh_mass) > tol]


def eq13(
    periastron: float, 
    ir_radius: float, 
    ir_angle: float, 
    bh_mass: float, 
    incl: float, 
    n: int = 0
) -> float:
    """
    Relation between radius (where photon was emitted in accretion disk), a and P.
    P can be converted to b, yielding the polar coordinates (b, a) on the photographic plate

    This function get called almost everytime when you need to calculate some black hole property
    """
    q = calc_q(periastron, bh_mass)
    if q is np.nan: return np.nan
    z_inf = zeta_inf(periastron, bh_mass)
    m = k_squared(periastron, bh_mass)  # mpmath takes m = k² as argument.
    ell_inf = ellipkinc(z_inf, m)  # Elliptic integral F(zeta_inf, k)
    g = np.arccos(cos_gamma(ir_angle, incl))

    # Calculate the argument of sn 
    # WARNING: paper has an error here: \sqrt(P / Q) should be in denominator, not numerator
    # There's no way that \gamma and \sqrt(P/Q) can end up on the same side of the division
    if n == 0:  # higher order image
        ellips_arg = g / (2. * np.sqrt(periastron / q)) + ell_inf
    elif n > 0:  # direct image
        ell_k = ellipk(m)  # calculate complete elliptic integral of mod m = k²
        ellips_arg = (g - 2. * n * np.pi) / (2. * np.sqrt(periastron / q)) - ell_inf + 2. * ell_k
    else: raise NotImplementedError("Only 0 and positive integers are allowed for the image order.")

    sn, _, _, _ = ellipj(ellips_arg, m)
    term1 = -(q - periastron + 2. * bh_mass) 
    term2 = (q - periastron + 6. * bh_mass) * sn * sn
    
    # solve this for zero
    return 4. * bh_mass * periastron - ir_radius * (term1 + term2)


def calc_periastron(
    radius: float, 
    incl: float, 
    alpha: float, 
    bh_mass: float, 
    midpoint_iterations: int=100, 
    plot_inbetween: bool=False,
    n: int=0, 
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

    def get_plot(X, Y, solution, radius=radius):
        fig = plt.figure()
        plt.title("Eq13(P)\nr={}, a={}".format(radius, round(alpha, 5)))
        plt.xlabel('P')
        plt.ylabel('Eq13(P)')
        plt.axhline(0, color='black')
        plt.plot(X, Y)
        plt.scatter(solution, 0, color='red')
        return plt

    photon_sphere_radius = 3*np.sqrt(3)*bh_mass
    x = np.linspace(photon_sphere_radius, 2. * radius, 2)
    y = np.array([eq13(P, radius, alpha, bh_mass, incl, n) for P in x])
    if y[0] > 0: return np.nan  # no periastron found that solves eq13: can happen for small values of R and alpha.
    
    kwargs_eq13 = {"ir_radius": radius, "ir_angle": alpha, "bh_mass": bh_mass, "incl": incl, "n": n}
    x, y = improve_solutions_midpoint(
        func=eq13, 
        kwargs=kwargs_eq13,
        x=x, 
        y=y, 
        iterations=midpoint_iterations)
    if plot_inbetween: get_plot(x, y, x).show()
    return np.mean(x)


def calc_impact_parameter(
    radius, 
    incl, 
    alpha, 
    bh_mass, 
    n=0, 
    midpoint_iterations=100, 
    plot_inbetween=False,
) -> float:
    """
    Given a value for r (BH frame) and alpha (BH/observer frame), calculate the corresponding periastron value
    This periastron is then converted to an impact parameter b, yielding the observer frame coordinates (b, alpha).
    Does this by generating range of periastron values, evaluating eq13 on this range and using a midpoint method
    to iteratively improve which periastron value solves equation 13.

    The considered initial periastron range must not be lower than min_periastron (i.e. the photon sphere),
    otherwise non-physical solutions will be found. These are interesting in their own right (the equation yields
    complex solutions within radii smaller than the photon sphere!), but are for now outside the scope of this project.
    Must be large enough to include solution, hence the dependency on the radius (the bigger the radius of the
    accretion disk where you want to find a solution, the bigger the periastron solution is, generally)

    Args:
        radius (float): radius on the accretion disk (BH frame)
        incl (float): inclination of the black hole
        alpha: angle along the accretion disk (BH frame and observer frame)
        bh_mass (float): mass of the black hole
        midpoint_iterations (int): amount of midpoint iterations to do when searching a periastron value solving eq13
        plot_inbetween (bool): plot
    """
    # alpha_obs is flipped alpha/bh if n is odd
    if n % 2 == 1: alpha = (alpha + np.pi) % (2*np.pi)
    
    periastron_solution = calc_periastron(
        radius, 
        incl, 
        alpha, 
        bh_mass, 
        midpoint_iterations, 
        plot_inbetween,
        n)

    if periastron_solution is np.nan:
        if n == 1: 
            return 6.6*bh_mass
        # There are no perigees for small r and alpha close to n*np.pi
        if np.pi/2 < alpha < 3*np.pi/2:  # no perigee for small R in lensed upper half. 
            # TODO: found by trial and error, but why is b ~ 6.6M for this limit?
            return 6.6*bh_mass
        else:
            # No perigee in lower half, close to middle.
            return ellipse(radius, alpha, incl)
    return calc_b_from_periastron(periastron_solution, bh_mass)


def phi_inf(periastron, M):
    q = calc_q(periastron, M)
    ksq = (q - periastron + 6. * M) / (2. * q)
    z_inf = zeta_inf(periastron, M)
    phi = 2. * (np.sqrt(periastron / q)) * (ellipk(ksq) - ellipkinc(z_inf, ksq))
    return phi


def mu(periastron, bh_mass):
    return float(2 * phi_inf(periastron, bh_mass) - np.pi)


def ellipse(r, a, incl) -> float:
    """Equation of an ellipse, reusing the definition of cos_gamma.
    This equation can be used for calculations in the Newtonian limit (large P = b, small a)
    or to visualize the equatorial plane."""
    a = (a + np.pi/2) % (2*np.pi)  # rotate 90 degrees for consistency with rest of the code
    major_axis = r
    minor_axis = abs(major_axis*np.cos(incl))
    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
    return minor_axis / np.sqrt((1 - (eccentricity * np.cos(a))**2))


def flux_intrinsic(r, acc, bh_mass):
    r_ = r / bh_mass
    log_arg = (
        (np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3))) / \
        ((np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
    f = (3. * bh_mass * acc / (8 * np.pi)) * (1 / ((r_ - 3) * r ** 2.5)) * \
        (np.sqrt(r_) - np.sqrt(6) + 3 ** -.5 * np.log10(log_arg))
    return f


def flux_observed(r, acc, bh_mass, redshift_factor):
    flux_intr = flux_intrinsic(r, acc, bh_mass)
    flux_observed = flux_intr / redshift_factor ** 4
    return flux_observed


def redshift_factor(radius, angle, incl, bh_mass, b):
    r"""
    Calculate the gravitational redshift factor (ignoring cosmological redshift): 
    
    .. math::

        1 + z = (1 - \Omega b \cos(\eta)) \left( -g_{tt} - 2 \Omega g_{t\phi} - \Omega^2 g_{\phi\phi} \right)^{-1/2}
    
    Attention:
        :cite:`Luminet_1979` doe snot have the correct equation for the redshift factor. 
        The correct formula is given above.
    """
    # gff = (radius * np.sin(incl) * np.sin(angle)) ** 2
    # gtt = - (1 - (2. * M) / radius)
    z_factor = (1. + np.sqrt(bh_mass / (radius ** 3)) * b * np.sin(incl) * np.sin(angle)) * \
               (1 - 3. * bh_mass / radius) ** -.5
    return z_factor


if __name__ == '__main__':
    M = 1
    solver_params = {
        'initial_guesses': 20,
        'midpoint_iterations': 10,
        'plot_inbetween': True,
        'min_periastron': 2*M + 1e-5
        }
    print(calc_periastron(3, 10, 0, 1, **solver_params))

    # writeFramesEq13(5, solver_params=solver_params)
