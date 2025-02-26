"""Math routines for :cite:t:`Luminet_1979`.

This module contains the mathematical routines to calculate the trajectory of photons around 
a Swarzschild black hole, as described in :cite:t:`Luminet_1979`.
"""
import numpy as np
from scipy.special import ellipj, ellipk, ellipkinc

from luminet.solver import improve_solutions

def calc_q(p: float, bh_mass: float) -> float:
    r"""Convert perigee :math:`P` to :math:`Q`
     
    The variable :math:`Q` has no explicit physical meaning, but makes
    many equations more readable.

    .. math::

       Q = \sqrt{(P - 2M)(P + 6M)}

    Args:
        periastron (float): Periastron distance
        bh_mass (float): Black hole mass

    Returns:
        float: :math:`Q`
    """
    if p < 2.0 * bh_mass:
        return np.nan
    return np.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))


def calc_b_from_perigee(p: float, bh_mass: float) -> float:
    r"""Get impact parameter :math:`b` from the photon perigee :math:`P`


    .. math::

       b = \sqrt{\frac{P^3}{P - 2M}}

    Args:
        p (float): Perigee distance
        bh_mass (float): Black hole mass

    Attention:
        :cite:t:`Luminet_1979` has a typo here. 
        The fraction on the right hand side equals :math:`b^2`, not :math:`b`.
        You can verify this by filling in :math:`u_2` in Equation 3.
        Only this way do the limits :math:`P -> 3M` and :math:`P >> M` hold true,
        as well as the value for :math:`b_c`

    Returns:
        float: Impact parameter :math:`b`
    """
    if p <= 2.0 * bh_mass:
        return np.nan
    return np.sqrt(p**3 / (p - 2.0 * bh_mass))


def calc_k(periastron: float, bh_mass: float) -> float:
    r"""Calculate the modulus of the elliptic integral

    The modulus is defined as:
     
    .. math::
    
       k = \sqrt{\frac{Q - P + 6M}{2Q}}

    Args:
        periastron (float): Periastron distance
        bh_mass (float): Black hole mass

    Returns:
        float: Modulus of the elliptic integral

    Attention:
        Mind the typo in :cite:t:`Luminet_1979`. The numerator should be in brackets.
    """
    q = calc_q(periastron, bh_mass)
    if q is np.nan:
        return np.nan
    # WARNING: Paper has an error here. There should be brackets around the numerator.
    return np.sqrt((q - periastron + 6 * bh_mass) / (2 * q))


def calc_k_squared(p: float, bh_mass: float):
    r"""Calculate the squared modulus of elliptic integral

    .. math::

       k^2 = m = \frac{Q - P + 6M}{2Q}     
    
    Attention:
        :cite:t:`Luminet_1979` uses the non-squared modulus in the elliptic integrals.
        This is just a convention. However, ``scipy`` asks for the squared modulus :math:`m=k^2`, not the modulus.

    Args:
        p (float): Perigee distance
        bh_mass (float): Black hole mass
    """
    q = calc_q(p, bh_mass)
    if q is np.nan:
        return np.nan
    # WARNING: Paper has an error here. There should be brackets around the numerator.
    return (q - p + 6 * bh_mass) / (2 * q)


def calc_zeta_inf(p: float, bh_mass: float) -> float:
    r"""Calculate :math:`\zeta_\infty` 
    
    This is used in the Jacobi incomplete elliptic integral :math:`F(\zeta_\infty, k)`

    .. math::

       \zeta_\infty = \arcsin \left( \sqrt{\frac{Q - P + 2M}{Q - P + 6M}} \right)

    Args:
        p (float): Perigee distance
        bh_mass (float): Black hole mass

    Returns:
        float: :math:`\zeta_\infty`
    """
    q = calc_q(p, bh_mass)
    if q is np.nan:
        return np.nan
    arg = (q - p + 2 * bh_mass) / (q - p + 6 * bh_mass)
    z_inf = np.arcsin(np.sqrt(arg))
    return z_inf


def calc_zeta_r(p: float, r: float, bh_mass: float) -> float:
    r"""Calculate :math:`zeta_r`
     
    This is used for the Jacobi incomplete elliptic integral for higher-order images.

    .. math::

       \zeta_r = \arcsin \left( \sqrt{\frac{Q - P + 2M + \frac{4MP}{r}}{Q - P + 6M}} \right)

    Args:
        p (float): Perigee distance
        r (float): Radius in the black hole frame.
        bh_mass (float): Black hole mass

    Returns:
        float: :math:`\zeta_r`
    """
    q = calc_q(p, bh_mass)
    if q is np.nan:
        return np.nan
    a = (q - p + 2 * bh_mass + (4 * bh_mass * p) / r) / (
        q - p + (6 * bh_mass)
    )
    s = np.arcsin(np.sqrt(a))
    return s


def calc_cos_gamma(alpha: float, incl: float) -> float:
    r"""Calculate :math:`\cos(\gamma)`

    This is used in the argument of the Jacobi elliptic integrals.

    .. math::

       \cos(\gamma) = \frac{\cos(\alpha)}{\sqrt{\cos(\alpha)^2 + \frac{1}{\tan(\theta_0)^2}}}

    Args:
        alpha (float): Angle in the black hole frame
        incl (float): Inclination of the observer :math:`\theta_0`

    Returns:
        float: :math:`\cos(\gamma)`
    """
    return np.cos(alpha) / np.sqrt(np.cos(alpha) ** 2 + 1 / (np.tan(incl) ** 2))


def calc_sn(
    p: float,
    angle: float,
    bh_mass: float,
    incl: float,
    order: int = 0,
) -> float:
    r"""Calculate the elliptic function :math:`\text{sn}`

    For direct images, this is:

    .. math::

        \text{sn} \left( \frac{\gamma}{2 \sqrt{P/Q}} + F(\zeta_{\infty}, k) \right)

    For higher order images, this is:

    .. math::

        \text{sn} \left( \frac{\gamma - 2n\pi}{2 \sqrt{P/Q}} - F(\zeta_{\infty}, k) + 2K(k) \right)

    Here, :math:`F` is the incomplete elliptic integral of the first kind, 
    and :math:`K` is the complete elliptic integral of the first kind.
    Elliptic integrals and elliptic functions are related:

    .. math::

       u &= F(\phi,m) \\
       \text{sn}(u|m) &= sin(\phi)


    Attention:
        Note that ``scipy`` uses the modulus :math:`m = k^2` in the elliptic integrals, 
        not the modulus :math:`k`.

    Args:
        p (float): Perigee distance
        angle (float): Angle in the black hole frame :math:`\alpha`
        bh_mass (float): Black hole mass
        incl (float): Inclination of the observer :math:`\theta_0`
        order (int): Order of the image. Default is :math:`0` (direct image).

    Returns:
        float: Value of the elliptic integral :math:`\text{sn}`
    """
    q = calc_q(p, bh_mass)
    if q is np.nan:
        return np.nan
    z_inf = calc_zeta_inf(p, bh_mass)
    m = calc_k_squared(p, bh_mass)  # mpmath takes m = k² as argument.
    ell_inf = ellipkinc(z_inf, m)  # Elliptic integral F(zeta_inf, k)
    g = np.arccos(calc_cos_gamma(angle, incl))

    if order == 0:  # higher order image
        ellips_arg = g / (2.0 * np.sqrt(p / q)) + ell_inf
    elif order > 0:  # direct image
        ell_k = ellipk(m)  # calculate complete elliptic integral of mod m = k²
        ellips_arg = (
            (g - 2.0 * order * np.pi) / (2.0 * np.sqrt(p / q))
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
    p: float,
    ir_angle: float,
    bh_mass: float,
    incl: float,
    order: int = 0,
) -> float:
    """Calculate the radius on the black hole accretion disk from a photon's perigee value.

    Args:
        p (float): Periastron distance. This is directly related to the observer coordinate frame :math:`b`
        ir_angle (float): Angle of the observer/bh coordinate frame.
        bh_mass (float): Black hole mass
        incl (float): Inclination of the black hole
        order (int): Order of the image. Default is :math:`0` (direct image).

    Attention:
        This is not the equation used to solve for the perigee value :math:`P`.
        For the equation that is optimized in order to convert between black hole and observer frame,
        see :py:meth:`perigee_optimization_function`.

    Returns:
        float: Black hole frame radius :math:`r` of the photon trajectory.
    """
    sn = calc_sn(p, ir_angle, bh_mass, incl, order)
    q = calc_q(p, bh_mass)

    term1 = -(q - p + 2.0 * bh_mass)
    term2 = (q - p + 6.0 * bh_mass) * sn * sn

    return 4.0 * bh_mass * p / (term1 + term2)


def perigee_optimization_function(
    p: float,
    ir_radius: float,
    ir_angle: float,
    bh_mass: float,
    incl: float,
    order: int = 0,
) -> float:
    r"""Cost function for the optimization of the periastron value.

    This function is optimized to find the periastron value that solves Equation 13 in cite:t:`Luminet1979`:

    .. math::

        4 M P - r (Q - P + 2 M) + r (Q - P + 6 M) \text{sn}^2 \left( \frac{\gamma}{2 \sqrt{P/Q}} + F(\zeta_{\infty}, k) \right) = 0

    When the above equation is zero, the photon perigee value :math:`P` is correct.

    See also:
        :py:meth:`solve_for_perigee` to calculate the perigee of a photon orbit, given an accretion disk radius of origin :math:`R`.

    Args:
        periastron (float): Periastron distance
        ir_radius (float): Radius in the black hole frame
        ir_angle (float): Angle in the black hole frame
        bh_mass (float): Black hole mass
        incl (float): Inclination of the black hole
        order (int): Order of the image. Default is :math:`0` (direct image).

    Returns:
        float: Cost function value. Should be zero when the photon perigee value is correct.
    """
    q = calc_q(p, bh_mass)
    if q is np.nan:
        return np.nan
    sn = calc_sn(p, ir_angle, bh_mass, incl, order)
    term1 = -(q - p + 2.0 * bh_mass)
    term2 = (q - p + 6.0 * bh_mass) * sn * sn
    zero_opt = 4.0 * bh_mass * p - ir_radius * (term1 + term2)
    return zero_opt


def solve_for_perigee(
    radius: float,
    incl: float,
    alpha: float,
    bh_mass: float,
    order: int = 0,
) -> float:
    r"""Calculate the perigee of a photon trajectory, when the black hole coordinates are known.

    This photon perigee can be converted to an impact parameter :math:`b`, yielding the observer frame coordinates :math:`(b, \alpha)`.

    See also:
        :py:meth:`perigee_optimization_function` for the optimization function used.
    
    See also:
        :py:meth:`solve_for_impact_parameter` to also convert periastron distance to impact parameter :math:`b` (observer frame).

    Args:
        radius (float): radius on the accretion disk (BH frame)
        incl (float): inclination of the black hole
        alpha: angle along the accretion disk (BH frame and observer frame)
        bh_mass (float): mass of the black hole
        order (int): Order of the image. Default is :math:`0` (direct image).

    Returns:
        float: Perigee distance :math:`P` of the photon
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
            perigee_optimization_function(periastron_guess, radius, alpha, bh_mass, incl, order)
            for periastron_guess in periastron_initial_guess
        ]
    )
    assert not any(np.isnan(y)), "Initial guess contains nan values"

    # If the solution is not in the initial range it likely doesnt exist for these input parameters
    # can happen for high inclinations and small radii -> photon orbits have P<3M, but the photon
    # does not travel this part of the orbit.
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
        func=perigee_optimization_function,
        x=periastron_initial_guess,
        y=y,
        kwargs=kwargs_eq13,
    )
    return periastron


def solve_for_impact_parameter(
    radius,
    incl,
    alpha,
    bh_mass,
    order=0,
) -> float:
    r"""Calculate observer coordinates of a BH frame photon.

    This method solves Equation 13 to get the photon perigee distance for a given coordinate on the black hole accretion
    disk :math:`(r, \alpha)`. 
    The observer coordinates :math:`(b, \alpha)` are then calculated from the perigee distance. 

    Attention:
        Photons that originate from close to the black hole, and the front of the accretion disk, have orbits whose
        perigee is below :math:`3M` (and thus would be absorbed by the black hole), but still make it to the camera in the observer frame.
        These photons are not absorbed by the black hole, since they simply never actually travel the part of their orbit that lies below :math:`3M`

    Args:
        radius (float): radius on the accretion disk (BH frame)
        incl (float): inclination of the black hole
        alpha: angle along the accretion disk (BH frame and observer frame)
        bh_mass (float): mass of the black hole
        order (int): Order of the image. Default is :math:`0` (direct image).

    Returns:
        float: Impact parameter :math:`b` of the photon
    """
    # alpha_obs is flipped alpha/bh if n is odd
    if order % 2 == 1:
        alpha = (alpha + np.pi) % (2 * np.pi)

    periastron_solution = solve_for_perigee(radius, incl, alpha, bh_mass, order)

    # Photons that have no perigee and are not due to the exception described above are simply absorbed
    if periastron_solution is np.nan:
        if order == 0 and ((alpha < np.pi / 2) or (alpha > 3 * np.pi / 2)):
            # Photons with small R in the lower half of the image originate from photon orbits that
            # have a perigee < 3M. However, these photons are not absorbed by the black hole and do in fact reach the camera,
            # since they never actually travel this forbidden part of their orbit.
            # --> Return the newtonian limit i.e. just an ellipse, like the rings of saturn that are visible in front of saturn.
            return ellipse(radius, alpha, incl)
        else:
            return np.nan
    b = calc_b_from_perigee(periastron_solution, bh_mass)
    return b


def ellipse(r, a, incl) -> float:
    r"""Equation of an ellipse
    
    This equation can be used for calculations in the Newtonian limit (large :math:`P \approx b`)
    It is also used to interpolate photons that originate from close to the black hole, and the front of the accretion disk.
    In this case, their perigee theoretically lies below :math:`3M`, but they are not absorbed by the black hole, as
    they travel away from the black hole, and never actually reach the part of their orbit that lies below :math:`3M`.

    Args:
        r (float): radius on the accretion disk (BH frame)
        a (float): angle along the accretion disk (BH frame and observer frame)
        incl (float): inclination of the black hole

    Returns:
        float: Impact parameter :math:`b` of the photon trajectory in the observer frame, which is in this case identical to the radius in the black hole frame :math:`r`
    
    """
    a = (a + np.pi / 2) % (
        2 * np.pi
    )  # rotate 90 degrees for consistency with rest of the code
    major_axis = r
    minor_axis = abs(major_axis * np.cos(incl))
    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
    return minor_axis / np.sqrt((1 - (eccentricity * np.cos(a)) ** 2))


def calc_flux_intrinsic(r, acc, bh_mass):
    r"""Calculate the intrinsic flux of a photon.
    
    The intrinsic flux is not redshift-corrected. Observed photons will have a flux
    that deviates from this.

    .. math::

        F_s = \frac{3 M \dot{M}}{8 \pi (r - 3) r^{2.5}} \left( \sqrt{r} - \sqrt{6} + \frac{1}{\sqrt{3}} \log \left( \frac{\sqrt{r} + \sqrt{3}}{\sqrt{6} + \sqrt{3}} \right) \right)

    Args:
        r (float): radius on the accretion disk (BH frame)
        acc (float): accretion rate
        bh_mass (float): mass of the black hole

    Returns:
        float: Intrinsic flux of the photon :math:`F_s`
    """
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


def calc_flux_observed(r, acc, bh_mass, redshift_factor):
    r"""Calculate the observed bolometric flux of a photon :math:`F_o`

    .. math::

        F_o = \frac{F_s}{(1 + z)^4}
    
    Args:
        r (float): radius on the accretion disk (BH frame)
        acc (float): accretion rate
        bh_mass (float): mass of the black hole
        redshift_factor (float): gravitational redshift factor

    Returns:
        float: Observed flux of the photon :math:`F_o`
    """
    flux_intr = calc_flux_intrinsic(r, acc, bh_mass)
    flux_observed = flux_intr / redshift_factor**4
    return flux_observed


def calc_redshift_factor(radius, angle, incl, bh_mass, b):
    r"""
    Calculate the gravitational redshift factor (ignoring cosmological redshift):

    .. math::

        1 + z = (1 - \Omega b \cos(\eta)) \left( -g_{tt} - 2 \Omega g_{t\phi} - \Omega^2 g_{\phi\phi} \right)^{-1/2}

    Attention:
        :cite:t:`Luminet_1979` does not have the correct equation for the redshift factor.
        The correct formula is given above.
    """
    # gff = (radius * np.sin(incl) * np.sin(angle)) ** 2
    # gtt = - (1 - (2. * M) / radius)
    z_factor = (
        1.0 + np.sqrt(bh_mass / (radius**3)) * b * np.sin(incl) * np.sin(angle)
    ) * (1 - 3.0 * bh_mass / radius) ** -0.5
    return z_factor
