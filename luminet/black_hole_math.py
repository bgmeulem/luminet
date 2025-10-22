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
        as well as the value for :math:`b_c`.  The resulting images of the paper are correct though.


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
        Mind the typo in :cite:t:`Luminet_1979`. The numerator should be in brackets. The resulting images of the paper are correct though.

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
    r"""Calculate :math:`\zeta_r`
     
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


def calc_Z1(bh_mass, a):
    r"""Calculate :math:`Z1` for Kerr black holes.

    The variable :math:`Z1` is used to calculate the innermost orbit for Kerr black holes.

    .. math::
    
       Z_1 \equiv 1 + \sqrt[3]{1-a_*^2}\left[ \sqrt[3]{1+a_*} + \sqrt[3]{1-a_*} \right]

    Args:
        bh_mass (float): Mass of the black hole.
        a (float): Specific angular momentum of the black hole. Should always be between :math:`-1` and :math:`1`. :math:`a > 0` if the accretion disk orbits in the same direction as the hole rotates; :math:`a < 0` if it orbits in the opposite direction.

    See also:
     :cite:t:`Page_1974` Equation 15l

    See also:
        :meth:`calc_innermost_orbit` for the calculation of the innermost orbit of Kerr black holes.
    """
    a_ = a/bh_mass
    return 1 + (1-a_**2)**(1/3)*((1+a_)**(1/3) + (1 - a_)**(1/3))


def calc_Z2(bh_mass, a):
    r"""Calculate :math:`Z2` for Kerr black holes.

    The variable :math:`Z2` is used to calculate the innermost orbit for Kerr black holes.

    .. math::

       Z_2 \equiv \sqrt{3a_*^2+Z_1^2}

    Args:
        bh_mass (float): Mass of the black hole.
        a (float): Specific angular momentum of the black hole. Should always be between :math:`-1` and :math:`1`. :math:`a > 0` if the accretion disk orbits in the same direction as the hole rotates; :math:`a < 0` if it orbits in the opposite direction.

    See also:
        :cite:t:`Page_1974` Equation 15m

    See also:
        :meth:`calc_innermost_orbit` for the calculation of the innermost orbit of Kerr black holes.
    """
    Z1 = calc_Z1(bh_mass, a)
    a_ = a/bh_mass
    return np.sqrt(3*a_**2 + Z1**2)


def calc_innermost_stable_orbit(bh_mass, a):
    r"""Calculcate the innermost stable orbit :math:`r_{ms}` for a Kerr black hole.

    A larger specific angular momentum :math:`a` will yield innermost orbits closer to the black hole.

    .. math::

       \begin{align*}
       r_{ms} &= Mx_0^2 \\
       x_0^2 &= 3 + Z_2 - sgn(a_*)\sqrt{(3-Z_1)(3+Z_1+2Z_2) }
       \end{align*}

    Args:
        bh_mass (float): Mass of the black hole.
        a (float): Specific angular momentum of the black hole. Should always be between :math:`-1` and :math:`1`. :math:`a > 0` if the accretion disk orbits in the same direction as the hole rotates; :math:`a < 0` if it orbits in the opposite direction.

    See also:
        :cite:t:`Page_1974`

    See also:
        :meth:`calc_Z1` and :meth:`calc_Z2`.
    """
    Z1 = calc_Z1(bh_mass, a)
    Z2 = calc_Z2(bh_mass, a)
    a_ = a/bh_mass
    return bh_mass*(
        3 + Z2 - np.sign(a_)*((3-Z1)*(3+Z1+2*Z2))**.5
    )


def calc_x0(bh_mass, a):
    r"""Calculcate :math:`x_0` for Kerr black holes.

    .. math::

       x_0 = \sqrt{\frac{r_{ms}}{M}}

    Args:
        bh_mass (float): Mass of the black hole.
        a (float): Specific angular momentum of the black hole. Should always be between :math:`-1` and :math:`1`. :math:`a > 0` if the accretion disk orbits in the same direction as the hole rotates; :math:`a < 0` if it orbits in the opposite direction.
    
    See also:
        :cite:t:`Page_1974` Equation 15k

    See also:
        meth:`calc_innermost_orbit` for the calculation of :math:`r_{ms}`
    """
    rms = calc_innermost_stable_orbit(bh_mass, a)
    return np.sqrt(rms/bh_mass)


def calc_f_kerr(bh_mass, a, r):
    r"""Calculate the :math:`f`-function from :cite:t:`Page_1974` (Equation 12)

    The :math:`f`-function is used when calculating the relationship between intrinsic flux and radius for an accretion disk:

    .. math::

       F_s(r) = \frac{\dot{M}_0}{4\pi}e^{-(\nu + \psi + \mu)}f

    Here, :math:`\nu`, :math:`\psi` and :math:`\mu` are metric coefficients (functions of :math:`r`) of the Kerr metric. :math:`\dot{M}_0` is the radius-independent, time-averaged rate at which mass flows inward. Defining the innermost stable orbit as :math:`r_{ms}`, :math:`x=\sqrt{r/M}=\sqrt{r^*}`, :math:`x_0=\sqrt{r_{ms}/M}` and :math:`a^*=a/M`, the :math:`f`-function is defined as:

    .. math::

       \begin{align*}
        f = &\frac{3}{2M}\frac{1}{x^2(x^3 - 3x + 2a^*)}\Bigg[ x - x_0 - \frac{3}{2}a^*\ln\left(\frac{x}{x_0}\right) \\
         &- \frac{3(x_1 - a^*)^2}{x_1(x_1-x_2)(x_1-x_3)}\ln\left(\frac{x-x_1}{x_0-x_1}\right) \\
         &- \frac{3(x_2 - a^*)^2}{x_2(x_2-x_1)(x_2-x_3)}\ln\left(\frac{x-x_2}{x_0-x_2}\right) \\
         &- \frac{3(x_3 - a^*)^2}{x_3(x_3-x_1)(x_3-x_2)}\ln\left(\frac{x-x_3}{x_0-x_3}\right) \Bigg]
        \end{align*}

    , where

    .. math::

       \begin{align*}
       x_1 &= 2\cos(\frac{1}{3}\cos^{-1}(a_*) - \frac{\pi}{3}) \\
       x_2 &= 2\cos(\frac{1}{3}\cos^{-1}(a_*) + \frac{\pi}{3}) \\
       x_3 &= -2\cos(\frac{1}{3}\cos^{-1}(a_*)) \\
       \end{align*}

    For a Swarzschild black hole, :math:`a=0` and these simplify to:

    .. math::

       \begin{align*}
       x_1 &= \sqrt{3} \\
       x_2 &= 0 \\
       x_3 &= - \sqrt{3} \\
       f &= \frac{3}{2M}\frac{1}{{r^{*}}^{1.5}(r^*-3)}\left[x - x_0 + \frac{\sqrt{3}}{2}\ln\left( \frac{(\sqrt{6} - \sqrt{3})(\sqrt{r^*}+\sqrt{3})}{(\sqrt{6} + \sqrt{3})(\sqrt{r^*} - \sqrt{3})}  \right)  \right]
       \end{align*}

    Args:
        bh_mass (float): Mass of the black hole.
        a (float): Specific angular momentum of the black hole. Should always be between :math:`-1` and :math:`1`. :math:`a > 0` if the accretion disk orbits in the same direction as the hole rotates; :math:`a < 0` if it orbits in the opposite direction.
        r (float): Radius of the orbit

    Attention:
        :cite:t:`Luminet_1979` has a mistake in Equation 15. The factor in fromt of the :math:`log` should be :math:`\sqrt{3}/2` instead of :math:`\sqrt{3}/3`. This can be verified by solving :cite:t:`Page_1974` Equation 15n.  The resulting images of the paper are correct though.

    See also:
        :cite:t:`Page_1974` for more information.

    See also:
        :meth:`calc_flux_intrinsic_kerr` for the calculation of the intrinsic flux.

    See also:
       :meth:`calc_innermost_stable_orbit` for the calculation of :math:`r_{ms}`
    """
    a_ = a/bh_mass
    x = np.sqrt(r/bh_mass)
    x0 = calc_x0(bh_mass, a)
    x1 =  2*np.cos(np.arccos(a_)/3 - np.pi/3)
    x2 =  2*np.cos(np.arccos(a_)/3 + np.pi/3)
    x3 = -2*np.cos(np.arccos(a_)/3)
    A = 3 * (2*bh_mass)**-1 * (x**2 * (x**3 - 3*x + 2*a_))**-1
    f = A * (
        x - x0 - 1.5*a_*np.log(x/x0)
        - 3*(x1-a_)**2 * np.log((x-x1)/(x0-x1)) / (x1*(x1-x2)*(x1-x3))
        - 3*(x2-a_)**2 * np.log((x-x2)/(x0-x2)) / (x2*(x2-x1)*(x2-x3))
        - 3*(x3-a_)**2 * np.log((x-x3)/(x0-x3)) / (x3*(x3-x1)*(x3-x2))
    )
    return f


def calc_flux_intrinsic_kerr(bh_mass, a, r, acc):
    r"""Calculate the intrinsic flux of the accretion disk of a Kerr black hole, in function of the accretion rate, specific angular momentum, and radius of emission.
    
    The intrinsic flux is not redshift-corrected. Observed photons will have a flux that deviates from this by a factor of :math:`1/(1+z)^4`

    The intrinsic flux in function of the radius is defined as:

    .. math::

       F_s(r) &= \frac{\dot{M_0}}{4\pi}e^{-(\nu+\psi+\mu)}f \\

    where

    .. math::

       \begin{align*}
       e^{\nu+\psi+\mu} &= r \\
       f &= -\Omega_{,r}(E^{\dagger}-\Omega L^\dagger)^{-2}\int_{r_{ms}}^r(E^\dagger \ - \Omega L^\dagger)L^\dagger_{,r}dr
       \end{align*}

    Args:
        bh_mass (float): Mass of the black hole.
        a (float): Specific angular momentum of the black hole. Should always be between :math:`-1` and :math:`1`. :math:`a > 0` if the accretion disk orbits in the same direction as the hole rotates; :math:`a < 0` if it orbits in the opposite direction.
        r (float): Radius of the orbit.
        acc (float): (initial) accretion rate of the black hole :math:`\dot{M}_0`

    See also:
        :meth:`calc_f_kerr` for an algebraic expression of the :math:`f` function.

    """
    f = calc_f_kerr(bh_mass=bh_mass, a=a, r=r)
    exp_nupsimu = r
    return acc * f / exp_nupsimu / 4 / np.pi


def calc_flux_intrinsic_swarzschild(bh_mass, r, acc):
    r"""Calculate the intrinsic flux of a photon.
    
    The intrinsic flux is not redshift-corrected. Observed photons will have a flux that deviates from this by a factor of :math:`1/(1+z)^4`

    .. math::

       F_s = \frac{3 M \dot{M}}{8 \pi (r^* - 3) {r^*}^{5/2}} \left( \sqrt{r^*} - \sqrt{6} + \frac{\sqrt{3}}{2} \log \left( \frac{(\sqrt{r^*} + \sqrt{3})(\sqrt{6}-\sqrt{3})}{(\sqrt{6} + \sqrt{3})(\sqrt{r^*}-\sqrt{3})} \right) \right)

    where :math:`r^*=r/M`

    Args:
        r (float): radius on the accretion disk (BH frame)
        acc (float): accretion rate
        bh_mass (float): mass of the black hole

    Returns:
        float: Intrinsic flux of the photon :math:`F_s`

    Attention:
        :cite:t:`Luminet_1979` has a mistake in Equation 15. The factor in fromt of the :math:`log` should be :math:`\sqrt{3}/2` instead of :math:`\sqrt{3}/3`. This can be verified by solving :cite:t:`Page_1974` Equation 15n. The resulting images of the paper are correct though.
    """
    r_ = r / bh_mass
    log_arg = (np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3)) / ((np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
    A = 3 * bh_mass * acc / (8 * np.pi)/ ((r_ - 3) * r_**2.5)
    f = A * (np.sqrt(r_) - np.sqrt(6) + (np.sqrt(3)/2) * np.log(log_arg))
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
    flux_intr = calc_flux_intrinsic_swarzschild(r=r, acc=acc, bh_mass=bh_mass)
    flux_observed = flux_intr / redshift_factor**4
    return flux_observed


def calc_redshift_factor(radius, angle, incl, bh_mass, b):
    r"""
    Calculate the gravitational redshift factor (ignoring cosmological redshift):

    .. math::

        1 + z = (1 - \Omega b \cos(\eta)) \left( -g_{tt} - 2 \Omega g_{t\phi} - \Omega^2 g_{\phi\phi} \right)^{-1/2}

    Attention:
        :cite:t:`Luminet_1979` does not have the correct equation for the redshift factor.
        The correct formula is given above. The resulting images of the paper are correct though.
    """
    # gff = (radius * np.sin(incl) * np.sin(angle)) ** 2
    # gtt = - (1 - (2. * M) / radius)
    z_factor = (
        1.0 + np.sqrt(bh_mass / (radius**3)) * b * np.sin(incl) * np.sin(angle)
    ) * (1 - 3.0 * bh_mass / radius) ** -0.5
    return z_factor
