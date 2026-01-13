import luminet.black_hole_math as bhmath
import numpy as np

class Photon:
    def __init__(self, radius, alpha, impact_parameter, z_factor=0.0, flux_o=0.0):
        self.radius = radius
        self.alpha = alpha
        self.impact_parameter = impact_parameter
        self.z_factor = z_factor
        """Redshift factor of the photon. This is usually calculated vectorized by the :class:`~luminet.black_hole.BlackHole` class"""
        self.flux_o = flux_o
        """Observed flux of the photon. This is usually calculated vectorized by the :class:`~luminet.black_hole.BlackHole` class"""


def sample_photon(min_r, max_r, incl, bh_mass, n) -> Photon:
    """Sample a random photon from the accretion disk."""
    alpha = np.random.random() * 2 * np.pi

    # Biased sampling towards circle center
    r = np.float64(min_r + (max_r - min_r) * np.random.random())
    # Evenly sampling
    # r = np.float64(min_r + (max_r - min_r) * np.random.random()**2)
    b = bhmath.solve_for_impact_parameter(r, incl, alpha, bh_mass, n)
    if b is np.nan:
        raise ValueError(f"b is nan for r={r}, alpha={alpha}, incl={incl}, M={bh_mass}, n={n}")

    return Photon(radius=r, alpha=alpha, impact_parameter=b)
