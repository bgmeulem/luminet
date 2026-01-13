from luminet.black_hole import BlackHole

def test_photons(n=1000):
    """
    Test if black hole can sample individual photons
    """

    bh = BlackHole(incl=1.3)
    photons, ghost_photons = bh.sample_photons(n)
    return photons, ghost_photons
