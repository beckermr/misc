import galsim

jc = galsim.ShearWCS(0.263, galsim.Shear(g1=0.01, g2=0.01)).jacobian()

jacobian_dict = {
    'dudx': jc.dudx,
    'dudy': jc.dudy,
    'dvdx': jc.dvdx,
    'dvdy': jc.dvdy
}

gauss_psf = True
n_sims = 10