import galsim

jc = galsim.ShearWCS(0.263, galsim.Shear(g1=0.1, g2=0.1)).jacobian()

jacobian_dict = {
    'dudx': jc.dudx,
    'dudy': jc.dudy,
    'dvdx': jc.dvdx,
    'dvdy': jc.dvdy
}

gauss_psf = True
n_sims = 100
