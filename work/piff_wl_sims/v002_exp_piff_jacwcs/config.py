import galsim
import numpy as np

jc = galsim.ShearWCS(
    0.263,
    galsim.Shear(g1=np.random.uniform(low=-0.1, high=0.1),
                 g2=np.random.uniform(low=-0.1, high=0.1))).jacobian()

jacobian_dict = {
    'dudx': jc.dudx,
    'dudy': jc.dudy,
    'dvdx': jc.dvdx,
    'dvdy': jc.dvdy
}

gauss_psf = False
n_sims = 100
