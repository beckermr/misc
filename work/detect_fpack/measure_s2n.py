import sys
import fitsio
import subprocess
import galsim
import numpy as np
import tqdm

if len(sys.argv) > 1:
    n_images = int(sys.argv[1])
else:
    n_images = 10

rng = np.random.RandomState()

scale = 0.263
gal = galsim.Exponential(half_light_radius=0.5)
psf = galsim.Gaussian(fwhm=0.9)
obj = galsim.Convolve([gal, psf])

img = obj.drawImage(nx=23, ny=23, scale=scale).array

target_s2n = 5
total_noise = np.sqrt(np.sum(img*img) / np.sum(img)) / target_s2n
print(total_noise)

img_noise = total_noise * np.sqrt(n_images)
print(img_noise)

det_img = np.zeros_like(img)
for _ in tqdm.trange(n_images):
    _img = img + rng.normal(size=img.shape) * img_noise
    if True:
        fitsio.write('./test.fits', _img, clobber=True)
        subprocess.check_call(
            'rm -f test.fits.fz && fpack test.fits', shell=True)
        _img = fitsio.read('test.fits.fz')
        subprocess.check_call('rm -f test.fits test.fits.fz', shell=True)

    det_img += _img

det_img /= n_images

s2n = np.sqrt(np.sum(det_img*img)) / total_noise

print('s2n:', s2n)
