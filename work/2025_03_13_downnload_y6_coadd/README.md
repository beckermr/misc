# Downloading DES Y6 Coadd Data

To generate a conda env on your mac do this

```bash
CONDA_SUBDIR=osx-64 conda create -n des-query des-easyaccess ipython ipykernel oracle-instant-client fitsio ultraplot
```

Then activate the environment

```bash
conda activate des-query
```

Then run

```bash
easyaccess --db desoper -l query_images.sql
```

This will create a file named `coadd_img.fits` in the current directory. This file contains path information for the coadd images.

To get a coadd path from this file, do the following:

```python
In [1]: import fitsio

In [2]: data = fitsio.read("coadd_img.fits")

In [3]: data
Out[3]:
array([('DES1959-5040-i', 'DES1959-5040', 'OPS_Taiga/multiepoch/Y6A1/r4575/DES1959-5040/p01/coadd', 'DES1959-5040_r4575p01_i.fits', '.fz', 'i', 2730717),
       ('DES1959-5248-Y', 'DES1959-5248', 'OPS_Taiga/multiepoch/Y6A1/r4575/DES1959-5248/p01/coadd', 'DES1959-5248_r4575p01_Y.fits', '.fz', 'Y', 2730718),
       ('DES1959-5248-z', 'DES1959-5248', 'OPS_Taiga/multiepoch/Y6A1/r4575/DES1959-5248/p01/coadd', 'DES1959-5248_r4575p01_z.fits', '.fz', 'z', 2730718),
       ('DES2001-5748-g', 'DES2001-5748', 'OPS_Taiga/multiepoch/Y6A1/r4575/DES2001-5748/p01/coadd', 'DES2001-5748_r4575p01_g.fits', '.fz', 'g', 2730720),
       ('DES2014-5457-i', 'DES2014-5457', 'OPS_Taiga/multiepoch/Y6A1/r4575/DES2014-5457/p01/coadd', 'DES2014-5457_r4575p01_i.fits', '.fz', 'i', 2730732),
       ('DES2019-6122-g', 'DES2019-6122', 'OPS_Taiga/multiepoch/Y6A1/r4575/DES2019-6122/p01/coadd', 'DES2019-6122_r4575p01_g.fits', '.fz', 'g', 2730738),
       ('DES1957-5414-i', 'DES1957-5414', 'OPS_Taiga/multiepoch/Y6A1/r4575/DES1957-5414/p01/coadd', 'DES1957-5414_r4575p01_i.fits', '.fz', 'i', 2731717),
       ('DES1959-5457-z', 'DES1959-5457', 'OPS_Taiga/multiepoch/Y6A1/r4575/DES1959-5457/p01/coadd', 'DES1959-5457_r4575p01_z.fits', '.fz', 'z', 2731718),
       ('DES1959-5705-Y', 'DES1959-5705', 'OPS_Taiga/multiepoch/Y6A1/r4575/DES1959-5705/p01/coadd', 'DES1959-5705_r4575p01_Y.fits', '.fz', 'Y', 2731719)],
      dtype=[('KEY', '<U104'), ('TILENAME', '<U80'), ('PATH', '<U1000'), ('FILENAME', '<U400'), ('COMPRESSION', '<U20'), ('BAND', '<U20'), ('PFW_ATTEMPT_ID', '>i8')])

In [4]: import os

In [5]: os.path.join("https://desar2.cosmology.illinois.edu/DESFiles/desarchive/", data["PATH"][0], data["FILENAME"][0] + data["COMPRESSION"][0])
Out[5]: 'https://desar2.cosmology.illinois.edu/DESFiles/desarchive/OPS_Taiga/multiepoch/Y6A1/r4575/DES1959-5040/p01/coadd/DES1959-5040_r4575p01_i.fits.fz'
```

You can use wget/curl with your DESDM username and password to download the image.
