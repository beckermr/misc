# Simple DS Y3 End2End Sims

## Measuring Shear

### ngmixer + ngmix v1

```bash
python ../measure_shear_nocancel.py outputs-*/ngmixer/*/output/*
```

### ngmix v1.3

```bash
python ../measure_shear_nocancel.py outputs-*/metacal/*/*
```

## Results

### v001
 - mag 18 exp galaxies on grid
 - true detection
 - round Gaussian PSF
 - rendered w/ 'auto'


```
nobj: 3357713
njack: 100
m [1e-3]: -0.889469 +/- 0.133388
c [1e-4]: -0.002968 +/- 0.015885
```

### v003
 - mag 18 exp galaxies on grid
 - true detection
 - PIFF PSFs
 - rendered w/ 'auto'

```
nobj: 3269487
njack: 100
m [1e-3]: -1.688921 +/- 0.140998
c [1e-4]: -0.015971 +/- 0.017097
```

### v004
 - mag 18 exp galaxies on grid
 - true detection
 - round Gaussian PSF represented as a pixelized SB profile w/ scale 0.125 arcsec
 - rendered w/ 'auto'

```
nobj: 3357668
njack: 100
m [1e-3]: -1.036445 +/- 0.125293
c [1e-4]: 0.025032 +/- 0.017771
```

### v005
 - mag 18 exp galaxies on grid
 - true detection
 - non-round, spatially varying Gaussian PSF represented as a
   pixelized SB profile w/ scale 0.125 arcsec
 - rendered w/ 'auto'

```
nobj: 3357664
njack: 100
m [1e-3]: -0.951751 +/- 0.130623
c [1e-4]: 0.002035 +/- 0.017376
```

### v006
 - mag 18 exp galaxies on grid
 - true detection
 - non-round, spatially varying Gaussian PSF represented as a
   pixelized SB profile w/ scale 0.125 arcsec
 - noise added to PSF image to bring it to s/n~100
 - rendered w/ 'auto'

```
nobj: 3359471
njack: 100
m [1e-3]: -2.324636 +/- 0.143676
c [1e-4]: 0.020524 +/- 0.017190
```
