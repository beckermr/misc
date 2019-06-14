# Piff WL Sims

This directory has simple simulations to test using DES Y3 Piff PSFs for
WL simulations. The simulations are a grid of postage stamps, each with a
single exponential rendered in the middle. They then use a randomly selected
position from a single DES exposure Piff model as the PSF.

## v000_exp_gauss

This simulation used a Gaussian PSF as a null test of the code.

```
s2n: 10
# of sims: 288500
m       : 0.000255 +/- 0.000006
c       : -0.000000 +/- 0.000000
```

## v001_exp_piff

This simulation is the same as `v000_exp_gauss` but with Piff PSFs.

```
s2n: 10
# of sims: 995184
m       : 0.000250 +/- 0.000004
c       : -0.000000 +/- 0.000000
```

## v002_exp_piff_interp_diff

This simulation is the same as `v001_exp_piff` but the real-space
interpolant differs between metacal and that used in galsim to render
the objects.

```
s2n: 10
# of sims: 995329
m       : 0.000275 +/- 0.000003
c       : -0.000000 +/- 0.000000
```

## v003_exp_psfex

Same as v001 but with psfex.

```
s2n: 10
# of sims: 262800
m       : 0.000297 +/- 0.000007
c       : 0.000000 +/- 0.000000
```

## v004_exp_psfex_small_psf_stamp

This one uses a smaller PSFEx stamp for the PSF model images fed into
metacal than the ones used to render the objects.

```
s2n: 10
# of sims: 169900
m       : 0.000549 +/- 0.000011
c       : 0.000001 +/- 0.000000
```

## v005_exp_psfex_extrasmall_psf_stamp

Just like v004 but with 17 pixel postage stamps for the PSF.

```
s2n: 10
# of sims: 1000000
m       : 0.010274 +/- 0.000020
c       : 0.000012 +/- 0.000000
```

## v006_exp_piff_extrasmall_psf_stamp

Just like v001 but with 17 pixel postage stamps for the PSF.

```
s2n: 10
# of sims: 984102
m       : 0.012678 +/- 0.000170
c       : -0.000047 +/- 0.000003
```

## v006_exp_piff_small21_psf_stamp

Just like v001 but with 21 pixel postage stamps for the PSF.

```

```
