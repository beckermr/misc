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

```