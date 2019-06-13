#!/bin/bash

export OMP_NUM_THREADS=1

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    # the condor system creates a scratch directory for us,
    # and cleans up afterward
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir
else
    # otherwise use the TMPDIR
    tmpdir='.'
    mkdir -p $tmpdir
fi

# conda activate bnl
for seed in `seq $1 $2`
do
    rm -f $tmpdir/gals.fits
    rm -f $tmpdir/psfs.fits
    galsim config.yaml image.random_seed=${seed} output.file_name=$tmpdir/gals.fits
    galsim config-psf.yaml image.random_seed=${seed} output.file_name=$tmpdir/psfs.fits
    python run.py $tmpdir/gals.fits $tmpdir/psfs.fits ${seed} $tmpdir/out_${1}_${2}_${seed}.pkl
done

python collate.py ${3}/out_${1}_${2}.pkl $tmpdir/out_${1}_${2}_*

rm -f $tmpdir/gals.fits
rm -f $tmpdir/psfs.fits
for seed in `seq $1 $2`
do
    rm -f $tmpdir/out_${1}_${2}_${seed}.pkl
done
