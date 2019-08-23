#!/bin/bash

export OMP_NUM_THREADS=1

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    # the condor system creates a scratch directory for us,
    # and cleans up afterward
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir

    conda activate bnl
else
    # otherwise use the TMPDIR
    tmpdir='.'
    mkdir -p $tmpdir
fi

for seed in `seq $1 $2`
do
    python run.py ${seed} $tmpdir/out_${1}_${2}_${seed}.pkl
done

python collate.py ${3}/out_${1}_${2}.pkl $tmpdir/out_${1}_${2}_*

for seed in `seq $1 $2`
do
    rm -f $tmpdir/out_${1}_${2}_${seed}.pkl
done
