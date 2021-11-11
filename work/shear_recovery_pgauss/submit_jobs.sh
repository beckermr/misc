#!/bin/bash

for fn in `ls job_*.sh`; do sbatch $fn; done
