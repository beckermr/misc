#!/bin/bash

for seed 130 2314 34123 23431 23432 234 234321 8575 545243 563563; do
  echo $seed

  echo "
  #!/bin/bash
  #SBATCH --job-name=pg${seed}
  #SBATCH --account=metashear
  #SBATCH --partition=bdwall
  #SBATCH --nodes=1
  #SBATCH --ntasks-per-node=1
  #SBATCH --exclusive
  #SBATCH --output=log_pg${seed}_lcrc%j.oe
  #SBATCH --time=12:00:00

  source ~/.bashrc
  conda activate bebop

  srun python test_mdet.py --seed=${seed}
" > job_${seed}.#!/bin/sh
done
