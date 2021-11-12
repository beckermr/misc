#!/bin/bash

rm job_*.sh

# 130 2314 34123 23431 23432 234 234321 8575 545243 563563
for i in `seq 1 30`; do
  seed=$(python -c "import numpy as np; print(np.random.randint(1, 2**31))")
  echo $seed

  echo "\
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
" > job_${seed}.sh
done
