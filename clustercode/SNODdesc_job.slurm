#!/bin/bash
#SBATCH --job-name=SNODdesc      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-499            # job array with 500 tasks
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ib4602@princeton.edu
#SBATCH -o ./report/output.%a.out # STDOUT

# module purge
module load anaconda3/2022.10
# module load julia/1.8.2

# conda activate /scratch/network/ib4602/net-inf

python run_sim.py