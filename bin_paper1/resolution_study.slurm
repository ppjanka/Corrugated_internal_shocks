#!/bin/bash

#SBATCH -A snic2021-3-29      # allocation

#SBATCH -J resolution
#SBATCH -t 01:00:00           # time requested

#SBATCH --nodes=1             # number of nodes
#SBATCH -p main               # partition
# SBATCH --mem=512G            # memory requested

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

nproc=24 # use all cores available

# module setup
module load PDC

module load PrgEnv-gnu/8.2.0
module swap gcc gcc/11.2.0

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate /cfs/klemming/projects/snic/snic2020-4-12/ppjanka/anaconda/envs/intsh2-tf
echo Using python from $(which python). # print to have a record of which env is used

# figure out the work directory
RUNDIR="/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1"

# launch job
cd $RUNDIR

# processing
srun python resolution_study.py $nproc
