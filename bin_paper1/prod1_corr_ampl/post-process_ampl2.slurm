#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A 2020-3-30

#SBATCH -J IS2-PP-ca2         # job name
#SBATCH -t 06:00:00           # time requested

#SBATCH --nodes=1             # number of nodes

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# Use a light-GPU Tegner thin node
#SBATCH --gres=gpu:K420:1
nproc=24

# module setup
module load anaconda
source activate intsh2
module load ffmpeg

# environment vars
RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1
datafolder=prod1_corr_ampl
corr_ampl=2

# launch job
cd $RUNDIR

# join the files from different processors
for corr in 0 1
do
    #srun join_rst.sh $datafolder/results_corr${corr}ampl${corr_ampl}
    srun join_vtk.sh $datafolder/results_corr${corr}ampl${corr_ampl}
done

# process diagnostics
srun python paper1_dashboard.py -comparison $RUNDIR/$datafolder/results_corr0ampl${corr_ampl} $RUNDIR/$datafolder/results_corr1ampl${corr_ampl} -nproc $nproc -opt_tf 0 -opt_numba 1 -convert_vtk 1 -force_recalc 0