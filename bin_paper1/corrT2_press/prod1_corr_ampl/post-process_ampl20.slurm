#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A 2021-3-29

#SBATCH -J IS2pp-t2a20        # job name
#SBATCH -t 24:00:00           # time requested

#SBATCH --nodes=1             # number of nodes

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# SBATCH --gres=gpu:K420:1 # light-GPU Tegner thin node
# SBATCH --gres=gpu:K80:2  # heavy-GPU Tegner node

nproc=-1 # use all cores available

# module setup
module load anaconda
source activate intsh2-tf
module load pigz # needed for parallelized tar
module load cuda
module load ffmpeg

# environment vars
RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/corrT2_press/prod1_corr_ampl
corr_ampl=20

# launch job
cd $RUNDIR

# process diagnostics
srun python paper1_dashboard.py -comparison $RUNDIR/results_corr0ampl${corr_ampl}.tgz $RUNDIR/results_corr1ampl${corr_ampl}.tgz -nproc $nproc -opt_tf 1 -opt_numba 1 -convert_vtk 1 -tar_when_done 1 -force_recalc 0