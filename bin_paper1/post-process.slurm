#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A 2020-3-30

#SBATCH -J IS2-PP             # job name
#SBATCH -t 00:30:00           # time requested

#SBATCH --nodes=1             # number of nodes

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# Use a light-GPU Tegner thin node
#SBATCH --gres=gpu:K420:1

RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1

#datafolder=beskow_test1_SMR
#datafolder=beskow_test2_noSMR_hitres
datafolder=beskow_test2_noSMR

# launch job
cd $RUNDIR
srun join_rst.sh $datafolder
srun join_vtk.sh $datafolder