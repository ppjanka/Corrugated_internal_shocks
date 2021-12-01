#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A 2020-3-30

#SBATCH -J IS2-PP-sw1         # job name
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

# shell width for this script
SHELL_WIDTH_FRAC=1 # shell width as a fraction of the gap size (5 lt-s)
SHELL_WIDTH_LABEL=`echo $SHELL_WIDTH_FRAC | sed -r 's/[.]+/p/g'`
SHELL_WIDTH=`awk -v s=$SHELL_WIDTH_FRAC 'BEGIN {print s*5.0}'`
XCEN=`awk -v s=$SHELL_WIDTH 'BEGIN {print 2.5 + 0.5*s}'`

# environment vars
RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1
datafolder=prod2_shell_width

# launch job
cd $RUNDIR

# join the files from different processors
if false # these are done in run*.slurm script, to limit the number of files stored
then
    for corr in 0 1
    do
        srun join_rst.sh $datafolder/results_corr${corr}shWidth$SHELL_WIDTH_LABEL
        srun join_vtk.sh $datafolder/results_corr${corr}shWidth$SHELL_WIDTH_LABEL
    done
fi

# process diagnostics
srun python paper1_dashboard.py -comparison $RUNDIR/$datafolder/results_corr0shWidth$SHELL_WIDTH_LABEL $RUNDIR/$datafolder/results_corr1shWidth$SHELL_WIDTH_LABEL -nproc $nproc -opt_tf 1 -opt_numba 1 -convert_vtk 1 -tar_when_done 1 -force_recalc 0