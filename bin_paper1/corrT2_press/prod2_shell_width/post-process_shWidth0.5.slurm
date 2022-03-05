#!/bin/bash

#SBATCH -A snic2021-3-29      # allocation

#SBATCH -J IS2pp-t2sw0.5
#SBATCH -t 06:00:00           # time requested

#SBATCH --nodes=1             # number of nodes
#SBATCH -p main               # partition
#SBATCH --mem=512G            # memory requested

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

nproc=96  #-1 # use all cores available

# module setup
module load PDC

module load PrgEnv-gnu/8.2.0
module swap gcc gcc/11.2.0

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate /cfs/klemming/projects/snic/snic2020-4-12/ppjanka/anaconda/envs/intsh2-tf
echo Using python from $(which python). # print to have a record of which env is used

module load ffmpeg

# shell width for this script
SHELL_WIDTH_FRAC=0.5
SHELL_WIDTH_LABEL=`echo $SHELL_WIDTH_FRAC | sed -r 's/[.]+/p/g'`
SHELL_WIDTH=`awk -v s=$SHELL_WIDTH_FRAC 'BEGIN {print s*5.0}'`
XCEN=`awk -v s=$SHELL_WIDTH 'BEGIN {print 2.5 + 0.5*s}'`

# environment vars
RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/corrT2_press/prod2_shell_width

# launch job
cd $RUNDIR

# join the files from different processors
for corr in 0 1; do
    $RUNDIR/join_all.sh "results_corr${corr}shWidth$SHELL_WIDTH_LABEL" nproc=1 tar_when_done=0 athena_dir="../../.."
done

# process diagnostics
srun python paper1_dashboard.py -comparison $RUNDIR/results_corr0shWidth$SHELL_WIDTH_LABEL $RUNDIR/results_corr1shWidth$SHELL_WIDTH_LABEL -nproc $nproc -opt_tf 0 -opt_numba 1 -convert_vtk 1 -tar_when_done 1 -force_recalc 0 # -low_memory 1
