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
eval "$(conda shell.bash hook)"
conda activate /cfs/klemming/projects/snic/snic2020-4-12/ppjanka/anaconda/envs/intsh2-tf
which python # print to have a record of which env is used

module load pigz # needed for parallelized tar
module load cuda
module load ffmpeg

# GNU parallel setup
export PATH="/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/gnu-parallel/parallel-20211222/bin:$PATH"

# environment vars
RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/corrT2_press/prod1_corr_ampl
corr_ampl=20

# launch job
cd $RUNDIR

# remove all restart files but the last one, tar the latter to save file no
echo Joining and cleaning up the snapshot filesystem..
for corr in 0 1; do
    #cp $RUNDIR/results_corr${corr}ampl${corr_ampl} $RUNDIR/backup/results_corr${corr}ampl${corr_ampl}
    # for some reason, join_vtk.c does not work properly in parallel; hence nproc=1 MUST be used.. ://
    $RUNDIR/join_all.sh $RUNDIR/results_corr${corr}ampl${corr_ampl} nproc=1 tar_when_done=0 athena_dir=../../..
done
echo  - joining and cleaning done.


# process diagnostics
srun python paper1_dashboard.py -comparison $RUNDIR/results_corr0ampl${corr_ampl} $RUNDIR/results_corr1ampl${corr_ampl} -nproc $nproc -opt_tf 0 -opt_numba 1 -convert_vtk 0 -tar_when_done 0 -force_recalc 1 -low_memory 1