#!/bin/bash

#SBATCH -A snic2021-3-29      # allocation

#SBATCH -J IS2pp-t1a50
#SBATCH -t 24:00:00           # time requested

#SBATCH --nodes=1             # number of nodes
#SBATCH -p main               # partition
#SBATCH --mem=512G            # memory requested

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# corrugation amplitude for this script
CORR_TYPE=1
PROD_TYPE=1
CORR_AMPL=50

nproc=32 # use all cores available

# module setup
module load PDC

module load PrgEnv-gnu/8.2.0
module swap gcc gcc/11.2.0

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate /cfs/klemming/projects/snic/snic2020-4-12/ppjanka/anaconda/envs/intsh2-tf
echo Using python from $(which python). # print to have a record of which env is used

module load ffmpeg

# figure out the work directory
declare -a corr_type_folders=( [1]="corrT1_dens" [2]="corrT2_press" [3]="corrT3_vel" [4]="corrT4_bfield" )
declare -a prod_type_folders=( [1]="prod1_corr_ampl" [2]="prod2_shell_width" )
RUNDIR="/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/${corr_type_folders[${CORR_TYPE}]}/${prod_type_folders[${PROD_TYPE}]}"

# launch job
cd $RUNDIR

# remove all restart files but the last one, tar the latter to save file no
echo Joining and cleaning up the snapshot filesystem..
for corr in 0 1; do
    # for some reason, join_vtk.c does not work properly in parallel; hence nproc=1 MUST be used.. ://
    $RUNDIR/join_all.sh "$RUNDIR/results_corr${corr}ampl${CORR_AMPL}" nproc=1 tar_when_done=0 athena_dir="../../.."
done
echo Joining and cleaning done.
echo; echo


# process diagnostics
srun python paper1_dashboard.py -comparison $RUNDIR/results_corr0ampl${CORR_AMPL} $RUNDIR/results_corr1ampl${CORR_AMPL} -nproc $nproc -opt_tf 0 -opt_numba 1 -convert_vtk 1 -tar_when_done 0 -force_recalc 0 # -low_memory 1
#srun python paper1_dashboard.py -curvature $RUNDIR/results_corr1ampl${CORR_AMPL} -nproc $nproc -opt_tf 0 -opt_numba 1 -convert_vtk 1 -tar_when_done 0 -force_recalc 1 # -low_memory 1
