#!/bin/bash

#SBATCH -A snic2021-3-29      # allocation

#SBATCH -J FsynExp
#SBATCH -t 24:00:00           # time requested

#SBATCH --nodes=1             # number of nodes
#SBATCH -p main               # partition
#SBATCH --mem=512G            # memory requested

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

nproc=-1 # use all cores available

# module setup
module load PDC

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate /cfs/klemming/projects/snic/snic2020-4-12/ppjanka/anaconda/envs/intsh2-tf
echo Using python from $(which python). # print to have a record of which env is used

module load ffmpeg

# environment vars
RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/corrT2_press/prod1_corr_ampl
CORR_AMPL=50

# launch job
cd $RUNDIR


# process diagnostics
srun python paper1_dashboard.py -expLongFsyn $RUNDIR/results_corr0ampl${CORR_AMPL} $RUNDIR/results_corr1ampl${CORR_AMPL} -nproc $nproc -opt_tf 0 -opt_numba 1 -convert_vtk 1 -tar_when_done 0 -force_recalc 0 # -low_memory 1
