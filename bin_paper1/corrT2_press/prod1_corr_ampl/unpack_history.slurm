#!/bin/bash

#SBATCH -A snic2021-3-29      # allocation

#SBATCH -J IS2-t2unpack
#SBATCH -t 01:00:00           # time requested

#SBATCH --nodes=1             # number of nodes
#SBATCH -p main               # partition

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# module setup
module load PDC

# environment vars
RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/corrT2_press/prod1_corr_ampl

# launch job
cd $RUNDIR
./unpack_history.sh