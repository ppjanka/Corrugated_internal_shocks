#!/bin/bash

#SBATCH -A snic2021-3-29      # allocation

#SBATCH -J IS2t4a20           # job name
#SBATCH -t 12:00:00           # time requested

#SBATCH --nodes=2             # number of nodes
#SBATCH -p main               # partition
#SBATCH --cpus-per-task=1     # number of cpus per MPI process (2: all cores, ignore hyperthreading)
nproc=512 # 2nodes x 2xCPU x 64cores x2 (hyperthreading)

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# corrugation amplitude for this script
CORR_AMPL=5

# set up environment -- GCC compiler + Cray MPI + Cray HDF5
module load PDC
module load PrgEnv-gnu/8.2.0
module swap gcc gcc/11.2.0
module load cray-mpich/8.1.11
module load cray-hdf5-parallel/1.12.0.7
module load craype-hugepages64M
module load GSL/2.7-cpeGNU-21.11

RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/corrT4_bfield/prod1_corr_ampl

# launch job
cd $RUNDIR

# corrugated, 2D ------------------------------------------------------------------
corr=1
srun $RUNDIR/athena -i $RUNDIR/athinput.IntSh2_paper1 problem/corr_ampl=0.$CORR_AMPL problem/corr_switch=1 domain1/AutoWithNProc=$nproc -d $RUNDIR/results_corr${corr}ampl$CORR_AMPL

# non-corrugated, 1D --------------------------------------------------------------
corr=0
srun $RUNDIR/athena -i $RUNDIR/athinput.IntSh2_paper1 problem/corr_ampl=0.$CORR_AMPL problem/corr_switch=0 domain1/Nx2=2 domain1/AutoWithNProc=$nproc -d $RUNDIR/results_corr${corr}ampl$CORR_AMPL
