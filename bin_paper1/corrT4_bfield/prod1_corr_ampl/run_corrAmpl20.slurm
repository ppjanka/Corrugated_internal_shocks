#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A 2021-3-29

#SBATCH -J IS2t4a20           # job name
#SBATCH -t 24:00:00           # time requested

#SBATCH --nodes=8             # number of nodes

# Use any nodes available with all cores per node:
# SBATCH --cpus-per-task=2    # number of cpus per MPI process (2: all cores, ignore hyperthreading)

# Use only Haswell nodes
#SBATCH -C Haswell
#SBATCH --ntasks-per-node=32 # number of MPI processes per node (Haswell)
#SBATCH -n 256
nproc=256

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# corrugation amplitude for this script
CORR_AMPL=20

# set up environment -- GCC compiler + Cray MPI + Cray HDF5
module swap PrgEnv-cray PrgEnv-gnu/6.0.7
module swap gcc gcc/10.2.0
module load cray-mpich/7.7.14
module load cray-hdf5-parallel/1.10.6.1
module load craype-hugepages64M

module load gsl/2.3
module load pigz

# use the Cray wrappers
#export CXX=CC
#export CC=cc
#export FC=ftn

# set up Athena 4.2 environment variables
export CC="/opt/cray/pe/craype/2.6.1/bin/cc"
export CFLAGS="-I/pdc/vol/gsl/2.3/GNU/8.3/include"
export CPP="/opt/cray/pe/craype/2.6.1/bin/CC"
export CPPFLAGS="-I/pdc/vol/gsl/2.3/GNU/8.3/include"
export LDFLAGS="-L/pdc/vol/gsl/2.3/GNU/8.3/lib"
export LIBS="-lgsl"

# export OMP_NUM_THREADS=1

RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/corrT4_bfield/prod1_corr_ampl

# launch job
cd $RUNDIR

# corrugated, 2D ------------------------------------------------------------------
corr=1
srun $RUNDIR/athena -i $RUNDIR/athinput.IntSh2_paper1 problem/corr_ampl=0.$CORR_AMPL problem/corr_switch=1 domain1/AutoWithNProc=$nproc -d $RUNDIR/results_corr${corr}ampl$CORR_AMPL

# remove all restart files but the last one, tar the latter to save file no
echo Joining and cleaning up the snapshot filesystem..
srun ../../../join_all.sh $RUNDIR/results_corr${corr}ampl$CORR_AMPL nproc=$nproc tar_when_done=0
echo  - joining and cleaning done.

# non-corrugated, 1D --------------------------------------------------------------
corr=0
srun $RUNDIR/athena -i $RUNDIR/athinput.IntSh2_paper1 problem/corr_ampl=0.$CORR_AMPL problem/corr_switch=0 domain1/Nx2=2 domain1/AutoWithNProc=$nproc -d $RUNDIR/results_corr${corr}ampl$CORR_AMPL

# remove all restart files but the last one, tar the latter to save file no
echo Joining and cleaning up the snapshot filesystem..
srun ../../../join_all.sh $RUNDIR/results_corr${corr}ampl$CORR_AMPL nproc=$nproc tar_when_done=0
echo  - joining and cleaning done.