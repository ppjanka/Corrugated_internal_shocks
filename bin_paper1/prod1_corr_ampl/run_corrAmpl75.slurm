#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A 2020-3-30

#SBATCH -J IS2ca5             # job name
#SBATCH -t 16:00:00           # time requested

#SBATCH --nodes=8             # number of nodes

# Use any nodes available with all cores per node:
# SBATCH --cpus-per-task=2    # number of cpus per MPI process (2: all cores, ignore hyperthreading)

# Use only Haswell nodes
#SBATCH -C Haswell
#SBATCH --ntasks-per-node=32 # number of MPI processes per node (Haswell)
#SBATCH -n 256

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# corrugation amplitude for this script
CORR_AMPL=75

# set up environment
ENV_CONFIG=2

if [ $ENV_CONFIG -eq 1 ]
then
    # set up environment -- Intel compiler + Cray MPI + Cray HDF5
    #  -> 232 cycles / 5min on 4 Haswell nodes on Beskow with hugepages64M
    module swap PrgEnv-cray PrgEnv-intel/6.0.7
    module load intel/18.0.0.128
    module load cray-mpich/7.7.14
    module load cray-hdf5-parallel/1.10.6.1
    module load craype-hugepages64M
elif [ $ENV_CONFIG -eq 2 ]
then
    # set up environment -- GCC compiler + Cray MPI + Cray HDF5
    module swap PrgEnv-cray PrgEnv-gnu/6.0.7
    module swap gcc gcc/10.2.0
    module load cray-mpich/7.7.14
    module load cray-hdf5-parallel/1.10.6.1
    module load craype-hugepages64M
else
    echo Incorrect ENV_CONFIG value. Aborting.
    exit
fi
module load gsl/2.3

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

RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/prod1_corr_ampl

# launch job
cd $RUNDIR

# corrugated, 2D
srun $RUNDIR/athena -i $RUNDIR/athinput.IntSh2_paper1 problem/corr_ampl=$CORR_AMPL problem/corr_switch=1 -d $RUNDIR/results_corr1ampl$CORR_AMPL

# non-corrugated, 1D
srun $RUNDIR/athena -i $RUNDIR/athinput.IntSh2_paper1 problem/corr_ampl=$CORR_AMPL problem/corr_switch=0 domain1/Nx2=2 -d $RUNDIR/results_corr0ampl$CORR_AMPL