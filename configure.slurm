#!/bin/bash
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A 2020-3-30

#SBATCH -J IS2-config         # job name
#SBATCH -t 00:10:00           # time requested

#SBATCH --nodes=1             # number of nodes

# Use any nodes available with all cores per node:
#SBATCH --cpus-per-task=2    # number of cpus per MPI process (2: all cores, ignore hyperthreading)

# Use only Haswell nodes
# SBATCH -C Haswell
# SBATCH --ntasks-per-node=32 # number of MPI processes per node (Haswell)
# SBATCH -n 128

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# set up environment
ENV_CONFIG=1

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

RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2

# launch job
cd $RUNDIR
srun make clean
srun -n 1 $RUNDIR/configure --enable-mpi --enable-special-relativity --with-problem=IntSh2-paper1 --with-eos=adiabatic --with-coord=cartesian --with-order=2p --with-integrator=vl --with-flux=hlld --with-gas=mhd --enable-smr CC="/opt/cray/pe/craype/2.6.1/bin/cc" CFLAGS="-I/pdc/vol/gsl/2.3/GNU/8.3/include" CPP="/opt/cray/pe/craype/2.6.1/bin/CC" CPPFLAGS="-I/pdc/vol/gsl/2.3/GNU/8.3/include" LDFLAGS="-L/pdc/vol/gsl/2.3/GNU/8.3/lib" LIBS="-lgsl"
srun -n 1 make -j all