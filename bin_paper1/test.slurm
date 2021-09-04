#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A 2020-3-30

#SBATCH -J IntSh2             # job name
#SBATCH -t 24:00:00           # time requested

#SBATCH --nodes=1             # number of nodes

# Use any nodes available with all cores per node:
# SBATCH --cpus-per-task=2    # number of cpus per MPI process (2: all cores, ignore hyperthreading)

# Use only Haswell nodes
#SBATCH -C Haswell
#SBATCH --ntasks-per-node=32 # number of MPI processes per node (Haswell)
#SBATCH -n 128

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# set up environment -- Intel compiler + Cray MPI + Cray HDF5
#  -> 232 cycles / 5min on 4 Haswell nodes on Beskow with hugepages64M
module swap PrgEnv-cray PrgEnv-intel/6.0.7
module load intel/18.0.0.128
module load cray-mpich/7.7.14
module load cray-hdf5-parallel/1.10.6.1
module load craype-hugepages64M

# use the Cray wrappers
export CXX=CC
export CC=cc
export FC=ftn

# export OMP_NUM_THREADS=1

RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1

# launch job
srun $RUNDIR/athena -i $RUNDIR/athinput.IntSh2_paper1_SMR -d $RUNDIR/beskow_test1_SMR