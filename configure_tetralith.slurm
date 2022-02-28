#!/bin/bash

#SBATCH -J IS2-config         # job name
#SBATCH -t 00:10:00           # time requested

#SBATCH --nodes=1             # number of nodes

# SBATCH --mail-type=begin
# SBATCH --mail-type=end
# SBATCH --mail-user=patryk.pjanka@su.se

# set up environment -- GCC + IntelMPI + HDF5
module load buildenv-impi-gcc/2018a-eb
module load GSL/2.4-nsc1
module load HDF5/1.8.19-nsc1-intel-2018a-eb

# set up Athena 4.2 compilation variables
export LIBS="-lgsl"

RUNDIR=/proj/snic2020-4-12/users/x_patpj/intsh2

# launch job
cd $RUNDIR

# prepare the SMR version
if false; then
    srun -n 1 make clean
    srun -n 1 $RUNDIR/configure \
        --enable-mpi \
        --enable-special-relativity \
        --with-problem=IntSh2-paper1 \
        --with-eos=adiabatic \
        --with-coord=cartesian \
        --with-order=2p \
        --with-integrator=vl \
        --with-flux=hlld \
        --with-gas=mhd \
        --enable-smr \
        LIBS=$LIBS
    srun -n 1 make -j all
fi

# prepare the noSMR version
srun -n 1 make clean
srun -n 1 $RUNDIR/configure \
    --enable-mpi \
    --enable-special-relativity \
    --with-problem=IntSh2-paper1 \
    --with-eos=adiabatic \
    --with-coord=cartesian \
    --with-order=2p \
    --with-integrator=vl \
    --with-flux=hlld \
    --with-gas=mhd \
    LIBS=$LIBS
srun -n 1 make -j all