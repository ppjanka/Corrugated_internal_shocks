#!/bin/bash

#SBATCH -A snic2021-3-29      # allocation

#SBATCH -J IS2-config         # job name
#SBATCH -t 00:10:00           # time requested

#SBATCH --nodes=1             # number of nodes
#SBATCH -p main               # partition

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# set up environment -- GCC compiler + Cray MPI + Cray HDF5
module load PDC
module load PrgEnv-gnu/8.2.0
module swap gcc gcc/11.2.0
module load cray-mpich/8.1.11
module load cray-hdf5-parallel/1.12.0.7
module load craype-hugepages64M
module load GSL/2.7-cpeGNU-21.11

# use the Cray wrappers
#export CXX=CC
#export CC=cc
#export FC=ftn

# set up Athena 4.2 environment variables
export CC='cc'
export CFLAGS="-I/pdc/software/21.11/eb/software/GSL/2.7-cpeGNU-21.11/include"
export CPP='CC'
export CPPFLAGS="-I/pdc/software/21.11/eb/software/GSL/2.7-cpeGNU-21.11/include"
export LDFLAGS="-L/pdc/software/21.11/eb/software/GSL/2.7-cpeGNU-21.11/lib"
export LIBS="-lgsl"

# export OMP_NUM_THREADS=1

RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2

# launch job
cd $RUNDIR

# prepare the SMR version
#srun make clean
#srun -n 1 $RUNDIR/configure --enable-mpi --enable-special-relativity --with-problem=IntSh2-paper1 --with-eos=adiabatic --with-coord=cartesian --with-order=2p --with-integrator=vl --with-flux=hlld --with-gas=mhd --enable-smr CC="/opt/cray/pe/craype/2.6.1/bin/cc" CFLAGS="-I/pdc/vol/gsl/2.3/GNU/8.3/include" CPP="/opt/cray/pe/craype/2.6.1/bin/CC" CPPFLAGS="-I/pdc/vol/gsl/2.3/GNU/8.3/include" LDFLAGS="-L/pdc/vol/gsl/2.3/GNU/8.3/lib" LIBS="-lgsl"
#srun -n 1 make -j all MACHINE=dardel
#mv $RUNDIR/bin/athena $RUNDIR/bin_paper1/athena_SMR

# prepare the noSMR version
srun make clean
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
    CC=$CC CFLAGS=$CFLAGS CPP=$CPP CPPFLAGS=$CPPFLAGS LDFLAGS=$LDFLAGS LIBS=$LIBS
srun -n 1 make -j all MACHINE=dardel
#mv $RUNDIR/bin/athena $RUNDIR/bin_paper1/