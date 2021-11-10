#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A 2020-3-30

#SBATCH -J IS2sw10            # job name
#SBATCH -t 24:00:00           # time requested

#SBATCH --nodes=64            # number of nodes

# Use any nodes available with all cores per node:
# SBATCH --cpus-per-task=2    # number of cpus per MPI process (2: all cores, ignore hyperthreading)

# Use only Haswell nodes
#SBATCH -C Haswell
#SBATCH --ntasks-per-node=32 # number of MPI processes per node (Haswell)
#SBATCH -n 2048
NPROC=2048

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# shell width for this script
SHELL_WIDTH_FRAC=10 # shell width as a fraction of the gap size (5 lt-s)
SHELL_WIDTH_LABEL=`echo $SHELL_WIDTH_FRAC | sed -r 's/[.]+/p/g'`
SHELL_WIDTH=`awk -v s=$SHELL_WIDTH_FRAC 'BEGIN {print s*5.0}'`

# keep the distance between shells constant (at 5 lt-s)
XCEN=`awk -v s=$SHELL_WIDTH 'BEGIN {print 2.5 + 0.5*s}'`

# keep the distance between each shell and the simulation boundary wide enough (16.5 lt-s or a single shell width, whichever larger)
BOX_SIZE=`awk -v s=$SHELL_WIDTH -v c=$XCEN 'BEGIN {if (s < 16.5) {print c+0.5*s+16.5} else {print c+1.5*s}}'`

# run for as long as it takes for light to cross the box, or 20, whichever longer
TLIM=`awk -v b=$BOX_SIZE 'BEGIN {if (b < 21) {print 20.0} else {print b}}'`

# ensure that resolution is comparable between runs and always a power of 2
X1_RES=`awk -v bs=$BOX_SIZE 'BEGIN {print 2.0^(1+int(log(bs*8192.0/20.0)/log(2.0)))}'`

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

RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/prod2_shell_width

# launch job
cd $RUNDIR

# corrugated, 2D ------------------------------------------------------------------

if true
then

    corr=1
    #srun $RUNDIR/athena -i $RUNDIR/athinput.IntSh2_paper1 time/tlim=$TLIM domain1/x1min=-$BOX_SIZE domain1/x1max=$BOX_SIZE domain1/Nx1=$X1_RES domain1/AutoWithNProc=$NPROC problem/xcen_sh1=-$XCEN problem/width_sh1=$SHELL_WIDTH problem/xcen_sh2=$XCEN problem/width_sh2=$SHELL_WIDTH problem/corr_switch=${corr} -d $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL
    srun $RUNDIR/athena -r $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL/joined_rst/IntSh2-p1.0003.rst -d $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL

    # remove all restart files but the last one, tar the latter to save file no
    echo Cleaning up restart files..
    srun ../join_rst.sh $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL
    final_rst_no=`ls $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL/joined_rst/*.rst | awk 'BEGIN{max=0000}{n=split($0,words,"."); if(words[n-1] > max) {max=words[n-1]}}END{print max}'`
    final_to_rm=`printf %04d $(($final_rst_no-1))`
    for i in {0000..${final_to_rm}}
    do
        rm `eval echo $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL/joined_rst/*$i.rst`
    done
    if false
    then
        tar -cvzf $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL/final_rst.tgz $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL/joined_rst
        rm -r $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL/joined_rst
        echo  - restart files cleaning done.

        # the following really should be done in post-processing (it's serial), but given file number constraints, we may not have a choice...
        echo Cleaning up vtk files...
        cd ..
        srun join_vtk.sh $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL
        echo  - vtk files cleaning done.
    fi

fi

# non-corrugated, 1D --------------------------------------------------------------
if false
then

    corr=0
    srun $RUNDIR/athena -i $RUNDIR/athinput.IntSh2_paper1 time/tlim=$TLIM domain1/x1min=-$BOX_SIZE domain1/x1max=$BOX_SIZE domain1/Nx1=$X1_RES domain1/AutoWithNProc=$NPROC problem/xcen_sh1=-$XCEN problem/width_sh1=$SHELL_WIDTH problem/xcen_sh2=$XCEN problem/width_sh2=$SHELL_WIDTH problem/corr_switch=${corr} domain1/Nx2=2 -d $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL

        # remove all restart files but the last one, tar the latter to save file no
        echo Cleaning up restart files..
        srun ../join_rst.sh $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL
        final_rst_no=`ls $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL/joined_rst/*.rst | awk 'BEGIN{max=0000}{n=split($0,words,"."); if(words[n-1] > max) {max=words[n-1]}}END{print max}'`
        final_to_rm=`printf %04d $(($final_rst_no-1))`
        for i in {0000..${final_to_rm}}
        do
            rm `eval echo $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL/joined_rst/*$i.rst`
        done
        tar -cvzf $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL/final_rst.tgz $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL/joined_rst
        rm -r $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL/joined_rst
        echo  - restart files cleaning done.

        # the following really should be done in post-processing (it's serial), but given file number constraints, we may not have a choice...
        echo Cleaning up vtk files...
        cd ..
        srun join_vtk.sh $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL
        echo  - vtk files cleaning done.

fi