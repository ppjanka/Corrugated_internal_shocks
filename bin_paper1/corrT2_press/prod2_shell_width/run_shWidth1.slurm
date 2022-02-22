#!/bin/bash

#SBATCH -A snic2021-3-29      # allocation

#SBATCH -J IS2t2sw1           # job name
#SBATCH -t 12:00:00           # time requested

#SBATCH --nodes=2             # number of nodes
#SBATCH -p main               # partition
#SBATCH --cpus-per-task=1     # number of cpus per MPI process (2: all cores, ignore hyperthreading)
NPROC=512 # 2nodes x 2xCPU x 64cores x2 (hyperthreading)

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# shell width for this script
SHELL_WIDTH_FRAC=1 # shell width as a fraction of the gap size (5 lt-s)
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

# set up environment -- GCC compiler + Cray MPI + Cray HDF5
module load PDC
module load PrgEnv-gnu/8.2.0
module swap gcc gcc/11.2.0
module load cray-mpich/8.1.11
module load cray-hdf5-parallel/1.12.0.7
module load craype-hugepages64M
module load GSL/2.7-cpeGNU-21.11

RUNDIR='/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/corrT2_press/prod2_shell_width'

# launch job
cd $RUNDIR

# corrugated, 2D ------------------------------------------------------------------
corr=1
srun $RUNDIR/athena \
 -i $RUNDIR/athinput.IntSh2_paper1 \
  time/tlim=$TLIM \
  domain1/x1min=-$BOX_SIZE \
  domain1/x1max=$BOX_SIZE \
  domain1/Nx1=$X1_RES \
  domain1/AutoWithNProc=$NPROC \
  problem/xcen_sh1=-$XCEN \
  problem/width_sh1=$SHELL_WIDTH \
  problem/xcen_sh2=$XCEN \
  problem/width_sh2=$SHELL_WIDTH \
  problem/corr_switch=${corr} \
 -d $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL

# non-corrugated, 1D --------------------------------------------------------------
corr=0
srun $RUNDIR/athena \
 -i $RUNDIR/athinput.IntSh2_paper1 \
  time/tlim=$TLIM \
  domain1/x1min=-$BOX_SIZE \
  domain1/x1max=$BOX_SIZE \
  domain1/Nx1=$X1_RES \
  domain1/AutoWithNProc=$NPROC \
  problem/xcen_sh1=-$XCEN \
  problem/width_sh1=$SHELL_WIDTH \
  problem/xcen_sh2=$XCEN \
  problem/width_sh2=$SHELL_WIDTH \
  problem/corr_switch=${corr} \
 -d $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL
