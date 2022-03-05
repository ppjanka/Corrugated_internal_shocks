#!/bin/bash

#SBATCH -A snic2021-3-29      # allocation

#SBATCH -J IS2t3sw2
#SBATCH -t 24:00:00           # time requested

#SBATCH --nodes=2             # number of nodes
#SBATCH -p main               # partition
#SBATCH --cpus-per-task=1     # number of cpus per MPI process (2: all cores, ignore hyperthreading)
NPROC=512 # 2nodes x 2xCPU x 64cores x2 (hyperthreading)

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# shell width for this script
SHELL_WIDTH_FRAC=2
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

RUNDIR='/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/corrT3_vel/prod2_shell_width'

# Usage: run_from_scratch <corr[0,1]>
run_from_scratch() {
    declare corr=$1
    declare extra_args=''
    if [ $corr -eq 0 ]; then
        extra_args='domain1/Nx2=2'
    fi
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
      $extra_args \
     -d $RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL
}
# Usage: process_restarts <results folder containing id* dirs>
process_restarts() {
    if [[ $(ls $1/id*/*.rst 2> /dev/null) && ! -d $1/joined_rst ]]; then
        mkdir $1/joined_rst
    fi
    mv $1/id*/*.rst $1/joined_rst 2> /dev/null

    declare final_rst_no=$( \
        ls $1/joined_rst/*.rst | awk 'BEGIN{max=0000}{n=split($0,words,"."); if(words[n-1] > max) {max=words[n-1]}}END{print max}' \
        || echo 0 \
    )
    echo $final_rst_no
}

# launch job
cd $RUNDIR

for corr in 1 0; do
    declare outdir="$RUNDIR/results_corr${corr}shWidth$SHELL_WIDTH_LABEL"
    if [ -d "$outdir" ]; then
        declare final_rst_no=$(process_restarts "$outdir")
        if [ $final_rst_no -gt 0 ]; then
            echo "Running  from restart no $final_rst_no..."
            srun $RUNDIR/athena -r "$outdir/joined_rst/IntSh2-p1.$(printf "%04i" $final_rst_no).rst" -d "$outdir"
        else
            echo "Running $outdir from scratch..."
            run_from_scratch $corr
        fi
    else
        echo "Running $outdir from scratch..."
        run_from_scratch $corr
    fi
done
