#!/bin/bash

#SBATCH -A snic2021-3-29      # allocation

#SBATCH -J IS2t2a1
#SBATCH -t 12:00:00           # time requested

#SBATCH --nodes=2             # number of nodes
#SBATCH -p main               # partition
#SBATCH --cpus-per-task=1     # number of cpus per MPI process (2: all cores, ignore hyperthreading)
nproc=512 # 2nodes x 2xCPU x 64cores x2 (hyperthreading)

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# corrugation amplitude for this script
CORR_AMPL=1

# set up environment -- GCC compiler + Cray MPI + Cray HDF5
module load PDC
module load PrgEnv-gnu/8.2.0
module swap gcc gcc/11.2.0
module load cray-mpich/8.1.11
module load cray-hdf5-parallel/1.12.0.7
module load craype-hugepages64M
module load GSL/2.7-cpeGNU-21.11

RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/corrT2_press/prod1_corr_ampl

# Usage: run_from_scratch <corr[0,1]>
run_from_scratch() {
    declare corr=$1
    declare extra_args=''
    if [ $corr -eq 0 ]; then
        extra_args='domain1/Nx2=2'
    fi
    srun $RUNDIR/athena -i $RUNDIR/athinput.IntSh2_paper1 problem/corr_ampl=0.$CORR_AMPL problem/corr_switch=0 $extra_args domain1/AutoWithNProc=$nproc -d $RUNDIR/results_corr${corr}ampl$CORR_AMPL
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
    declare outdir="$RUNDIR/results_corr${corr}ampl$CORR_AMPL"
    if [ -d "$outdir" ]; then
        declare final_rst_no=$(process_restarts "$outdir")
        if [ $final_rst_no -gt 0 ]; then
            echo "Running corr${corr}ampl$CORR_AMPL from restart no $final_rst_no..."
            srun $RUNDIR/athena -r "$RUNDIR/results_corr${corr}ampl$CORR_AMPL/joined_rst/IntSh2-p1.$(printf "%04i" $final_rst_no).rst" -d "$RUNDIR/results_corr${corr}ampl$CORR_AMPL"
        else
            echo "Running corr${corr}ampl$CORR_AMPL from scratch..."
            run_from_scratch $corr
        fi
    else
        echo "Running corr${corr}ampl$CORR_AMPL from scratch..."
        run_from_scratch $corr
    fi
done
