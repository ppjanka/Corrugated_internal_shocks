#!/bin/bash

#SBATCH -A snic2021-3-29

#SBATCH -J IS2caTar           # job name
#SBATCH -t 24:00:00           # time requested

#SBATCH --nodes=1             # number of nodes
#SBATCH -p main

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# environment vars
RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/corrT1_dens/prod1_corr_ampl

# launch job
cd $RUNDIR

declare -a snapshots=( [2]=42 [5]=38 [10]=44 [20]=49 [50]=58 [75]=61 [100]=65 )

for corr in 0 1
do
    for ampl in 1 2 5 10 20 50 75 100
    do
	# create a string of files to extract
	to_extract=''
	for fno in $(seq $((${snapshots[ampl]}-10)) $((${snapshots[ampl]}+20))); do
	    to_extract+=" results_corr${corr}ampl${ampl}/joined_vtk/IntSh2-p1.$(printf %04d $fno).vtk.pkl"
  	echo "Files to extract for ampl $ampl: $to_extract."
	# once ready, extract
    	tar --use-compress-program=pigz -xf results_corr${corr}ampl${ampl}.tgz $to_extract
    done
done
