#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A 2020-3-30

#SBATCH -J IS2caTar           # job name
#SBATCH -t 06:00:00           # time requested

#SBATCH --nodes=1             # number of nodes

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# modules
module load pigz

# environment vars
RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/prod1_corr_ampl

# launch job
cd $RUNDIR

for corr in 0 1
do
    for ampl in 50 75 # 1 10 100 2 20 5 50 75
    do
        if [[ -d results_corr${corr}ampl${ampl} ]]
        then
            cd results_corr${corr}ampl${ampl}
            # clean up the folders
            rm -r id{1..255}
            # process the rst files
            tar --use-compress-program=pigz -cvf final_rst.tgz joined_rst
            rm -r joined_rst
            # tar the whole directory
            cd $RUNDIR
            cp results_corr${corr}ampl${ampl}/history.pkl history_corr${corr}ampl${ampl}.pkl
            tar --use-compress-program=pigz -cvf results_corr${corr}ampl${ampl}.tgz results_corr${corr}ampl${ampl}
            if [[ -f results_corr${corr}ampl${ampl}.tgz ]]
            then
                rm -r results_corr${corr}ampl${ampl}
            fi
        fi
    done
done