#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A 2020-3-30

#SBATCH -J IS2sWtar           # job name
#SBATCH -t 24:00:00           # time requested

#SBATCH --nodes=1             # number of nodes

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=patryk.pjanka@su.se

# modules
module load pigz

# environment vars
RUNDIR=/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1/prod2_shell_width

# launch job
cd $RUNDIR
for corr in 1 0
do
    for sw in 5
    do
        if [[ -d results_corr${corr}shWidth${sw} ]]
        then
            cd results_corr${corr}shWidth${sw}
            # clean up the folders
            rm -r id{1..2047}
            # process the rst files
            tar --use-compress-program=pigz -cvf final_rst.tgz joined_rst
            rm -r joined_rst
            # tar the whole directory
            cd $RUNDIR
            cp results_corr${corr}shWidth${sw}/history.pkl history_corr${corr}shWidth${sw}.pkl
            tar --use-compress-program=pigz -cvf results_corr${corr}shWidth${sw}.tgz results_corr${corr}shWidth${sw}
            if [[ -f results_corr${corr}shWidth${sw}.tgz ]]
            then
                rm -r results_corr${corr}shWidth${sw}
            fi
        fi
    done
done