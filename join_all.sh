#!/bin/bash

# This script joins all vtk and rst outputs in the directory indicated as argument, regardless of the number of id* and lev* folders inside
# File: join_all.sh
# Author: Patryk Pjanka, 2022
# Instructions:
#   1) copy or link join_all.sh to athena/bin
#   2) usage:
#      "./join_all.sh < folder with id* inside, given as -d to the Athena 4.2 (Athena_Cversion) run >" [arguments]
#   3) arguments"
#      nproc=<int> (default=1) number of processes for parallel processing
#      last_rst_only=<1/0> (default=1) if yes, only the highest-number rst file will be kept
#      tar_when_done=<1/0> (default=0) tar joined_rst and the results folder when done
#   4) pre-requisites:
#      pigz for parallel tar archiving
#      GNU parallel for bash-level parallel processing

# parse arguments, thanks to JRichardsz (https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts)
export nproc=1
export last_rst_only=1
export tar_when_done=0
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    export "$KEY"=$VALUE
done
echo join_all.sh will proceed with $nproc processes.

# RST FILES ---------------------------------------------------------------------

echo Processing rst files...

if [ ! -d $1/joined_rst ]; then
    mkdir $1/joined_rst
fi
mv $1/id*/*.rst $1/joined_rst

if [ $last_rst_only == 1 ]; then
    final_rst_no=`ls $1/joined_rst/*.rst | awk 'BEGIN{max=0000}{n=split($0,words,"."); if(words[n-1] > max) {max=words[n-1]}}END{print max}'`
    final_to_rm=`printf %04d $(($final_rst_no-1))`
    rm `eval echo $1/joined_rst/*.{0000..${final_to_rm}}.rst`
fi

if [ $tar_when_done == 1 ]; then
    workdir=`pwd`
    cd $1
    tar --use-compress-program=pigz -cvf final_rst.tgz joined_rst
    cd $workdir
    if [[ -f $1/final_rst.tgz ]]
    then
        rm -r $1/joined_rst
    fi
fi

echo  - rst files processed.

# VTK FILES ---------------------------------------------------------------------

echo Processing vtk files...

# compile join_vtk.c on first use
if [ ! -f ../vis/vtk/join_vtk ]; then
    echo Compiling join_vtk on first use.
    gcc -Wall -W -o ../vis/vtk/join_vtk ../vis/vtk/join_vtk.c -lm
fi

# join the vtk files
if [ ! -d $1/joined_vtk ]; then
    mkdir $1/joined_vtk
fi
nlevels=$(ls -d $1/id0/lev* | wc -l)
export nlevels
echo The snapshot directory contains $nlevels mesh refinement levels.
# set up for parallel processing
function process_snapshot {
    # args: <the global $1, results directory> <filepath to process>`
    filename=$(basename $2)
    if [ ! -f $1/joined_vtk/lev$nlevels/$filename ]; then
        fileno=$(echo $filename | awk '{split($0,words,"."); print words[2];}')
        echo $fileno
        ../vis/vtk/join_vtk -o $1/joined_vtk/$filename $1/id*/*$fileno.vtk
        rm $1/id*/*$fileno.vtk
        for level in $(seq 1 $nlevels)
        do
            mkdir $1/joined_vtk/lev$level
            ../vis/vtk/join_vtk -o $1/joined_vtk/lev$level/$filename $1/id*/lev$level/*$fileno.vtk
            rm $1/id*/lev$level/*$fileno.vtk
        done
    fi
}
export -f process_snapshot
# process in parallel
ls $1/id0/*.vtk | parallel -I% --max-args 1 --jobs $nproc process_snapshot $1 %

echo  - vtk files processed.

# CLEANUP --------------------------------------------------------------------

echo Directory cleanup...

mv $1/id0/*.hst $1
function remove_folder {
    n_elems=$(ls $1/* | wc -l)
    if [ $n_elems == 0 ]; then
        rm -r $1
    fi
}
export -f remove_folder
ls -d $1/id* | parallel -I% --max-args 1 --jobs $nproc remove_folder %

if [ $tar_when_done == 1 ]; then
    dirname=`echo $1 | awk '{n=split($0,words,"/"); print(words[n]);}`
    tar --use-compress-program=pigz -cvf $dirname.tgz $dirname
    if [[ -f $1.tgz ]]
    then
        rm -r $1
    fi
fi

echo  - directory cleanup done.