#!/bin/bash

# This script uses join_vtk to join all vtk outputs in the directory indicated as argument, regardless of the number of id* folders inside it
# File: join_vtk.sh
# Author: Patryk Pjanka, 2021
# Usage:
#   1) copy or link join_vtk.sh to athena/bin
#   2) usage:
#      serial "./join_vtk.sh < folder with id* inside, given as -d to athena run >"
#      parallel "./join_vtk.sh < folder with id* inside, given as -d to athena run > <nproc>"

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
if [ $# -ge 2 ]; then
    nproc=$2
else
    nproc=1
fi
export nproc
echo join_vtk.sh will proceed with $nproc processes.
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

# remove the fragmented data
#rm $1/id*/*.vtk
#for level in $(seq 1 $nlevels)
#do
#    rm $1/id*/lev$level/*.vtk
#done