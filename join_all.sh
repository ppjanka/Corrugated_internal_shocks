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
nproc=1
last_rst_only=1
tar_when_done=0
athena_dir=".."
for ARGUMENT in ${@:1}
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    declare "$KEY"="$VALUE"
done
echo join_all.sh will proceed with $nproc processes.

# RST FILES ---------------------------------------------------------------------

echo Processing rst files...

if [[ $(ls $1/id*/*.rst 2> /dev/null) && ! -d $1/joined_rst ]]; then
    mkdir $1/joined_rst
fi
mv $1/id*/*.rst $1/joined_rst 2> /dev/null

if [ $last_rst_only -eq 1 ]; then
    final_rst_no=$(ls $1/joined_rst/*.rst | awk 'BEGIN{max=0000}{n=split($0,words,"."); if(words[n-1] > max) {max=words[n-1]}}END{print max}')
    final_to_rm=$(printf %04d $(($final_rst_no-1)))
    rm $(eval echo $1/joined_rst/*.{0000..${final_to_rm}}.rst) 2> /dev/null
fi

if [[ $tar_when_done -eq 1 && -d $1/joined_rst ]]; then
    workdir=$(pwd)
    cd $1
    tar --use-compress-program=pigz -cvf final_rst.tgz joined_rst
    cd $workdir
    if [ -f $1/final_rst.tgz ]
    then
        rm -r $1/joined_rst
    fi
fi

echo  - rst files processed.

# VTK FILES ---------------------------------------------------------------------

echo Processing vtk files...

MIN_VTK_FILESIZE=1000 # minimal expected vtk filesize, in bytes

# compile join_vtk.c on first use
if [ ! -f $athena_dir/vis/vtk/join_vtk ]; then
    echo Compiling join_vtk on first use.
    gcc -Wall -W -o $athena_dir/vis/vtk/join_vtk $athena_dir/vis/vtk/join_vtk.c -lm
fi

# join the vtk files
if [ ! -d $1/joined_vtk ]; then
    mkdir $1/joined_vtk
fi
nlevels=$(ls -d $1/id0/lev* 2> /dev/null | wc -l)
echo The snapshot directory contains $nlevels mesh refinement levels.
# set up for parallel processing
get_filesize () {
    stat -c%s "$1"
}
process_snapshot () {
    # args: <the global $1, results directory> <filepath to process>`
    declare filename=$(basename $2)
    if [[ ( $nlevels -eq 0 && ! -f $1/joined_vtk/$filename ) || ! -f $1/joined_vtk/lev$nlevels/$filename ]]; then
        declare fileno=$(echo $filename | awk '{split($0,words,"."); print words[2];}')
        echo $fileno
        $athena_dir/vis/vtk/join_vtk -o $1/joined_vtk/$filename $1/id*/*$fileno.vtk
        if [[ -f $1/joined_vtk/$filename && $(get_filesize $1/joined_vtk/$filename) -gt $MIN_VTK_FILESIZE ]]; then
            rm $1/id*/*$fileno.vtk 2> /dev/null
        fi
        for level in $(seq 1 $nlevels)
        do
            mkdir $1/joined_vtk/lev$level
            $athena_dir/vis/vtk/join_vtk -o $1/joined_vtk/lev$level/$filename $1/id*/lev$level/*$fileno.vtk
            if [[ -f $1/joined_vtk/lev$level/$filename && $(get_filesize $1/joined_vtk/lev$level/$filename) -gt $MIN_VTK_FILESIZE ]]; then
                rm $1/id*/lev$level/*$fileno.vtk 2> /dev/null
            fi
        done
    fi
}
if [[ $nproc -gt 1 && $(parallel --version &> /dev/null) ]]; then
    # process in parallel
    export -f get_filesize
    export -f process_snapshot
    ls $1/id0/*.vtk 2> /dev/null | parallel -I% --max-args 1 --jobs $nproc process_snapshot $1 %
else
    # process sequentially
    for snapfile in $1/id0/*.vtk; do
        process_snapshot "$1" "$snapfile"
    done
fi

echo  - vtk files processed.

# CLEANUP --------------------------------------------------------------------

echo Directory cleanup...

MIN_RST_FILESIZE=1000000 # minimal expected rst filesize, in bytes

if [ -f $1/id0/*.hst ]; then
    mv $1/id0/*.hst $1
fi
function remove_folder {
    declare -i n_elems=$(ls $1/* 2> /dev/null | wc -l)
    if [ $n_elems -eq 0 ]; then
        rm -r $1
    fi
}
if [[ $nproc -gt 1 && $(parallel --version &> /dev/null) ]]; then
    export -f remove_folder
    ls -d $1/id* 2> /dev/null | parallel -I% --max-args 1 --jobs $nproc remove_folder %
else
    for folder in $1/id*; do
        remove_folder "$folder" 2> /dev/null
    done
fi

if [[ -d $1 && $tar_when_done -eq 1 ]]; then
    dirname=$(basename $1)
    tar --use-compress-program=pigz -cvf $dirname.tgz $dirname
    if [[ -f $1.tgz && $(get_filesize $1.tgz) -gt $MIN_RST_FILESIZE  ]]
    then
        rm -r $1
    fi
fi

echo  - directory cleanup done.