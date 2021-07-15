#!/bin/bash

# This script uses join_vtk to join all vtk outputs in the directory indicated as argument, regardless of the number of id* folders inside it
# File: join_vtk.sh
# Author: Patryk Pjanka, 2021
# Usage:
#        1) copy or link join_vtk.sh to athena/bin
#        2) run by "./join_vtk.sh < folder with id* inside, given as -d to athena run >"

# compile join_vtk.c on first use
if [ ! -f ../vis/vtk/join_vtk ]; then
    echo Compiling join_vtk on first use.
    gcc -Wall -W -o ../vis/vtk/join_vtk ../vis/vtk/join_vtk.c -lm
fi

# join the vtk files
rm -r $1/joined_vtk
mkdir $1/joined_vtk
nlevels=$(ls -d $1/id0/lev* | wc -l)
for filepath in $1/id0/*.vtk
do
    filename=$(basename $filepath)
    fileno=$(echo $filename | awk '{split($0,words,"."); print words[2];}')
    echo $fileno
    ../vis/vtk/join_vtk -o $1/joined_vtk/$filename $1/id*/*$fileno.vtk
    for level in $(seq 1 $nlevels)
    do
        mkdir $1/joined_vtk/lev$level
        ../vis/vtk/join_vtk -o $1/joined_vtk/lev$level/$filename $1/id*/lev$level/*$fileno.vtk
    done
done
rm $1/id*/*.vtk