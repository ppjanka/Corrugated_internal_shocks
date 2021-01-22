#!/bin/bash
rm -r $1/joined
mkdir $1/joined
nlevels=$(ls -d $1/id0/lev* | wc -l)
for filepath in $1/id0/*.vtk
do
    filename=$(basename $filepath)
    fileno=$(echo $filename | awk '{split($0,words,"."); print words[2];}')
    echo $fileno
    ../vis/vtk/join_vtk -o $1/joined/$filename $1/id*/*$fileno.vtk
    for level in $(seq 1 $nlevels)
    do
        mkdir $1/joined/lev$level
        ../vis/vtk/join_vtk -o $1/joined/lev$level/$filename $1/id*/lev$level/*$fileno.vtk
    done
done
rm -r $1/id*