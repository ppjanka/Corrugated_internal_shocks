#!/bin/bash

# This script uses join_lis to join all lis outputs in the directory indicated as argument, regardless of the number of id* folders inside it
# File: join_lis.sh
# Author: Patryk Pjanka, 2021
# Usage:
#        1) copy or link join_lis.sh to athena/bin
#        2) run by "./join_lis.sh < folder with id* inside, given as -d to athena run >"

# compile join_lis.c on first use
if [ ! -f ../vis/particle/join_lis ]; then
    echo Compiling join_lis on first use.
    gcc -Wall -W -o ../vis/particle/join_lis ../vis/particle/join_lis.c -lm
fi

# prepare the output folder
rm -r $1/joined_lis
mkdir $1/joined_lis

# find the number of id* folders
nproc=`ls -d $1/id* | wc -l`

# find the number of snapshots
nsnap=`ls $1/id0/*.lis | wc -l`

# find the filename properties
prefix=`ls $1/id0/*.lis | head -n 1 | awk '{split($0,words,"/"); split(words[3], words2, "."); print(words2[1]);}'`
suffix=`ls $1/id0/*.lis | head -n 1 | awk '{split($0,words,"/"); split(words[3], words2, "."); print(words2[3]);}'`

# join using join_lis
( cd $1 && ../../vis/particle/join_lis -p $nproc -o $prefix -i $prefix -s $suffix -d joined_lis -f 0:$(($nsnap-1)) )

# clean up
rm $1/id*/*.lis