#!/usr/bin/env bash

# parse arguments and optimize
declare taropt=''
$(pigz -V &> /dev/null) && \
  taropt='--use-compress-program=pigz'
declare force_recalc=false
while getopts :f opt; do
    case $opt in
        f) force_recalc=true;;
        *) echo "Unknown option used. Aborting."; exit;;
    esac
done

echo Unpacking history.pkl from results archives..
for archive in $(ls results*.tgz 2> /dev/null); do
    # figure out the simulations ID
    echo "$archive"
    declare label=$(basename -s .tgz "$archive")
    label=${label:8}
    declare outfile="history_$label.pkl"
    if [[ $force_recalc || ! -e $outfile ]]; then
    # find out the path of history.pkl
        declare hisfile="$(basename -s .tgz "$archive")/history.pkl"
        # extract
        tar $taropt -xf "$archive" "$hisfile"
        # rename and clean up
        mv "$hisfile" "$outfile"
        declare hisdir=$(dirname "$hisfile")
        if [ $(ls $hisdir/* 2> /dev/null | wc -l) -eq 0 ]; then
            rm -r $hisdir
        fi
    fi
done
echo done.