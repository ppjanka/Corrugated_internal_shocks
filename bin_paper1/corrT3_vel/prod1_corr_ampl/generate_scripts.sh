#!/usr/bin/env bash

# Generate run and post-processing slurm scripts based on the template given as argument
# Patryk Pjanka, 2022

# - will replace the PAR_ENV env variable in each script based on the list hard coded here
declare PAR_ENV='CORR_AMPL'
declare -a VALS=( 1 2 5 10 20 50 75 100 )

for arg in $@; do
    
    declare pathstem=$(dirname $(realpath "$arg"))
    declare template_name=$(basename -s '.slurm' "$arg")
    declare filename

    echo "Generating .slurm scripts based on $template_name in $pathstem... "

    for VAL in ${VALS[@]}; do
        filename=$( echo $template_name | sed 's/[0-9]\+$//' )"$VAL.slurm"
        if [ "$filename" = "$template_name.slurm" ]; then
            continue
        fi
        echo -n "$VAL "
        # change the environment variable value and the job name
        awk -v par_env="$PAR_ENV=" -v par_val=$VAL '{\
          if (substr($0,0,10) == par_env) {\
            printf("%s%s\n",par_env,par_val);\
          } else if (substr($0,0,10) == "#SBATCH -J") {\
            split($0,words);\
            gsub("[0-9]+$",par_val,words[3]);\
            printf("#SBATCH -J %s\n", words[3]);\
          } else {\
            print($0);\
          };\
        }' "$template_name.slurm" > "$pathstem/$filename"
    done
    echo

    echo 'done.'

done