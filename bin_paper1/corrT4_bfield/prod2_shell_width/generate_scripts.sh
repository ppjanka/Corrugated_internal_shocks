#!/usr/bin/env bash

# Generate run and post-processing slurm scripts based on the template given as argument
# Patryk Pjanka, 2022

# Usage: ./generate_scripts.sh <run_script.slurm> <post-process_script.slurm> 

# - will replace the PAR_ENV env variable in each script based on the list hard coded here
declare PAR_ENV='SHELL_WIDTH_FRAC'
declare -a VALS=( 0.1 0.2 0.5 1 2 5 10 )

# prepare a script with sbatch commands for the whole folder
echo -e '#!/usr/env/bin bash\n' > submit_all.sh
chmod +x submit_all.sh

echo "Generating .slurm scripts for parameter study... "

declare dependency
for VAL in ${VALS[@]}; do
  
  echo -n "$VAL "
  dependency=false

  for arg in $@; do
    
    declare pathstem=$(dirname $(realpath "$arg"))
    declare template_name=$(basename -s '.slurm' "$arg")
    declare filename

    filename=$( echo $template_name | sed 's/[0-9]\+$//' )"$VAL.slurm"

    if [ "$filename" != "$template_name.slurm" ]; then

      # change the environment variable value and the job name
      awk -v par_env="$PAR_ENV=" -v par_val=$VAL '{\
        if (substr($0,0,17) == par_env) {\
          printf("%s%s\n",par_env,par_val);\
        } else if (substr($0,0,10) == "#SBATCH -J") {\
          split($0,words);\
          gsub("[0-9]+$",par_val,words[3]);\
          printf("#SBATCH -J %s\n", words[3]);\
        } else {\
          print($0);\
        };\
      }' "$template_name.slurm" > "$pathstem/$filename"

    fi

    # generate submission commands
    if ! $dependency; then
      echo "jobno=\$(sbatch $filename | awk '{print \$4}'); echo \"Submitted run script \$jobno.\"" >> submit_all.sh
      dependency=true
    else
      echo -e "sbatch -d afterok:\$jobno $filename\n" >> submit_all.sh
    fi

  done

done

echo -e '\ndone.'
