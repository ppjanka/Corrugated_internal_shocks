#!/usr/env/bin bash

jobno=$(sbatch run_corrAmpl1.slurm | awk '{print $4}'); echo "Submitted run script $jobno."
sbatch -d afterok:$jobno post-process_ampl1.slurm

jobno=$(sbatch run_corrAmpl2.slurm | awk '{print $4}'); echo "Submitted run script $jobno."
sbatch -d afterok:$jobno post-process_ampl2.slurm

jobno=$(sbatch run_corrAmpl5.slurm | awk '{print $4}'); echo "Submitted run script $jobno."
sbatch -d afterok:$jobno post-process_ampl5.slurm

jobno=$(sbatch run_corrAmpl10.slurm | awk '{print $4}'); echo "Submitted run script $jobno."
sbatch -d afterok:$jobno post-process_ampl10.slurm

jobno=$(sbatch run_corrAmpl20.slurm | awk '{print $4}'); echo "Submitted run script $jobno."
sbatch -d afterok:$jobno post-process_ampl20.slurm

jobno=$(sbatch run_corrAmpl50.slurm | awk '{print $4}'); echo "Submitted run script $jobno."
sbatch -d afterok:$jobno post-process_ampl50.slurm

jobno=$(sbatch run_corrAmpl75.slurm | awk '{print $4}'); echo "Submitted run script $jobno."
sbatch -d afterok:$jobno post-process_ampl75.slurm

jobno=$(sbatch run_corrAmpl100.slurm | awk '{print $4}'); echo "Submitted run script $jobno."
sbatch -d afterok:$jobno post-process_ampl100.slurm

