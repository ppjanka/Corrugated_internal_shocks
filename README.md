# An MHD view: corrugated internal shocks in relativistic jets

We use MHD simulations to investigate how corrugation (or rippling) of internal shells affects the dynamical and radiative properties of their collisions, in the context of the internal shock models of relativistic jets.

**Reference:** Pjanka et al. (2022), in prep.

**Note:** This is a fork of the publicly available Athena-Cversion MHD code (hereafter, Athena 4.2, please see relevant references in https://princetonuniversity.github.io/Athena-Cversion/AthenaDocsMP), augmented to handle relativistic test particles (see the [Athena4.2_CRtransport repository](https://github.com/ppjanka/Athena4.2_CRtransport)). For full documentation of the original code please go to https://princetonuniversity.github.io/Athena-Cversion/.

## Description of (relevant) contents

The following additions were made to the original Athena 4.2 file tree, to implement the physical problem of interest:
 - src/prob/IntSh2-paper1.c -- the problem generator, sets up the Athena 4.2 environment to run simulations as described in the paper,
 - join_???.sh -- Bash scripts to handle joining restart (rst) and output (vtk) files produced by Athena 4.2 (note: this version of Athena does not support HDF5, and initially produces a separate output file for each snapshot for each process; these scripts allow to collect and / or merge these files), join_all.sh combines the two other,
 - configure_\*.slurm -- submit jobs to configure (compile) Athena for specific HPC clusters,
 - bin_paper1 -- production run setup for the paper:
   - intsh2*.yml -- Anaconda environment definition files for all the Python scripts in the repository,
   - read_vtk.py -- a script from the original Athena repo with slight adjustments, used to read vtk files into Python,
   - athinput.IntSh2_paper1 -- generic input file for the problem (to be copied over to the relevant processing directory and adjusted as needed), 
   - paper1_dashboard.ipynb, .py -- the main diagnostics script of the problem (.py version should be created directly from Jupyter, and is only here for completeness), processes vtk files to calculate the relevant snapshot-level and time-dependent quantities summarized as binary files and / or movies,
   - resolution_study.ipynb, .py, .slurm -- script setup used to measure shock aspect ratio of the corrugated shocks (see paper for more details), .py script is created directly from Jupyter, and .slurm is used to run it on a cluster (adjust as needed),
   - paper1_paperPlots.ipynb -- uses the results of paper1_dashboard (and other scripts) to produce the final plots, as seen in the paper,
   - corrT* -- setup and data products for runs with different ways of shell corrugation (only density and pressure corrugation, i.e., T1 and T2, are eventually used in the paper),
     - prod* -- setup and data product folders for different parameter studies (only the dependence on corrugation amplitude, prod1_corr_ampl, is included in the paper),
       - athinput.IntSh2_paper1 -- the input file specifically adjusted for the configuration of interest,
       - run_\*.slurm, post-process\*.slurm -- slurm scripts to run and post-process each simulation of the parameter study, as indicated by amplitude tags,
       - generate_scripts.sh -- given a pair of run and post-process scripts for one amplitude, this Bash script generates pairs of such scripts for all the remaining amplitudes (thus, any adjustment only needs to be done once, and then propagated using generate_scripts), it also creates a convenient run_all.sh script, that will submit the slurm scripts with appropriate dependencies (so that run*.slurm needs to finish before post-process*.slurm starts),
       - FsynExp.slurm (only in corrT2) -- a slurm script to run post-processing with 1d and 2d profiles of magnetic fields combined in different ways, in order to investigate the increase in synchrotron emission in 2d runs (see paper for details).

## Typical workflow

1. Compile Athena:
 - adjust the configure*.slurm script to match your computing environment,
 - submit to compile,
 - move / copy the athena binary from /bin to the desired processing folder (e.g., /corrT1_dens/prod1_corr_ampl/).
2. Run the simulations and initial post-processing. In one of the directories within the corrT*/prod* tree (new or existing):
 - edit the athinput.* file to match your desired simulation parameters,
 - edit a single run*.slurm, post-process*.slurm pair to reflect the details of your run common to all simulations within the given directory,
 - edit generate_scripts.sh to describe what should change in the run*.slurm, post-process*.slurm scripts between different runs in the given directory (e.g., for prod1_corr_ampl, it is the corrugation amplitude),
 - run generate_scripts.sh to generate all the remaining slurm scripts (*generate.sh <edited run*.slurm> <edited post-process*.slurm>)
 - use the generated submit_all.sh to submit the slurm scripts on the cluster with appropriate dependencies,
 - if needed, edit one of the post-process*.slurm files to run a different type of analysis, then re-run from generate_scripts.sh onwards.
4. Extract final diagnostics:
 - submit FsynExp.slurm to run the magnetic field amplification tests (see paper),
 - submit resolution_study.slurm to extract the shock geometry properties,
 - once ready, run paper1_paperPlots.ipynb to extract the final plots, as they appear in the paper.
