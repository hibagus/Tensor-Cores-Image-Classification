#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Stampede2 SKX nodes
#
#   *** Serial Job on SKX Normal Queue ***
# 
# Last revised: 20 Oct 2017
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch skx.serial.slurm" on a Stampede2 login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#
#   -- For a good way to run multiple serial executables at the
#        same time, execute "module load launcher" followed
#        by "module help launcher".

#----------------------------------------------------

#SBATCH -J Bagus_Cats&Dogs         # Job name
#SBATCH -o Bagus_Cats&Dogs.o%j     # Name of stdout output file
#SBATCH -e Bagus_Cats&Dogs.e%j     # Name of stderr error file
#SBATCH -p v100                     # Queue (partition) name
#SBATCH -N 1                             # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                             # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 23:59:59                      # Run time (hh:mm:ss)
#SBATCH --mail-user=bagus@utexas.edu
#SBATCH --mail-type=all                  # Send email at begin and end of job
#SBATCH -A A-ee6                         # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...

source $HOME/.bashrc
conda activate CatsandDogsClassification
sleep infinity
# ---------------------------------------------------
