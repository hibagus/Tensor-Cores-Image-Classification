#!/bin/bash

#SBATCH -J Cats&Dogs                     # Job name
#SBATCH -o Cats&Dogs.o%j                 # Name of stdout output file
#SBATCH -e Cats&Dogs.e%j                 # Name of stderr error file
#SBATCH -p v100                          # Queue (partition) name
#SBATCH -N 1                             # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                             # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 23:59:59                      # Run time (hh:mm:ss)
#SBATCH -A A-ee6                         # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...

#source $HOME/.bashrc
#conda activate CatsandDogsClassification
sleep infinity
# ---------------------------------------------------
