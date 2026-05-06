#!/bin/bash
#SBATCH --job-name=a1_static
#SBATCH --account=ec403
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm-%j.out

source ~/.bashrc
set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3

export PYTHONPATH=$PWD:$PYTHONPATH
python3 experiments/a1_static_baseline.py