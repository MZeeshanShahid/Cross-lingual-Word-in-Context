#!/bin/bash
#SBATCH --job-name=b2_all_langs
#SBATCH --account=ec403
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm-%j.out

source ~/.bashrc
set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-accelerate/1.9.0-foss-2024a-Python-3.12.3

export PYTHONPATH=$PWD:$PYTHONPATH
python3 experiments/b2_all_langs.py