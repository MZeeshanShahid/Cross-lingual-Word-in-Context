#!/bin/bash
#SBATCH --job-name=c_predict
#SBATCH --account=ec403
#SBATCH --time=00:30:00
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

mkdir -p predictions

python3 predict.py models/c_no_only.pt \
  /fp/projects01/ec403/IN5550/obligatories/2/no_dev.jsonl.gz \
  predictions/no_only.jsonl

python3 predict.py models/c_de_en_no.pt \
  /fp/projects01/ec403/IN5550/obligatories/2/no_dev.jsonl.gz \
  predictions/de_en_no.jsonl

python3 predict.py models/c_de_en_es_no.pt \
  /fp/projects01/ec403/IN5550/obligatories/2/no_dev.jsonl.gz \
  predictions/de_en_es_no.jsonl

echo "--- no_only ---"
python3 evaluation.py predictions/no_only.jsonl \
  /fp/projects01/ec403/IN5550/obligatories/2/no_dev.jsonl.gz

echo "--- de_en_no ---"
python3 evaluation.py predictions/de_en_no.jsonl \
  /fp/projects01/ec403/IN5550/obligatories/2/no_dev.jsonl.gz

echo "--- de_en_es_no ---"
python3 evaluation.py predictions/de_en_es_no.jsonl \
  /fp/projects01/ec403/IN5550/obligatories/2/no_dev.jsonl.gz