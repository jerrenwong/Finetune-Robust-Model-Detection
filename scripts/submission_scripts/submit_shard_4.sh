#!/bin/bash
#SBATCH -o logs/generate_responses_shard_4.log-%j
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=48:00:00

# Loading Modules
source /etc/profile
module load conda/Python-ML-2025b-pytorch

python scripts/generate_responses.py --num_shards 10 --shard_id 4
