#!/bin/bash
#SBATCH -o logs/finetune_sft_shard_1.log-%j
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=48:00:00

# Loading Modules
source /etc/profile
module load conda/Python-ML-2025b-pytorch

python scripts/finetune_sft.py --num_shards 8 --shard_id 1
