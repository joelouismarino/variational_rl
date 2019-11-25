#!/usr/bin/env bash
#SBATCH --time=72:00:00
#SBATCH --job-name=agent
#SBATCH --mem=20GB
#SBATCH --output=agent.out
#SBATCH --gres=gpu:1
#SBATCH --account=def-bengioy

source ~/ENV/bin/activate
module load httpproxy

python main.py $*