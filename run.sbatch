#!/bin/bash
#SBATCH --job-name=cultural_grid_search
#SBATCH --output=logs/grid_search_%j.out
#SBATCH --error=logs/grid_search_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=amd-hm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --account=macs30123

# Run your Python script with multiprocessing
python run.py
