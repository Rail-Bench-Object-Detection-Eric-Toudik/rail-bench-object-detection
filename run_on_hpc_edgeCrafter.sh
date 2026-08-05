#!/bin/bash -l

#SBATCH --gres=gpu:v100:2
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --job-name=railbench-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


unset SLURM_EXPORT_ENV # remove env variable for slurm to avoid any sideffects (recommended)
module load python # Load Python of the HPC system

# for UV: set PATH with uv binary, set cache directory (so it's not put in $HOME)
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$WORK/uv-cache"
mkdir -p "$UV_CACHE_DIR"

# Configure proxy
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

# Go to code directy, install missing dependencies (if any) and run the training
cd "$WORK/rail-bench-object-detection"

mkdir -p logs
uv sync --frozen

cd models/EdgeCrafter/ecdetseg
uv run torchrun --standalone --nproc_per_node=2 train.py -c configs/ecdet/ecdet_l.yml --use-amp --seed=0 -t ecdet_l.pth