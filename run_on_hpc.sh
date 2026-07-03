#!/bin/bash -l

#SBATCH --gres=gpu::1
#SBATCH --time=6:00:00
#SBATCH --export=NONE
#SBATCH --job-name=railbench-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


unset SLURM_EXPORT_ENV # remove env variable for slurm to avoid any sideffects (recommended)
module load python # Load Python of th eHPC system

# for UV: set PATH with uv binary, set cache directory (so it's not put in $HOME)
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$WORK/uv-cache"
mkdir -p "$UV_CACHE_DIR"

# Go to code directy, install missing dependencies (if any) and run the training
cd "$WORK/rail-bench-object-detection"

mkdir -p logs
uv sync --frozen

cd src
uv run python main.py