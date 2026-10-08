#!/bin/bash -l

#SBATCH --gres=gpu::1
#SBATCH --time=20:00:00
#SBATCH --export=NONE
#SBATCH --job-name=railbench-inf-hybrid-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

unset SLURM_EXPORT_ENV
module load python

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$WORK/uv-cache"
mkdir -p "$UV_CACHE_DIR"

export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

cd "$WORK/rail-bench-object-detection"
mkdir -p logs
uv sync --frozen

uv run python -m src.run_inference \
  --split test \
  --ecdet-config-path config/ecdet_large_1024_config.yaml \
  --hybrid-inference \
  --direct-wandb-artifact eric-walsh-fau/Rail-Bench-Object-Detection-EdgeCrafter/tyrbjv6w_model:v0 \
  --sahi-wandb-artifact eric-walsh-fau/Rail-Bench-Object-Detection-EdgeCrafter/j3qccb4z_model:v0
