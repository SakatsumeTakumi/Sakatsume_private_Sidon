#!/usr/bin/env bash
#SBATCH --job-name=cleanse
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd $SLURM_SUBMIT_DIR

export HF_HOME="/home/sarulab/takumi_sakatsume/Sakatsume_private_Sidon/tmp/hf_${SLURM_JOB_ID}"
NUM_WORKERS=8

LOCAL_ROOT="/home/sarulab/takumi_sakatsume/Sakatsume_private_Sidon/tmp/cleanse_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_ROOT"
trap 'rm -rf "$LOCAL_ROOT"' EXIT

LOCAL_OUTPUT_DIR="${LOCAL_ROOT}/webdataset_output"
LOCAL_HF_OUTPUT_DIR="${LOCAL_ROOT}/hf_dataset"
mkdir -p "$LOCAL_OUTPUT_DIR"

uv run python examples/cleanse_multilingual_librispeech.py \
  --output-dir "$LOCAL_OUTPUT_DIR" \
  --target-sample-rate 48000 --batch-size 32 --loader-workers 16 \
  --num-workers "$NUM_WORKERS" \
  --shard-index "${SHARD_INDEX:=0}" \
  --chunk-seconds 25