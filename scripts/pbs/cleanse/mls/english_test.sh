#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -P gag51394
#PBS -j oe
#PBS -k oed

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

module load gcc >/dev/null 2>&1 || true
module load cuda >/dev/null 2>&1 || true

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-/groups/gcb50354/nakata_wataru/huggingface_cache}"

OUTPUT_PATTERN="${OUTPUT_PATTERN:-/groups/gcb50354/mls/english/test-%05d.tar.gz}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-5000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LOADER_WORKERS="${LOADER_WORKERS:-8}"
FEATURE_EXTRACTOR_PATH="${FEATURE_EXTRACTOR_PATH:-}"
DECODER_PATH="${DECODER_PATH:-}"
DEVICE="${DEVICE:-cuda}"
CHUNK_SECONDS="${CHUNK_SECONDS:-25.0}"
HIGHPASS_CUTOFF="${HIGHPASS_CUTOFF:-50.0}"
TARGET_SAMPLE_RATE="${TARGET_SAMPLE_RATE:-48000}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME}}"
LIMIT="${LIMIT:-}"
SKIP_ERRORS="${SKIP_ERRORS:-false}"
DRY_RUN="${DRY_RUN:-false}"
CHECKPOINT_REPO="${CHECKPOINT_REPO:-sarulab-speech/sidon-v0.1}"
FEATURE_EXTRACTOR_FILENAME="${FEATURE_EXTRACTOR_FILENAME:-feature_extractor_cuda.pt}"
DECODER_FILENAME="${DECODER_FILENAME:-decoder_cuda.pt}"
HF_TOKEN="${HF_TOKEN:-}"

cmd=(
  uv run python scripts/examples/cleanse_mls.py
  --output-pattern "$OUTPUT_PATTERN"
  --samples-per-shard "$SAMPLES_PER_SHARD"
  --batch-size "$BATCH_SIZE"
  --loader-workers "$LOADER_WORKERS"
  --device "$DEVICE"
  --chunk-seconds "$CHUNK_SECONDS"
  --highpass-cutoff "$HIGHPASS_CUTOFF"
  --target-sample-rate "$TARGET_SAMPLE_RATE"
  --hf-cache-dir "$HF_CACHE_DIR"
  --checkpoint-repo "$CHECKPOINT_REPO"
  --feature-extractor-filename "$FEATURE_EXTRACTOR_FILENAME"
  --decoder-filename "$DECODER_FILENAME"
  --hf-split test
)

if [ -n "$LIMIT" ]; then
  cmd+=(--limit "$LIMIT")
fi
if [ "$SKIP_ERRORS" = "true" ]; then
  cmd+=(--skip-errors)
fi
if [ "$DRY_RUN" = "true" ]; then
  cmd+=(--dry-run)
fi
if [ -n "$FEATURE_EXTRACTOR_PATH" ]; then
  cmd+=(--feature-extractor "$FEATURE_EXTRACTOR_PATH")
fi
if [ -n "$DECODER_PATH" ]; then
  cmd+=(--decoder "$DECODER_PATH")
fi
if [ -n "$HF_TOKEN" ]; then
  cmd+=(--hf-token "$HF_TOKEN")
fi

printf 'Launching cleanser:\n  %s\n' "${cmd[*]}"
"${cmd[@]}"
