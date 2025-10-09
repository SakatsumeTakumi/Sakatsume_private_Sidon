#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=3:00:00
#PBS -P gag51394
#PBS -j oe
#PBS -k oed

set -euo pipefail

cd "$PBS_O_WORKDIR"

export HF_HOME="/groups/gcb50354/nakata_wataru/huggingface_cache"
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://s3ds.mdx.jp}"
S3_OUTPUT_URI="s3://mls_sidon/english/"
S3_HF_OUTPUT_URI="s3://mls_sidon/english_hf/"
NUM_WORKERS=128


module load gcc

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required but not available in PATH" >&2
  exit 1
fi

if [ -z "${S3_OUTPUT_URI:-}" ]; then
  echo "S3_OUTPUT_URI must be set to the destination for cleansed shards" >&2
  exit 1
fi

if [ -z "${S3_HF_OUTPUT_URI:-}" ]; then
  echo "S3_HF_OUTPUT_URI must be set to the destination for the HF dataset export" >&2
  exit 1
fi

if [ -n "${PBS_ARRAY_INDEX:-}" ]; then
  SHARD_INDEX=$((PBS_ARRAY_INDEX - 1))
  NUM_WORKERS="${NUM_WORKERS:-${PBS_ARRAY_SIZE:-}}"
  if [ -z "$NUM_WORKERS" ]; then
    echo "Unable to determine total number of workers; set NUM_WORKERS when submitting the array job" >&2
    exit 1
  fi
else
  SHARD_INDEX="${SHARD_INDEX:-0}"
  NUM_WORKERS="${NUM_WORKERS:-1}"
fi

printf 'Running shard %s of %s (PBS index %s)\n' "$SHARD_INDEX" "$NUM_WORKERS" "${PBS_ARRAY_INDEX:-single}"

trim_trailing_slash() {
  case "$1" in
    */) printf '%s' "${1%/}" ;;
    *) printf '%s' "$1" ;;
  esac
}

S3_OUTPUT_BASE="$(trim_trailing_slash "$S3_OUTPUT_URI")"
S3_HF_OUTPUT_BASE="$(trim_trailing_slash "$S3_HF_OUTPUT_URI")"
SHARD_SUFFIX=$(printf 'shard_%05d' "$SHARD_INDEX")
S3_OUTPUT_URI_SHARD="${S3_OUTPUT_BASE}/${SHARD_SUFFIX}"
S3_HF_OUTPUT_URI_SHARD="${S3_HF_OUTPUT_BASE}/${SHARD_SUFFIX}"
S3_COMPLETION_OBJECT="${S3_HF_OUTPUT_URI_SHARD}/completed.txt"

AWS_ENDPOINT_FLAG=""
if [ -n "${S3_ENDPOINT_URL:-}" ]; then
  AWS_ENDPOINT_FLAG="--endpoint-url $S3_ENDPOINT_URL"
fi

AWS_SYNC_FLAGS="--no-progress"
if [ -n "${AWS_ENDPOINT_FLAG}" ]; then
  AWS_SYNC_FLAGS="$AWS_SYNC_FLAGS $AWS_ENDPOINT_FLAG"
fi

if aws s3 ls "$S3_COMPLETION_OBJECT" $AWS_ENDPOINT_FLAG >/dev/null 2>&1; then
  echo "Shard $SHARD_INDEX already completed; skipping"
  exit 0
fi

LOCAL_ROOT="${TMPDIR:-/tmp}/sidon_cleansing_${PBS_JOBID:-$$}_${SHARD_SUFFIX}"
LOCAL_OUTPUT_DIR="${LOCAL_ROOT}/webdataset_output"
LOCAL_HF_OUTPUT_DIR="${LOCAL_ROOT}/hf_dataset"
mkdir -p "$LOCAL_OUTPUT_DIR"
trap 'rm -rf "$LOCAL_ROOT"' EXIT

uv run python scripts/examples/cleanse_multilingual_librispeech.py \
  --output-dir "$LOCAL_OUTPUT_DIR" \
  --hf-dataset parler-tts/mls_eng \
  --split train \
  --hf-output-dir "$LOCAL_HF_OUTPUT_DIR" \
  --target-sample-rate 48000 --batch-size 32 --loader-workers 16 \
  --num-workers "$NUM_WORKERS" \
  --shard-index "$SHARD_INDEX" \
  --chunk-seconds 25

if [ -d "$LOCAL_OUTPUT_DIR" ] && find "$LOCAL_OUTPUT_DIR" -type f -mindepth 1 -print -quit >/dev/null 2>&1; then
  aws s3 sync "$LOCAL_OUTPUT_DIR" "$S3_OUTPUT_URI_SHARD" $AWS_SYNC_FLAGS
else
  echo "No webdataset shards found in $LOCAL_OUTPUT_DIR; skipping upload"
fi

if [ ! -d "$LOCAL_HF_OUTPUT_DIR" ]; then
  echo "Hugging Face export directory missing; expected $LOCAL_HF_OUTPUT_DIR" >&2
  exit 1
fi

aws s3 sync "$LOCAL_HF_OUTPUT_DIR" "$S3_HF_OUTPUT_URI_SHARD" $AWS_SYNC_FLAGS

COMPLETION_FILE_LOCAL="${LOCAL_ROOT}/completed.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$COMPLETION_FILE_LOCAL"
aws s3 cp "$COMPLETION_FILE_LOCAL" "$S3_COMPLETION_OBJECT" $AWS_SYNC_FLAGS
