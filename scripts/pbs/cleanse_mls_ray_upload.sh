#!/bin/sh
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -P gag51394
#PBS -j oe
#PBS -k oed
#PBS -v USE_SSH=1

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
export RAY_memory_monitor_interval_ms="${RAY_memory_monitor_interval_ms:-5000}"
export RAY_DATA_VERBOSE_PROGRESS=1

HF_DATASET="${HF_DATASET:-parler-tts/mls_eng}"
HF_LANGUAGE="${HF_LANGUAGE:-}"
HF_SPLIT="${HF_SPLIT:-train}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME}}"
HF_AUDIO_COLUMN="${HF_AUDIO_COLUMN:-audio}"
HF_KEY_COLUMN="${HF_KEY_COLUMN:-id}"
HF_STREAMING="${HF_STREAMING:-true}"
DEFAULT_SAMPLE_RATE="${DEFAULT_SAMPLE_RATE:-}"
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-192}"
RAY_PARALLELISM="${RAY_PARALLELISM:-64}"
RAY_LOCAL_MODE="${RAY_LOCAL_MODE:-}"  # set to non-empty for debugging
TARGET_SAMPLE_RATE="${TARGET_SAMPLE_RATE:-48000}"
CHECKPOINT_REPO="${CHECKPOINT_REPO:-sarulab-speech/sidon-v0.1}"
FEATURE_EXTRACTOR_PATH="${FEATURE_EXTRACTOR_PATH:-}"
DECODER_PATH="${DECODER_PATH:-}"
FEATURE_EXTRACTOR_FILENAME="${FEATURE_EXTRACTOR_FILENAME:-feature_extractor_cuda.pt}"
DECODER_FILENAME="${DECODER_FILENAME:-decoder_cuda.pt}"
DEVICE="${DEVICE:-auto}"
CHUNK_SECONDS="${CHUNK_SECONDS:-25}"
HIGHPASS_CUTOFF="${HIGHPASS_CUTOFF:-50}"
HUB_REPO_ID="${HUB_REPO_ID:-sarulab-speech/mls-sidon}"
HUB_TOKEN="${HUB_TOKEN:-}"
HUB_REVISION="${HUB_REVISION:-}"
HUB_PRIVATE="${HUB_PRIVATE:-false}"
HUB_COMMIT_MESSAGE="${HUB_COMMIT_MESSAGE:-Add Sidon-cleansed split}"
HUB_MAX_SHARD_SIZE="${HUB_MAX_SHARD_SIZE:-2GB}"
HUB_CONFIG_NAME="${HUB_CONFIG_NAME:-english}"
HUB_SPLIT_NAME="${HUB_SPLIT_NAME:-train}"
LOCAL_EXPORT_DIR="${LOCAL_EXPORT_DIR:-${TMPDIR:-/tmp}/sidon_ray_export_${PBS_JOBID:-$$}}"
PRESERVE_LOCAL_EXPORT="${PRESERVE_LOCAL_EXPORT:-false}"
SKIP_PUSH="${SKIP_PUSH:-false}"
DRY_RUN="${DRY_RUN:-false}"

if [ -z "$HUB_REPO_ID" ] && [ "$SKIP_PUSH" != "true" ]; then
  echo "HUB_REPO_ID must be set unless SKIP_PUSH=true" >&2
  exit 1
fi

mkdir -p "$LOCAL_EXPORT_DIR"
if [ "$PRESERVE_LOCAL_EXPORT" != "true" ]; then
  trap 'rm -rf "$LOCAL_EXPORT_DIR"' EXIT
fi

CMD=(
  uv run python scripts/examples/cleanse_mls_ray_upload.py
  --hf-dataset "$HF_DATASET"
  --split "$HF_SPLIT"
  --hf-cache-dir "$HF_CACHE_DIR"
  --hf-audio-column "$HF_AUDIO_COLUMN"
  --batch-size "$BATCH_SIZE"
  --ray-num-cpus "$RAY_NUM_CPUS"
  --target-sample-rate "$TARGET_SAMPLE_RATE"
  --checkpoint-repo "$CHECKPOINT_REPO"
  --feature-extractor-filename "$FEATURE_EXTRACTOR_FILENAME"
  --decoder-filename "$DECODER_FILENAME"
  --device "$DEVICE"
  --chunk-seconds "$CHUNK_SECONDS"
  --highpass-cutoff "$HIGHPASS_CUTOFF"
  --hub-commit-message "$HUB_COMMIT_MESSAGE"
  --hub-max-shard-size "$HUB_MAX_SHARD_SIZE"
  --local-export-dir "$LOCAL_EXPORT_DIR"
)

if [ -n "$HF_LANGUAGE" ]; then
  CMD+=(--language "$HF_LANGUAGE")
fi
if [ -n "$DEFAULT_SAMPLE_RATE" ]; then
  CMD+=(--default-sample-rate "$DEFAULT_SAMPLE_RATE")
fi
if [ -n "$LIMIT" ]; then
  CMD+=(--limit "$LIMIT")
fi
if [ -n "$HF_KEY_COLUMN" ]; then
  CMD+=(--hf-key-column "$HF_KEY_COLUMN")
fi
if [ "$HF_STREAMING" = "true" ]; then
  CMD+=(--hf-streaming)
fi
if [ -n "$FEATURE_EXTRACTOR_PATH" ]; then
  CMD+=(--feature-extractor "$FEATURE_EXTRACTOR_PATH")
fi
if [ -n "$DECODER_PATH" ]; then
  CMD+=(--decoder "$DECODER_PATH")
fi
if [ -n "$HUB_REVISION" ]; then
  CMD+=(--hub-revision "$HUB_REVISION")
fi
if [ -n "$HUB_CONFIG_NAME" ]; then
  CMD+=(--hub-config-name "$HUB_CONFIG_NAME")
fi
if [ -n "$HUB_SPLIT_NAME" ]; then
  CMD+=(--hub-split-name "$HUB_SPLIT_NAME")
fi
if [ "$HUB_PRIVATE" = "true" ]; then
  CMD+=(--hub-private)
fi
if [ "$SKIP_PUSH" = "true" ]; then
  CMD+=(--skip-push)
else
  CMD+=(--hub-repo-id "$HUB_REPO_ID")
  if [ -n "$HUB_TOKEN" ]; then
    CMD+=(--hub-token "$HUB_TOKEN")
  fi
fi
if [ "$DRY_RUN" = "true" ]; then
  CMD+=(--dry-run)
fi
if [ "$RAY_LOCAL_MODE" = "true" ]; then
  CMD+=(--ray-local-mode)
fi
if [ -n "$RAY_PARALLELISM" ]; then
  CMD+=(--ray-parallelism "$RAY_PARALLELISM")
fi

if [ -n "${HF_TOKEN:-}" ]; then
  CMD+=(--hf-token "$HF_TOKEN")
fi

printf 'Launching command:\n  %s\n' "${CMD[*]}"
"${CMD[@]}"

printf 'Cleansed dataset stored at %s\n' "$LOCAL_EXPORT_DIR"
