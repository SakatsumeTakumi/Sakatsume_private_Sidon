#!/usr/bin/env python3
"""Cleanse Multilingual LibriSpeech with Sidon via Ray Data and push to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import ray
import torch
from datasets import (
    Audio,
    Dataset,
    Features,
    IterableDataset,
    concatenate_datasets,
    load_dataset,
)
from huggingface_hub import hf_hub_download
from ray.data import Dataset as RayDataset

from scripts.examples.cleanse_multilingual_librispeech import decode_audio_entry, load_processor
from scripts.cleanse_webdataset import SpeechDenoisingProcessor

_PROCESSOR: SpeechDenoisingProcessor | None = None
_PROCESSOR_CONFIG: dict[str, Any] | None = None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset", default="facebook/multilingual_librispeech", help="Dataset identifier passed to datasets.load_dataset")
    parser.add_argument("--language", default=None, help="Optional dataset configuration (e.g. english, german)")
    parser.add_argument("--split", default="train", help="Dataset split to cleanse")
    parser.add_argument("--hf-cache-dir", default=None, help="Cache directory forwarded to datasets.load_dataset")
    parser.add_argument("--hf-token", default=None, help="Token used when downloading the source dataset and checkpoints")
    parser.add_argument("--hf-audio-column", default="audio", help="Audio column name in the source dataset")
    parser.add_argument("--hf-key-column", default="id", help="Column treated as the unique identifier")
    parser.add_argument("--default-sample-rate", type=int, default=None, help="Fallback sample rate when the dataset item omits one")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging the pipeline")
    parser.add_argument("--skip-errors", action="store_true", help="Skip samples that fail to cleanse instead of stopping")
    parser.add_argument("--hf-streaming", action="store_true", help="Use datasets streaming mode to avoid upfront materialisation")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size forwarded to Ray map_batches")
    parser.add_argument("--ray-address", default=None, help="Existing Ray cluster address (auto-starts local Ray when unset)")
    parser.add_argument("--ray-num-cpus", type=int, default=None, help="Requested CPU slots when starting a local Ray runtime")
    parser.add_argument("--ray-namespace", default=None, help="Namespace used when initialising Ray")
    parser.add_argument("--ray-local-mode", action="store_true", help="Run Ray operators synchronously for easier debugging")
    parser.add_argument("--ray-parallelism", type=int, default=None, help="Explicit parallelism passed to ray.data.from_huggingface")
    parser.add_argument("--hub-repo-id", required=True, help="Target Hugging Face dataset repository (e.g. org/dataset-name)")
    parser.add_argument("--hub-token", default=None, help="Explicit token used for push_to_hub (falls back to HF_TOKEN)")
    parser.add_argument("--hub-revision", default=None, help="Revision or branch that receives the commit")
    parser.add_argument("--hub-private", action="store_true", help="Create or update the dataset as a private repo")
    parser.add_argument("--hub-commit-message", default="Add Sidon-cleansed split", help="Commit message used for push_to_hub")
    parser.add_argument("--hub-max-shard-size", default="500MB", help="Maximum shard size forwarded to push_to_hub")
    parser.add_argument("--hub-config-name", default=None, help="Optional configuration name used when pushing to the Hub")
    parser.add_argument("--hub-split-name", default=None, help="Optional split name used when pushing to the Hub")
    parser.add_argument("--local-export-dir", default=None, help="Optional directory where the cleansed dataset is saved via save_to_disk")
    parser.add_argument("--skip-push", action="store_true", help="Build the cleansed dataset without pushing to the Hub")
    parser.add_argument("--feature-extractor", default=None, help="Path to feature_extractor_cuda.pt (downloaded when unset)")
    parser.add_argument("--decoder", default=None, help="Path to decoder_cuda.pt (downloaded when unset)")
    parser.add_argument("--checkpoint-repo", default="sarulab-speech/sidon-v0.1", help="Repository that hosts Sidon TorchScript checkpoints")
    parser.add_argument("--feature-extractor-filename", default="feature_extractor_cuda.pt", help="Feature extractor filename within the checkpoint repo")
    parser.add_argument("--decoder-filename", default="decoder_cuda.pt", help="Decoder filename within the checkpoint repo")
    parser.add_argument("--device", default="auto", help="Inference device for the TorchScript models")
    parser.add_argument("--chunk-seconds", type=float, default=30.0, help="Chunk size forwarded to the denoiser")
    parser.add_argument("--highpass-cutoff", type=float, default=50.0, help="High-pass filter applied before resampling")
    parser.add_argument("--target-sample-rate", type=int, default=48_000, help="Sampling rate stored alongside cleansed audio")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and print the plan without running the pipeline")
    return parser


def _resolve_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _resolve_checkpoint_path(path: str | None, repo: str, filename: str, token: str | None) -> str:
    if path:
        return path
    download_args: dict[str, Any] = {"repo_id": repo, "filename": filename}
    if token:
        download_args["token"] = token
    return hf_hub_download(**download_args)


def _init_ray(args: argparse.Namespace) -> None:
    if ray.is_initialized():
        return
    if args.ray_address:
        ray.init(address=args.ray_address, namespace=args.ray_namespace)
        return
    init_kwargs: dict[str, Any] = {}
    if args.ray_num_cpus is not None:
        init_kwargs["num_cpus"] = args.ray_num_cpus
    if args.ray_namespace is not None:
        init_kwargs["namespace"] = args.ray_namespace
    init_kwargs["local_mode"] = args.ray_local_mode
    init_kwargs["ignore_reinit_error"] = True
    ray.init(**init_kwargs)


def _load_source_dataset(args: argparse.Namespace, token: str | None) -> tuple[Dataset | IterableDataset, bool]:
    auth_kwargs: dict[str, Any] = {}
    if token:
        auth_kwargs["use_auth_token"] = token
    streaming = args.hf_streaming
    dataset = load_dataset(
        args.hf_dataset,
        name=args.language,
        split=args.split,
        cache_dir=args.hf_cache_dir,
        streaming=streaming,
        **auth_kwargs,
    )
    if args.limit is not None:
        if streaming:
            from itertools import islice

            limited = list(islice(dataset, args.limit))
            dataset = Dataset.from_list(limited)
            streaming = False
        else:
            limit = min(args.limit, len(dataset))
            dataset = dataset.select(range(limit))
    return dataset, streaming


def _hf_to_ray(
    dataset: Dataset | IterableDataset,
    parallelism: int | None,
    streaming: bool,
    *,
    stream_chunk_size: int = 2048,
) -> RayDataset:
    if streaming:
        from itertools import islice
        import pyarrow as pa
        from ray.data.block import BlockMetadata
        from ray.data.datasource import Datasource, ReadTask

        class HuggingFaceStreamingDatasource(Datasource):
            def __init__(self, iterable: IterableDataset, chunk_size: int):
                self._dataset = iterable
                self._chunk_size = max(chunk_size, 1)

            def estimate_inmemory_data_size(self) -> int | None:
                return None

            def get_read_tasks(self, requested_parallelism: int) -> list[ReadTask]:
                def read_iter():
                    iterator = iter(self._dataset)
                    while True:
                        batch = list(islice(iterator, self._chunk_size))
                        if not batch:
                            break
                        yield pa.Table.from_pylist(batch)

                metadata = BlockMetadata(
                    num_rows=None,
                    size_bytes=None,
                    exec_stats=None,
                    input_files=None,
                )
                return [ReadTask(read_iter, metadata)]

        datasource = HuggingFaceStreamingDatasource(
            dataset,
            chunk_size=stream_chunk_size,
        )
        # In streaming mode we cannot parallelise reads without random access support.
        return ray.data.read_datasource(datasource, override_num_blocks=1)

    if hasattr(ray.data, "from_huggingface"):
        kwargs: dict[str, Any] = {}
        if parallelism is not None:
            kwargs["parallelism"] = parallelism
        try:
            return ray.data.from_huggingface(dataset, **kwargs)
        except AttributeError:
            pass

    table = dataset.data.table  # pyarrow.Table backed by on-disk mmap
    total_rows = len(dataset)
    if total_rows == 0:
        return ray.data.from_arrow([])

    if parallelism and parallelism > 0:
        chunk_size = max(total_rows // parallelism, 1)
    else:
        chunk_size = total_rows

    tables = []
    for start in range(0, total_rows, chunk_size):
        length = min(chunk_size, total_rows - start)
        tables.append(table.slice(start, length))

    return ray.data.from_arrow(tables)


def _ensure_processor(config: dict[str, Any]) -> SpeechDenoisingProcessor:
    global _PROCESSOR, _PROCESSOR_CONFIG
    if _PROCESSOR is not None and _PROCESSOR_CONFIG == config:
        return _PROCESSOR
    runtime_args = SimpleNamespace(
        device=config["device"],
        chunk_seconds=config["chunk_seconds"],
        highpass_cutoff=config["highpass_cutoff"],
        target_sample_rate=config["target_sample_rate"],
    )
    _PROCESSOR = load_processor(runtime_args, config["feature_extractor_path"], config["decoder_path"])
    _PROCESSOR_CONFIG = dict(config)
    return _PROCESSOR


def _build_clean_batch_fn(
    *,
    audio_column: str,
    key_column: str | None,
    audio_feature: Audio | None,
    default_sample_rate: int | None,
    feature_sampling_rate: int | None,
    processor_config: dict[str, Any],
    skip_errors: bool,
    target_sample_rate: int,
):
    def cleanse_batch(batch: pd.DataFrame) -> pd.DataFrame:
        processor = _ensure_processor(processor_config)
        records = []
        pandas_records = batch.to_dict(orient="records")
        indices = list(batch.index)
        for idx, record in enumerate(pandas_records):
            sample_key = record.get(key_column) if key_column else None
            if sample_key is None:
                sample_key = indices[idx] if idx < len(indices) else None
            audio_entry = record.get(audio_column)
            if audio_entry is None:
                if skip_errors:
                    continue
                raise ValueError(f"Sample {sample_key} missing audio column '{audio_column}'")
            decoded = decode_audio_entry(
                audio_entry,
                audio_feature=audio_feature,
                sample_key=str(sample_key) if sample_key is not None else None,
                default_sample_rate=default_sample_rate,
                feature_sampling_rate=feature_sampling_rate,
                skip_errors=skip_errors,
            )
            if decoded is None:
                continue
            waveform, sample_rate = decoded
            restored = processor.process(waveform, sample_rate)
            record[audio_column] = {
                "array": restored.to(dtype=torch.float32).cpu().numpy().astype(np.float32),
                "sampling_rate": target_sample_rate,
            }
            records.append(record)
        if not records:
            return pd.DataFrame(columns=batch.columns)
        return pd.DataFrame.from_records(records)

    return cleanse_batch


def _ray_to_hf_dataset(dataset: RayDataset, features: Features) -> Dataset:
    columns = list(features.keys())
    shards: list[Dataset] = []

    for batch in dataset.iter_batches(batch_format="pandas"):
        if batch.empty:
            continue
        hf_batch = Dataset.from_pandas(batch, preserve_index=False)
        if "__index_level_0__" in hf_batch.column_names:
            hf_batch = hf_batch.remove_columns("__index_level_0__")
        shards.append(hf_batch)

    if not shards:
        empty_data = {name: [] for name in columns}
        return Dataset.from_dict(empty_data, features=features)

    combined = concatenate_datasets(shards)
    try:
        combined = combined.cast(features)
    except Exception as exc:
        print(f"Warning: failed to cast to target features ({exc}); proceeding without cast")
    return combined


def main() -> None:
    args = build_argparser().parse_args()

    dataset_token = _resolve_token(args.hf_token)
    feature_extractor_path = _resolve_checkpoint_path(
        args.feature_extractor,
        args.checkpoint_repo,
        args.feature_extractor_filename,
        dataset_token,
    )
    decoder_path = _resolve_checkpoint_path(
        args.decoder,
        args.checkpoint_repo,
        args.decoder_filename,
        dataset_token,
    )

    if args.dry_run:
        print("Dry run: would cleanse", args.hf_dataset)
        print("Using feature extractor from", feature_extractor_path)
        print("Using decoder from", decoder_path)
        return

    _init_ray(args)
    source_dataset, is_streaming = _load_source_dataset(args, dataset_token)
    if args.hf_audio_column not in source_dataset.column_names:
        raise ValueError(f"Column '{args.hf_audio_column}' not found in the source dataset")

    key_column = args.hf_key_column
    dataset_label = 'streaming' if is_streaming else 'in-memory'
    if not is_streaming:
        try:
            print(f"Loaded dataset with {len(source_dataset)} records ({dataset_label})")
        except Exception:
            print(f"Loaded dataset ({dataset_label})")
    else:
        print('Loaded dataset in streaming mode (size unknown upfront)')

    if key_column and key_column not in source_dataset.column_names:
        print(f"Warning: column '{key_column}' not found; falling back to dataset indices for sample keys")
        key_column = None

    raw_features = source_dataset.features
    column_features = dict(raw_features)
    audio_feature = column_features.get(args.hf_audio_column)
    feature_sampling_rate = None
    if isinstance(audio_feature, Audio):
        feature_sampling_rate = audio_feature.sampling_rate
    column_features[args.hf_audio_column] = Audio(sampling_rate=args.target_sample_rate)
    cleansed_features = Features(column_features)

    print("Converting Hugging Face dataset to Ray dataset...")
    chunk_hint = max(args.batch_size * 4, 2048)
    ray_dataset = _hf_to_ray(
        source_dataset,
        args.ray_parallelism,
        is_streaming,
        stream_chunk_size=chunk_hint,
    )
    processor_config = {
        "feature_extractor_path": feature_extractor_path,
        "decoder_path": decoder_path,
        "device": args.device,
        "chunk_seconds": args.chunk_seconds,
        "highpass_cutoff": args.highpass_cutoff,
        "target_sample_rate": args.target_sample_rate,
    }
    cleanse_fn = _build_clean_batch_fn(
        audio_column=args.hf_audio_column,
        key_column=key_column,
        audio_feature=audio_feature if isinstance(audio_feature, Audio) else None,
        default_sample_rate=args.default_sample_rate,
        feature_sampling_rate=feature_sampling_rate,
        processor_config=processor_config,
        skip_errors=args.skip_errors,
        target_sample_rate=args.target_sample_rate,
    )
    cleansed_ray = ray_dataset.map_batches(
        cleanse_fn,
        batch_size=args.batch_size,
        batch_format="pandas",
    )

    cleansed_dataset = _ray_to_hf_dataset(cleansed_ray, cleansed_features)

    if args.local_export_dir:
        cleansed_dataset.save_to_disk(args.local_export_dir)

    if args.skip_push:
        print("Skip push requested; dataset available locally")
    else:
        hub_token = _resolve_token(args.hub_token) or dataset_token
        push_kwargs = {
            "repo_id": args.hub_repo_id,
            "token": hub_token,
            "private": args.hub_private,
            "max_shard_size": args.hub_max_shard_size,
            "commit_message": args.hub_commit_message,
            "revision": args.hub_revision,
            "config_name": args.hub_config_name,
            "split": args.hub_split_name,
        }
        cleansed_dataset.push_to_hub(**{k: v for k, v in push_kwargs.items() if v is not None})

    print(f"Cleansed {cleansed_dataset.num_rows} samples from {args.hf_dataset}:{args.split}")
    ray.shutdown()


if __name__ == "__main__":
    main()
