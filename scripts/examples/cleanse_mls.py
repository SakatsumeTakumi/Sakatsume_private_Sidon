#!/usr/bin/env python3
"""Cleanse the MLS English split with Sidon, preserving metadata and batching via DataLoader."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torchaudio
import webdataset as wds
from datasets import Dataset, load_dataset, Audio
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.cleanse_webdataset import (  # noqa: E402
    SpeechDenoisingProcessor,
    serialise_flac,
)

HF_DATASET = "parler-tts/mls_eng"
HF_SPLIT = "train"
AUDIO_COLUMN = "audio"
KEY_COLUMN = "id"
TEXT_COLUMN = "text"


def resolve_checkpoint(
    candidate_path: str | None,
    repo_id: str,
    filename: str,
    *,
    token: str | None,
) -> str:
    """Return a local path to the requested checkpoint, downloading when needed."""
    if candidate_path:
        return candidate_path
    auth: dict[str, str] = {}
    resolved_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if resolved_token:
        auth["token"] = resolved_token
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, **auth)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            f"Failed to download '{filename}' from '{repo_id}'. Set --feature-extractor/--decoder "
            "or provide valid credentials."
        ) from exc


@dataclass
class CollatedBatch:
    """Batch payload returned by the DataLoader collate function."""

    keys: list[str]
    waveforms: list[torch.Tensor]
    durations: list[float]
    metadata: list[dict[str, Any]]

    @property
    def size(self) -> int:
        return len(self.keys)


def _normalise_value(value: Any) -> Any:
    """Convert tensors and NumPy scalars to serialisable Python objects."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _encode_metadata(metadata: dict[str, Any]) -> bytes:
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        cleaned[key] = _normalise_value(value)
    return json.dumps(cleaned, ensure_ascii=False).encode("utf-8")


def _writer_worker(
    task_queue: "mp.Queue[tuple[str, torch.Tensor, dict[str, Any]]]",
    result_queue: "mp.Queue[tuple[str, str | None, str | None]]",
    output_pattern: str,
    samples_per_shard: int,
    target_sample_rate: int,
    text_column: str | None,
) -> None:
    """Background process that serialises and writes records."""
    try:
        with wds.ShardWriter(output_pattern, maxcount=samples_per_shard) as sink:
            while True:
                item = task_queue.get()
                if item is None:
                    break
                key, waveform, metadata = item
                try:
                    record = serialise_flac(key, waveform, target_sample_rate)
                    text_value = metadata.get(text_column) if text_column else None
                    if text_value:
                        record["text.txt"] = str(text_value)
                    record["metadata.json"] = _encode_metadata(metadata)
                    sink.write(record)
                    result_queue.put(("ok", key, None))
                except Exception as exc:  # pylint: disable=broad-except
                    result_queue.put(("error", key, repr(exc)))
        result_queue.put(("closed", None, None))
    except Exception as exc:  # pylint: disable=broad-except
        result_queue.put(("fatal", None, repr(exc)))


class AsyncShardWriter:
    """Run record serialisation and shard writes in a dedicated process."""

    def __init__(
        self,
        output_pattern: Path,
        samples_per_shard: int,
        target_sample_rate: int,
        text_column: str | None,
        *,
        max_queue_size: int | None = None,
    ) -> None:
        ctx = mp.get_context("spawn")
        queue_size = 0 if not max_queue_size or max_queue_size < 0 else max_queue_size
        self._tasks: mp.Queue = ctx.Queue(maxsize=queue_size)
        self._results: mp.Queue = ctx.Queue()
        self._process = ctx.Process(
            target=_writer_worker,
            args=(
                self._tasks,
                self._results,
                str(output_pattern),
                samples_per_shard,
                target_sample_rate,
                text_column,
            ),
            daemon=True,
        )
        self._process.start()
        self._finished = False
        self._finish_requested = False

    def submit(
        self, key: str, waveform: torch.Tensor, metadata: dict[str, Any]
    ) -> None:
        if self._finish_requested:
            raise RuntimeError("Writer has already been closed")
        cpu_waveform = waveform.detach().cpu().contiguous()
        self._tasks.put((key, cpu_waveform, metadata))

    def poll(self, *, block: bool = False) -> tuple[int, list[tuple[str | None, str]]]:
        drained = 0
        errors: list[tuple[str | None, str]] = []
        while True:
            try:
                if block:
                    message = self._results.get(timeout=0.1)
                else:
                    message = self._results.get_nowait()
            except queue.Empty:
                break
            status, key, payload = message
            if status == "ok":
                drained += 1
            elif status == "error":
                errors.append((key, payload or ""))
            elif status == "fatal":
                self._finished = True
                raise RuntimeError(
                    f"Writer process encountered a fatal error: {payload}"
                )
            elif status == "closed":
                self._finished = True
            else:
                continue
        return drained, errors

    def close(self) -> tuple[int, list[tuple[str | None, str]]]:
        total_drained = 0
        collected_errors: list[tuple[str | None, str]] = []
        if not self._finish_requested:
            self._tasks.put(None)
            self._finish_requested = True
        while True:
            drained, errors = self.poll(block=True)
            total_drained += drained
            collected_errors.extend(errors)
            if self._finished:
                break
        self._process.join()
        return total_drained, collected_errors


def prepare_waveform(audio_entry: dict) -> tuple[torch.Tensor, int]:
    array = audio_entry.get("array")
    sample_rate = audio_entry.get("sampling_rate")
    if array is None or sample_rate is None:
        raise ValueError("Audio entry missing array or sampling_rate")
    waveform = torch.as_tensor(array, dtype=torch.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    sample_rate = 16_000
    return waveform.contiguous(), int(sample_rate)


class CollateFunction:
    """Decode MLS samples into waveforms and metadata for batch processing."""

    def __init__(
        self,
        *,
        metadata_columns: Sequence[str],
        skip_errors: bool,
        audio_column: str,
        key_column: str,
    ) -> None:
        self.metadata_columns = list(metadata_columns)
        self.skip_errors = skip_errors
        self.audio_column = audio_column
        self.key_column = key_column

    def __call__(self, samples: Sequence[dict[str, Any]]) -> CollatedBatch:
        keys: list[str] = []
        waveforms: list[torch.Tensor] = []
        sample_rates: list[int] = []
        metadata: list[dict[str, Any]] = []
        durations: list[float] = []

        for sample in samples:
            raw_key = sample.get(self.key_column)
            key = str(raw_key) if raw_key is not None else uuid.uuid4().hex
            try:
                waveform, sample_rate = prepare_waveform(sample[self.audio_column])
                if sample_rate != 16_000:
                    waveform = torchaudio.functional.resample(
                        waveform, sample_rate, 16_000
                    )
            except Exception as exc:  # pylint: disable=broad-except
                if not self.skip_errors:
                    raise
                print(f"Skipping {key} during decode: {exc}")
                continue

            record_meta: dict[str, Any] = {}
            for column in self.metadata_columns:
                if column == self.audio_column:
                    continue
                record_meta[column] = sample.get(column)
            record_meta.setdefault(self.key_column, key)

            keys.append(key)
            waveforms.append(waveform)
            sample_rates.append(sample_rate)
            metadata.append(record_meta)
            durations.append(waveform.shape[-1] / sample_rate)
        waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)

        return CollatedBatch(
            keys=keys, waveforms=waveforms, metadata=metadata, durations=durations
        )


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    *,
    loader_workers: int,
    collate_fn: CollateFunction,
) -> Iterable[CollatedBatch]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        collate_fn=collate_fn,
    )
    for batch in dataloader:
        if isinstance(batch, CollatedBatch) and batch.size > 0:
            yield batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-pattern", required=True, help="Shard pattern passed to WebDataset"
    )
    parser.add_argument(
        "--samples-per-shard", type=int, default=5_000, help="Maximum records per shard"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Samples processed together"
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=8,
        help="PyTorch DataLoader worker threads",
    )
    parser.add_argument(
        "--feature-extractor", default=None, help="Path to feature_extractor_cuda.pt"
    )
    parser.add_argument("--decoder", default=None, help="Path to decoder_cuda.pt")
    parser.add_argument(
        "--checkpoint-repo",
        default="sarulab-speech/sidon-v0.1",
        help="Hugging Face repo providing Sidon checkpoints",
    )
    parser.add_argument(
        "--feature-extractor-filename",
        default="feature_extractor_cuda.pt",
        help="Feature extractor filename inside the repo",
    )
    parser.add_argument(
        "--decoder-filename",
        default="decoder_cuda.pt",
        help="Decoder filename inside the repo",
    )
    parser.add_argument(
        "--hf-token", default=None, help="Optional Hugging Face token for private repos"
    )
    parser.add_argument(
        "--device", default="auto", help="Inference device (cuda, cpu, or auto)"
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=25.0,
        help="Chunk duration passed to Sidon",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=48_000,
        help="Sample rate of the cleansed audio",
    )
    parser.add_argument(
        "--hf-dataset", default=HF_DATASET, help="Hugging Face dataset to cleanse"
    )
    parser.add_argument("--hf-split", default=HF_SPLIT, help="Dataset split to cleanse")
    parser.add_argument(
        "--hf-name", default=None, help="name argument passed to the load_dataset"
    )
    parser.add_argument(
        "--audio-column", default=AUDIO_COLUMN, help="Column containing audio payloads"
    )
    parser.add_argument(
        "--key-column", default=KEY_COLUMN, help="Column providing unique record ids"
    )
    parser.add_argument(
        "--text-column",
        default=TEXT_COLUMN,
        help="Column containing transcription text",
    )
    parser.add_argument("--hf-cache-dir", default=None, help="datasets cache directory")
    parser.add_argument(
        "--skip-errors", action="store_true", help="Skip items that fail to cleanse"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of dataset shards for distributed runs",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to cleanse when sharding",
    )
    parser.add_argument(
        "--check-safe-load",
        type=bool,
        default=False,
        help="Whether to check valid audio or not"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    text_column = args.text_column if args.text_column else None

    dataset = load_dataset(
        args.hf_dataset,
        args.hf_name,
        split=args.hf_split,
        cache_dir=args.hf_cache_dir,
        streaming=False,
        trust_remote_code=True,
    )
    dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index)

    if args.check_safe_load:
        dataset = dataset.cast_column(args.audio_column, Audio(decode=False))
        def safe_load(example):
            path = example[args.audio_column]["path"]
            try:
                torchaudio.info(path)
                return True
            except Exception:
                return False
        dataset = dataset.filter(safe_load,num_proc=32)
        dataset = dataset.cast_column(args.audio_column, Audio(decode=True))

    total_samples = len(dataset)
    if total_samples == 0:
        print("Dataset slice is empty; nothing to cleanse.")
        return

    output_pattern = Path(args.output_pattern).expanduser()
    total_batches = math.ceil(total_samples / args.batch_size)

    if args.num_shards > 1:
        output_pattern = (
            output_pattern.parent
            / f"{args.shard_index:05d}-of-{args.num_shards:05d}"
            / output_pattern.name
        )

    output_pattern.parent.mkdir(parents=True, exist_ok=True)
    if (output_pattern.parent / f"completed_{args.hf_split}.txt").exists():
        quit()

    feature_extractor_path = resolve_checkpoint(
        args.feature_extractor,
        args.checkpoint_repo,
        args.feature_extractor_filename,
        token=args.hf_token,
    )
    decoder_path = resolve_checkpoint(
        args.decoder,
        args.checkpoint_repo,
        args.decoder_filename,
        token=args.hf_token,
    )

    processor = SpeechDenoisingProcessor(
        feature_extractor_path=feature_extractor_path,
        decoder_path=decoder_path,
        device=args.device,
        chunk_seconds=args.chunk_seconds,
        target_sample_rate=args.target_sample_rate,
    )

    collate_fn = CollateFunction(
        metadata_columns=dataset.column_names,
        skip_errors=args.skip_errors,
        audio_column=args.audio_column,
        key_column=args.key_column,
    )

    processed = 0
    attempted = 0
    progress = tqdm(total=total_samples)

    writer = AsyncShardWriter(
        output_pattern=output_pattern,
        samples_per_shard=args.samples_per_shard,
        target_sample_rate=args.target_sample_rate,
        text_column=text_column,
        max_queue_size=max(args.batch_size, 1) * 8,
    )
    writer_errors: list[tuple[str | None, str]] = []
    abort_due_to_error = False

    try:
        for batch in iterate_batches(
            dataset,
            args.batch_size,
            loader_workers=args.loader_workers,
            collate_fn=collate_fn,
        ):
            attempted += batch.size
            try:
                cleaned_waveforms = processor.process_batch(
                    batch.waveforms,
                    expected_lengths=[
                        round(d * processor.target_sample_rate) for d in batch.durations
                    ],
                )
            except Exception as exc:  # pylint: disable=broad-except
                if not args.skip_errors:
                    raise
                progress.write(
                    f"Batch starting with {batch.keys[0]} failed to cleanse: {exc}",
                    file=progress.fp,
                )
                progress.update(batch.size)
                continue

            for key, cleaned, metadata in zip(
                batch.keys, cleaned_waveforms, batch.metadata
            ):
                writer.submit(key, cleaned, metadata)

            drained, errors = writer.poll()
            if drained:
                processed += drained
            if errors:
                writer_errors.extend(errors)
                for error_key, error_msg in errors:
                    message_key = error_key or "unknown key"
                    if args.skip_errors:
                        progress.write(
                            f"Record {message_key} failed during serialisation: {error_msg}",
                            file=progress.fp,
                        )
                    else:
                        abort_due_to_error = True
                if abort_due_to_error:
                    progress.update(batch.size)
                    break

            progress.update(batch.size)
    finally:
        drained, errors = writer.close()
        if drained:
            processed += drained
        if errors:
            writer_errors.extend(errors)
            for error_key, error_msg in errors:
                message_key = error_key or "unknown key"
                if args.skip_errors:
                    progress.write(
                        f"Record {message_key} failed during serialisation: {error_msg}",
                        file=progress.fp,
                    )

    progress.close()

    if writer_errors and not args.skip_errors:
        first_key, first_msg = writer_errors[0]
        record_id = first_key or "unknown key"
        raise RuntimeError(f"Failed to write record {record_id}: {first_msg}")

    if writer_errors and args.skip_errors:
        progress.write(
            f"Skipped {len(writer_errors)} records during serialisation.",
            file=progress.fp,
        )

    progress.write(
        f"Cleansed {processed} of {attempted} samples into {output_pattern}",
        file=progress.fp,
    )
    (output_pattern.parent / f"completed_{args.hf_split}.txt").touch()


if __name__ == "__main__":
    main()
