#!/usr/bin/env python3
"""Example driver for cleansing Hugging Face speech datasets with Sidon (defaults to facebook/multilingual_librispeech)."""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchaudio
from datasets import Audio, Dataset, Features, load_dataset
from typing import TYPE_CHECKING, cast
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from scripts.cleanse_webdataset import SpeechDenoisingProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-extractor",
        default=None,
        help="Path to feature_extractor_cuda.pt TorchScript bundle (defaults to downloading from Hugging Face)",
    )
    parser.add_argument(
        "--decoder",
        default=None,
        help="Path to decoder_cuda.pt TorchScript bundle (defaults to downloading from Hugging Face)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where cleansed shards will be stored",
    )
    parser.add_argument(
        "--hf-dataset",
        default="facebook/multilingual_librispeech",
        help="Hugging Face dataset identifier passed to datasets.load_dataset",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Dataset configuration to cleanse (e.g. english, german, french)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to cleanse (train, validation, test, 1_hours, etc.)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Computation device passed to the denoiser",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of samples processed per batch",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=4,
        help="Thread workers to parallelise sample decoding (CLI path only)",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=5000,
        help="Maximum samples per output shard",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="Chunk duration passed to the denoiser",
    )
    parser.add_argument(
        "--highpass-cutoff",
        type=float,
        default=50.0,
        help="High-pass cutoff prior to resampling",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=48_000,
        help="Sample rate written to cleansed audio",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total worker count when launching multiple CLI instances",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of this worker (0-based)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Logging frequency forwarded to the cleansing script",
    )
    parser.add_argument(
        "--default-sample-rate",
        type=int,
        default=None,
        help="Optional fallback sample rate if the dataset entry omits one",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue when an item fails to cleanse",
    )
    parser.add_argument(
        "--hf-streaming",
        action="store_true",
        help="Enable datasets streaming mode when using the CLI path",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Cache directory forwarded to datasets.load_dataset",
    )
    parser.add_argument(
        "--hf-audio-column",
        default="audio",
        help="Audio column name in the source dataset",
    )
    parser.add_argument(
        "--hf-key-column",
        default="id",
        help="Column used as the unique sample identifier",
    )
    parser.add_argument(
        "--hf-output-dir",
        default=None,
        help="If set, write a Hugging Face dataset with cleansed audio to this path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the constructed command (or intended HF export) without executing",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to cleanse_webdataset.py",
    )
    parser.add_argument(
        "--checkpoint-repo",
        default="sarulab-speech/sidon-v0.1",
        help="Hugging Face repository providing Sidon checkpoints",
    )
    parser.add_argument(
        "--feature-extractor-filename",
        default="feature_extractor_cuda.pt",
        help="Filename of the feature extractor within the checkpoint repository",
    )
    parser.add_argument(
        "--decoder-filename",
        default="decoder_cuda.pt",
        help="Filename of the decoder within the checkpoint repository",
    )
    return parser


def load_processor(args, feature_extractor_path: str, decoder_path: str):
    project_root = Path(__file__).resolve().parents[2]
    scripts_root = project_root / "scripts"
    for path in (project_root, scripts_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)
    from scripts.cleanse_webdataset import SpeechDenoisingProcessor

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return SpeechDenoisingProcessor(
        feature_extractor_path=feature_extractor_path,
        decoder_path=decoder_path,
        device=device,
        chunk_seconds=args.chunk_seconds,
        highpass_cutoff=args.highpass_cutoff,
        target_sample_rate=args.target_sample_rate,
    )


def collapse_waveform(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.to(dtype=torch.float32).cpu()
    if tensor.ndim > 1:
        tensor = tensor.mean(dim=0)
    return tensor


def decode_audio_entry(
    audio_entry,
    *,
    audio_feature: Audio | None,
    sample_key: str | None,
    default_sample_rate: int | None,
    feature_sampling_rate: int | None,
    skip_errors: bool,
):
    sample_label = sample_key or "[unknown]"
    entry = audio_entry


    array = entry["array"]
    sample_rate = entry["sampling_rate"]

    waveform_tensor = torch.as_tensor(array, dtype=torch.float32)

    sample_rate = (
        sample_rate
        if sample_rate is not None
        else (
            default_sample_rate if default_sample_rate is not None else feature_sampling_rate
        )
    )

    waveform = collapse_waveform(waveform_tensor)

    return waveform, int(sample_rate)


def _cleaned_generator(
    *,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    hf_cache_dir: str | None,
    other_columns: list[str],
    key_col: str,
    audio_col: str,
    processor_config: dict[str, object],
    feature_extractor_path: str,
    decoder_path: str,
    batch_size: int,
    target_sample_rate: int,
    default_sample_rate: int | None,
    skip_errors: bool,
    progress_desc: str,
    num_shards: int = 1,
    shard_index: int = 0,
):
    processor_args = SimpleNamespace(**processor_config)
    processor = load_processor(processor_args, feature_extractor_path, decoder_path)

    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=dataset_split,
        streaming=False,
        cache_dir=hf_cache_dir,
    )

    if num_shards > 1:
        dataset = dataset.shard(num_shards=num_shards, index=shard_index)

    audio_feature = dataset.features.get(audio_col)
    feature_sampling_rate = (
        audio_feature.sampling_rate if isinstance(audio_feature, Audio) else None
    )

    def build_record(metadata: dict[str, object], cleaned: torch.Tensor) -> dict[str, object]:
        record = dict(metadata)
        waveform = collapse_waveform(cleaned)
        record[audio_col] = {
            "array": waveform.numpy().astype(np.float32),
            "sampling_rate": target_sample_rate,
        }
        return record

    batch_waveforms: list[torch.Tensor] = []
    batch_sample_rates: list[int] = []
    batch_metadata: list[tuple[str | None, dict[str, object]]] = []

    progress = tqdm(dataset, desc=progress_desc, unit="sample")

    def flush_batch():
        nonlocal batch_waveforms, batch_sample_rates, batch_metadata
        if not batch_waveforms:
            return
        try:
            cleaned_batch = processor.process_batch(batch_waveforms, batch_sample_rates)
        except Exception as exc:
            if not skip_errors:
                raise
            print(
                f"Failed batch of {len(batch_waveforms)} samples: {exc}; retrying individually"
            )
            for (sample_key, metadata), waveform, sample_rate in zip(
                batch_metadata, batch_waveforms, batch_sample_rates
            ):
                try:
                    cleaned = processor.process(waveform, sample_rate)
                except Exception as single_exc:
                    print(
                        f"  Skipping sample {sample_key or '[unknown]'} due to error: {single_exc}"
                    )
                    continue
                yield build_record(metadata, cleaned)
        else:
            for (_, metadata), cleaned in zip(batch_metadata, cleaned_batch):
                yield build_record(metadata, cleaned)
        finally:
            batch_waveforms = []
            batch_sample_rates = []
            batch_metadata = []

    for sample in progress:
        metadata = {col: sample[col] for col in other_columns}
        sample_key = metadata.get(key_col) if key_col in metadata else sample.get(key_col)

        decoded = decode_audio_entry(
            sample[audio_col],
            audio_feature=cast(Audio | None, audio_feature),
            sample_key=sample_key,
            default_sample_rate=default_sample_rate,
            feature_sampling_rate=feature_sampling_rate,
            skip_errors=skip_errors,
        )
        if decoded is None:
            continue
        waveform, sample_rate = decoded

        batch_waveforms.append(waveform)
        batch_sample_rates.append(int(sample_rate))
        batch_metadata.append((sample_key, metadata))

        if len(batch_waveforms) == batch_size:
            yield from flush_batch()

    yield from flush_batch()


def resolve_checkpoint(path: str | None, repo: str, filename: str) -> str:
    return path or hf_hub_download(repo, filename=filename)


def build_cli_command(
    args,
    *,
    cleanse_script: Path,
    feature_extractor_path: str,
    decoder_path: str,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(cleanse_script),
        "--hf-dataset",
        args.hf_dataset,
    ]
    if args.language is not None:
        command.extend([
            "--hf-config",
            args.language,
        ])
    command.extend([
        "--hf-split",
        args.split,
        "--hf-key-column",
        args.hf_key_column,
        "--hf-audio-column",
        args.hf_audio_column,
        "--output-dir",
        str(output_dir),
        "--feature-extractor",
        feature_extractor_path,
        "--decoder",
        decoder_path,
        "--batch-size",
        str(args.batch_size),
        "--samples-per-shard",
        str(args.samples_per_shard),
        "--device",
        args.device,
        "--loader-workers",
        str(args.loader_workers),
        "--chunk-seconds",
        str(args.chunk_seconds),
        "--highpass-cutoff",
        str(args.highpass_cutoff),
        "--target-sample-rate",
        str(args.target_sample_rate),
        "--num-workers",
        str(args.num_workers),
        "--shard-index",
        str(args.shard_index),
        "--log-every",
        str(args.log_every),
    ])
    if args.default_sample_rate is not None:
        command.extend(["--default-sample-rate", str(args.default_sample_rate)])
    if args.skip_errors:
        command.append("--skip-errors")
    if args.hf_streaming:
        command.append("--hf-streaming")
    if args.extra_args:
        command.extend(args.extra_args)
    return command


def export_hf_dataset(
    args,
    feature_extractor_path: str,
    decoder_path: str,
) -> None:
    dataset = load_dataset(
        args.hf_dataset,
        name=args.language,
        split=args.split,
        streaming=False,
        cache_dir=args.hf_cache_dir,
    )
    audio_col = args.hf_audio_column
    key_col = args.hf_key_column
    other_columns = [col for col in dataset.column_names if col != audio_col]

    features = dataset.features.copy()
    generator_features = Features(
        {col: features[col] for col in other_columns if col in features}
    )
    generator_features[audio_col] = Audio(sampling_rate=args.target_sample_rate)

    del dataset
    processor_config = {
        "device": args.device,
        "chunk_seconds": args.chunk_seconds,
        "highpass_cutoff": args.highpass_cutoff,
        "target_sample_rate": args.target_sample_rate,
    }

    progress_desc = "Cleansing"
    if args.num_workers > 1:
        progress_desc = f"Cleansing (worker {args.shard_index + 1}/{args.num_workers})"

    processed_dataset = Dataset.from_generator(
        _cleaned_generator,
        features=generator_features,
        cache_dir=args.hf_cache_dir,
        gen_kwargs={
            "dataset_name": args.hf_dataset,
            "dataset_config": args.language,
            "dataset_split": args.split,
            "hf_cache_dir": args.hf_cache_dir,
            "other_columns": other_columns,
            "key_col": key_col,
            "audio_col": audio_col,
            "processor_config": processor_config,
            "feature_extractor_path": feature_extractor_path,
            "decoder_path": decoder_path,
            "batch_size": args.batch_size,
            "target_sample_rate": args.target_sample_rate,
            "default_sample_rate": args.default_sample_rate,
            "skip_errors": args.skip_errors,
            "progress_desc": progress_desc,
            "num_shards": args.num_workers,
            "shard_index": args.shard_index,
        },
    )

    if len(processed_dataset) == 0:
        print("No samples were cleansed; skipping dataset export.")
        return

    processed_dataset = processed_dataset.cast_column(
        audio_col, Audio(sampling_rate=args.target_sample_rate)
    )

    output_dir = Path(args.hf_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dataset.save_to_disk(str(output_dir))
    print(f"Saved cleansed dataset to {output_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cleanse_script = project_root / "cleanse_webdataset.py"
    if not cleanse_script.exists():
        raise FileNotFoundError(
            f"Unable to locate cleanse_webdataset.py next to this script (expected {cleanse_script})"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()

    feature_extractor_path = resolve_checkpoint(
        args.feature_extractor,
        args.checkpoint_repo,
        args.feature_extractor_filename,
    )
    decoder_path = resolve_checkpoint(
        args.decoder,
        args.checkpoint_repo,
        args.decoder_filename,
    )

    if args.hf_output_dir:
        if args.dry_run:
            dataset_desc = args.hf_dataset if args.language is None else f"{args.hf_dataset}/{args.language}"
            print(
                f"Would cleanse dataset '{dataset_desc}:{args.split}' and save to "
                f"{Path(args.hf_output_dir).expanduser()}"
            )
            return
        if args.hf_streaming:
            parser.error("--hf-output-dir cannot be used together with --hf-streaming")
        export_hf_dataset(args, feature_extractor_path, decoder_path)
        return

    command = build_cli_command(
        args,
        cleanse_script=cleanse_script,
        feature_extractor_path=feature_extractor_path,
        decoder_path=decoder_path,
        output_dir=output_dir,
    )

    if args.dry_run:
        print(" \\\n".join(command))
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
