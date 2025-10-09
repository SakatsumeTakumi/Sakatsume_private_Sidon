"""Cleanse WebDataset shards using Sidon's TorchScript models."""

from __future__ import annotations

import argparse
import io
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import torch
import torchaudio
import webdataset as wds
from tqdm.auto import tqdm

from sidon.cleansing import extract_seamless_m4t_features
from sidon.data.datamodule import torch_audio

AUDIO_EXTENSIONS = (".tar", ".tar.gz")


def collect_shards(inputs: Sequence[str]) -> List[str]:
    """Materialise shard paths from files, directories, or manifests."""
    shards: list[str] = []
    for raw_entry in inputs:
        entry = Path(raw_entry).expanduser()
        if entry.is_file():
            if entry.name.endswith(AUDIO_EXTENSIONS):
                shards.append(str(entry.resolve()))
                continue
            if entry.suffix == ".txt":
                with entry.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith(("s3://", "pipe:")):
                            shards.append(line)
                            continue
                        nested = Path(line)
                        if not nested.is_absolute():
                            nested = (entry.parent / nested).resolve()
                        shards.extend(collect_shards([str(nested)]))
                continue
            raise ValueError(f"Unsupported file type: {entry}")
        if entry.is_dir():
            for suffix in ("**/*.tar", "**/*.tar.gz"):
                for path in sorted(entry.glob(suffix)):
                    shards.append(str(path.resolve()))
            continue
        if raw_entry.startswith(("s3://", "pipe:")):
            shards.append(raw_entry)
            continue
        raise FileNotFoundError(f"Input not found: {raw_entry}")
    seen: set[str] = set()
    unique_shards: list[str] = []
    for shard in shards:
        if shard not in seen:
            seen.add(shard)
            unique_shards.append(shard)
    return unique_shards


def select_shards(
    shards: Sequence[str], num_shards: int, shard_index: int
) -> List[str]:
    """Pick the subset of shards assigned to a worker."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if num_shards == 1:
        return list(shards)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be within [0, num_shards)")
    total = len(shards)
    if total == 0:
        return []
    block = math.ceil(total / num_shards)
    start = block * shard_index
    end = min(start + block, total)
    if start >= end:
        return []
    return list(shards[start:end])


class SpeechDenoisingProcessor:
    """Run the TorchScripted feature extractor and decoder."""

    def __init__(
        self,
        feature_extractor_path: str,
        decoder_path: str,
        device: str,
        chunk_seconds: float,
        target_sample_rate: int,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.feature_extractor = torch.jit.load(
            feature_extractor_path, map_location=self.device
        )
        self.decoder = torch.jit.load(decoder_path, map_location=self.device)
        self.feature_extractor.eval()
        self.decoder.eval()
        self.chunk_samples = max(int(chunk_seconds * 16_000), 1_600)
        self.target_sample_rate = target_sample_rate

    def _prepare_waveform_batch(
        self,
        waveforms: Sequence[torch.Tensor] | torch.Tensor,
        sample_rates: Sequence[int],
    ) -> list[torch.Tensor]:
        waveforms_padded = torch.zeros(
            len(waveforms), self.chunk_samples, dtype=torch.float32
        )
        # warn if the waveforms are longer than chunk samples
        if waveforms.size(1) > self.chunk_samples:
            warnings.warn(
                f"sample truncated to {self.chunk_samples} samples"
            )
        if isinstance(waveforms, torch.Tensor):
            waveforms_padded[:, : waveforms.shape[-1]] = waveforms[:,:self.chunk_samples] # truncate samples if exceeding
        else:
            for idx, (waveform, sample_rate) in enumerate(zip(waveforms, sample_rates)):
                waveforms_padded[idx, : waveform.shape[-1]] = waveform

        return waveforms_padded

    def _prepare_waveform(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        return self._prepare_waveform_batch([waveform], [sample_rate])[0]

    @torch.inference_mode()
    def process(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return self.process_batch([waveform], [sample_rate])[0]

    @torch.inference_mode()
    def process_batch(
        self,
        waveforms: Sequence[torch.Tensor] | torch.Tensor,
        sample_rates: Optional[Sequence[int]] = None,
        expected_lengths: Optional[Sequence[int]] = None,
    ) -> List[torch.Tensor]:
        if expected_lengths is None:
            expected_lengths: list[int] = []
            for waveform, sample_rate in zip(waveforms, sample_rates):
                duration_seconds = waveform.shape[-1] / float(sample_rate)
                expected_lengths.append(
                    int(round(duration_seconds * self.target_sample_rate))
                )
        prepared_waveforms = self._prepare_waveform_batch(waveforms, sample_rates)

        start_time = time.perf_counter()
        features = extract_seamless_m4t_features(
            [x for x in prepared_waveforms],
            return_tensors="pt",
            padding_value=1.0,
            device=str(self.device),
        )
        feature_tensor = self.feature_extractor(
            features["input_features"].to(self.device)
        )["last_hidden_state"]
        restored_waveforms = self.decoder(feature_tensor.transpose(1, 2)).cpu()
        end_time = time.perf_counter()

        results: List[torch.Tensor] = []
        for sample_idx, sample in enumerate(restored_waveforms):
            restored_waveform = sample.view(-1)
            target_length = expected_lengths[sample_idx]
            current_length = restored_waveform.shape[-1]
            if target_length > 0 and current_length != target_length:
                diff = target_length - current_length
                if diff > 0:
                    restored_waveform = torch.nn.functional.pad(
                        restored_waveform, (0, diff)
                    )
                elif diff < 0:
                    restored_waveform = restored_waveform[:target_length]
            results.append(restored_waveform.contiguous())

        return results


def load_waveform(
    sample: dict,
    audio_key: str,
    default_sample_rate: int | None,
) -> tuple[torch.Tensor, int]:
    if audio_key not in sample:
        raise KeyError(f"Sample missing key '{audio_key}'")
    value = sample[audio_key]
    if isinstance(value, tuple) and len(value) == 2:
        waveform, sample_rate = value
    else:
        waveform, sample_rate = value, default_sample_rate
    if sample_rate is None:
        raise ValueError("Sample rate not provided; set --default-sample-rate")
    if isinstance(sample_rate, torch.Tensor):
        sample_rate = int(sample_rate.item())
    sample_rate = int(sample_rate)
    waveform = torch.as_tensor(waveform)
    return waveform, sample_rate


def serialise_flac(key: str, waveform: torch.Tensor, sample_rate: int) -> dict:
    buffer = io.BytesIO()
    audio = waveform.to(dtype=torch.float32).cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(buffer, audio, sample_rate, format="flac")
    return {"__key__": key, "flac": buffer.getvalue()}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Shard files, directories, or manifest txt files to cleanse",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where cleaned shards will be written",
    )
    parser.add_argument(
        "--feature-extractor",
        required=True,
        help="Path to the TorchScript feature extractor (feature_extractor_cuda.pt)",
    )
    parser.add_argument(
        "--decoder",
        required=True,
        help="Path to the TorchScript decoder (decoder_cuda.pt)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model inference (cuda, cpu, or auto)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="Chunk size (in seconds) fed into the feature extractor",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=48_000,
        help="Sample rate metadata stored in the output shards",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=5_000,
        help="Number of items per output shard",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples to process together",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="Thread workers used to decode samples within each batch",
    )
    parser.add_argument(
        "--audio-key",
        default="flac",
        help="Key containing the noisy waveform in the source dataset",
    )
    parser.add_argument(
        "--default-sample-rate",
        type=int,
        default=None,
        help="Fallback sample rate when the dataset entry omits it",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of parallel cleansing workers",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of this worker (0-indexed)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print progress every N processed samples",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue when an item fails to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs even if completed.txt is present",
    )
    parser.add_argument(
        "--shard-pattern",
        default="dataset-%06d.tar.gz",
        help="Filename pattern for output shards",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Optional Hugging Face dataset name (e.g. 'mozilla-foundation/common_voice_17_0')",
    )
    parser.add_argument(
        "--hf-config",
        default=None,
        help="Hugging Face dataset config name",
    )
    parser.add_argument(
        "--hf-split",
        default="train",
        help="Dataset split to cleanse when using Hugging Face datasets",
    )
    parser.add_argument(
        "--hf-audio-column",
        default="audio",
        help="Column containing audio entries for Hugging Face datasets",
    )
    parser.add_argument(
        "--hf-sample-rate-column",
        default=None,
        help="Column providing sample rate for Hugging Face datasets when not embedded in the audio column",
    )
    parser.add_argument(
        "--hf-key-column",
        default=None,
        help="Optional column used as the __key__ for Hugging Face datasets",
    )
    parser.add_argument(
        "--hf-streaming",
        action="store_true",
        help="Use streaming mode when loading Hugging Face datasets",
    )
    return parser


def iterate_in_batches(
    iterator: Iterable[dict], batch_size: int
) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for sample in iterator:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_hf_iterator(args: argparse.Namespace) -> tuple[Iterable[dict], Optional[int]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Install the 'datasets' package to use Hugging Face dataset support"
        ) from exc

    dataset = load_dataset(
        args.hf_dataset,
        name=args.hf_config,
        split=args.hf_split,
        streaming=args.hf_streaming,
    )
    if args.num_workers > 1:
        dataset = dataset.shard(num_shards=args.num_workers, index=args.shard_index)

    audio_feature = None
    if hasattr(dataset, "features") and args.hf_audio_column in dataset.features:
        audio_feature = dataset.features[args.hf_audio_column]

    if not args.hf_streaming:
        try:
            length: Optional[int] = len(dataset)  # type: ignore[arg-type]
        except TypeError:
            length = None
    else:
        length = None

    def iterator() -> Iterator[dict]:
        for index, record in enumerate(dataset):
            raw_entry = record[args.hf_audio_column]

            array = raw_entry["array"]
            sample_rate = raw_entry["sampling_rate"]

            waveform = torch.as_tensor(array, dtype=torch.float32)
            key: Optional[str]
            if args.hf_key_column and args.hf_key_column in record:
                key = record[args.hf_key_column]
            else:
                key = f"{args.hf_split}-{index:09d}"

            yield {
                "__key__": str(key),
                args.audio_key: (waveform, int(sample_rate)),
            }

    return iterator(), length


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not args.hf_dataset and not args.inputs:
        parser.error("Provide input shards/directories or set --hf-dataset")

    dataset_length: Optional[int] = None
    if args.hf_dataset:
        if args.audio_key == "flac":
            args.audio_key = args.hf_audio_column
        source_iter, dataset_length = build_hf_iterator(args)
    else:
        shards = collect_shards(args.inputs)
        shards = select_shards(shards, args.num_workers, args.shard_index)
        if not shards:
            parser.error("No shards selected for processing")

        dataset = wds.WebDataset(
            shards,
            shardshuffle=False,
            nodesplitter=lambda urls: urls,
            handler=wds.warn_and_continue,
        ).decode(
            wds.autodecode.basichandlers,
            torch_audio,
            handler=wds.warn_and_continue,
        )
        source_iter = dataset

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel = output_dir / "completed.txt"
    if sentinel.exists() and not args.force:
        print(
            f"Output directory {output_dir} already processed; use --force to overwrite"
        )
        return

    processor = SpeechDenoisingProcessor(
        feature_extractor_path=args.feature_extractor,
        decoder_path=args.decoder,
        device=args.device,
        chunk_seconds=args.chunk_seconds,
        highpass_cutoff=args.highpass_cutoff,
        target_sample_rate=args.target_sample_rate,
    )

    shard_pattern = str((output_dir / args.shard_pattern).resolve())
    processed = 0
    total_batches: Optional[int] = None
    if dataset_length is not None:
        total_batches = (dataset_length + args.batch_size - 1) // args.batch_size

    batch_iterator = iterate_in_batches(source_iter, args.batch_size)
    executor: ThreadPoolExecutor | None = None
    if args.loader_workers > 1:
        executor = ThreadPoolExecutor(max_workers=args.loader_workers)

    def decode_sample(sample: dict) -> Optional[tuple[str, torch.Tensor, int]]:
        key = sample.get("__key__")
        if key is None:
            if args.skip_errors:
                return None
            raise KeyError("Sample missing __key__")
        try:
            waveform, sample_rate = load_waveform(
                sample,
                audio_key=args.audio_key,
                default_sample_rate=args.default_sample_rate,
            )
        except Exception as exc:  # pylint: disable=broad-except
            if args.skip_errors:
                print(f"Failed to read {key}: {exc}")
                return None
            raise
        return str(key), waveform, sample_rate

    with wds.ShardWriter(shard_pattern, maxcount=args.samples_per_shard) as sink:
        for batch in tqdm(
            batch_iterator,
            desc="Cleansing",
            unit="batch",
            total=total_batches,
        ):
            keys: list[str] = []
            waveforms: list[torch.Tensor] = []
            sample_rates: list[int] = []

            if executor is not None:
                decoded_iter = executor.map(decode_sample, batch, chunksize=1)
            else:
                decoded_iter = (decode_sample(sample) for sample in batch)

            for decoded in decoded_iter:
                if decoded is None:
                    continue
                key, waveform, sample_rate = decoded
                keys.append(key)
                waveforms.append(waveform)
                sample_rates.append(sample_rate)

            if not keys:
                continue

            try:
                restored_waveforms = processor.process_batch(waveforms, sample_rates)
            except Exception as exc:  # pylint: disable=broad-except
                if args.skip_errors:
                    print(f"Failed to process batch starting with {keys[0]}: {exc}")
                    continue
                raise

            for key, restored in zip(keys, restored_waveforms):
                record = serialise_flac(key, restored, args.target_sample_rate)
                sink.write(record)
            processed += len(restored_waveforms)
            if processed % args.log_every == 0:
                print(f"Processed {processed} samples")

    sentinel.touch()
    print(f"Finished cleansing {processed} samples into {output_dir}")

    if executor is not None:
        executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
