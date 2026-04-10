#!/usr/bin/env python3

"""
Compute keyframe latents for keyframe-conditioned video generation training.
Processes keyframe images and saves them as latent representations with frame index
metadata, matching the inference-time VideoConditionByKeyframeIndex format.

IMPORTANT: This script must be run AFTER processing video latents (process_dataset.py),
as it reads the target resolution from the video latent files.

Usage:
    python scripts/process_keyframes.py dataset.csv \
        --latents-dir /path/to/preprocessed/latents \
        --model-path /path/to/ltx2.safetensors \
        --output-dir /path/to/preprocessed/keyframes
"""

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from einops import rearrange
from pillow_heif import register_heif_opener
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import Dataset
from transformers.utils.logging import disable_progress_bar

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder
from ltx_trainer.utils import open_image_as_srgb
from process_videos import encode_video

disable_progress_bar()
register_heif_opener()

# VAE spatial compression ratio (pixel = latent * 32)
VAE_SPATIAL_FACTOR = 32

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Process keyframe images and save latent representations for keyframe-conditioned training.",
)


def parse_keyframe_paths(keyframe_str: str) -> list[tuple[int, str]]:
    """Parse keyframe string into list of (frame_idx, path) tuples.
    Format: "0:path/to/frame0.png;60:path/to/frame60.png;120:path/to/frame120.png"
    Args:
        keyframe_str: Semicolon-separated keyframe specifications
    Returns:
        List of (frame_idx, path) tuples
    """
    if not keyframe_str or not isinstance(keyframe_str, str):
        return []

    keyframes = []
    for part in keyframe_str.strip().split(";"):
        part = part.strip()
        if not part:
            continue

        match = re.match(r"^(\d+):(.+)$", part)
        if not match:
            raise ValueError(f"Invalid keyframe format: '{part}'. Expected 'frame_idx:path'")

        keyframes.append((int(match.group(1)), match.group(2).strip()))

    return keyframes


def resize_and_center_crop(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize tensor preserving aspect ratio (filling target), then center crop.
    This matches the inference code's resize_and_center_crop in media_io.py.
    Args:
        tensor: Input tensor with shape (H, W, C) or (F, H, W, C)
        height: Target height
        width: Target width
    Returns:
        Tensor with shape (1, C, F, height, width)
    """
    if tensor.ndim == 3:
        tensor = rearrange(tensor, "h w c -> 1 c h w")
    elif tensor.ndim == 4:
        tensor = rearrange(tensor, "f h w c -> f c h w")
    else:
        raise ValueError(f"Expected 3 or 4 dimensions; got shape {tensor.shape}")

    _, _, src_h, src_w = tensor.shape
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)

    tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    crop_top = (new_h - height) // 2
    crop_left = (new_w - width) // 2
    tensor = tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width]

    return rearrange(tensor, "f c h w -> 1 c f h w")


def load_and_preprocess_keyframe(
    image_path: str,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Load a keyframe image and preprocess it for VAE encoding.
    Mimics the inference pipeline: load -> resize_and_center_crop -> normalize to [-1, 1].
    Args:
        image_path: Path to image file
        height: Target pixel height
        width: Target pixel width
        dtype: Target dtype
        device: Target device
    Returns:
        Tensor of shape (1, C, 1, height, width) ready for VAE encoding
    """
    from PIL import Image

    image = Image.open(image_path)
    np_array = np.array(image)[..., :3]  # Remove alpha channel if present
    tensor = torch.tensor(np_array, dtype=torch.float32, device=device)

    # Resize and center crop (same as inference)
    tensor = resize_and_center_crop(tensor, height, width)

    # Normalize to [-1, 1]
    return (tensor / 127.5 - 1.0).to(dtype=dtype)


class KeyframeDataset(Dataset):
    """Dataset for processing keyframe images with their target frame indices.
    Reads video latents to get exact target resolution for each sample.
    """

    def __init__(
        self,
        dataset_file: str | Path,
        keyframe_column: str,
        latents_dir: str | Path,
    ) -> None:
        """Initialize the keyframe dataset.
        Args:
            dataset_file: Path to CSV/JSON/JSONL metadata file
            keyframe_column: Column name containing keyframe specs ("0:path1;60:path2;...")
            latents_dir: Directory containing preprocessed video latents (*.pt files)
        """
        super().__init__()
        self.dataset_file = Path(dataset_file)
        self.latents_dir = Path(latents_dir)
        self.data_root = self.dataset_file.parent
        self.samples = self._load_dataset(keyframe_column)
        logger.info(f"Loaded {len(self.samples)} samples with valid keyframes and matching video latents")

    def _load_dataset(self, column: str) -> list[dict[str, Any]]:
        """Load and parse the dataset file."""
        suffix = self.dataset_file.suffix
        if suffix == ".csv":
            entries = self._read_csv(column)
        elif suffix == ".json":
            entries = self._read_json(column)
        elif suffix == ".jsonl":
            entries = self._read_jsonl(column)
        else:
            raise ValueError("Dataset file must be CSV, JSON, or JSONL format.")
        return entries

    def _read_csv(self, column: str) -> list[dict[str, Any]]:
        """Read entries from CSV."""
        df = pd.read_csv(self.dataset_file)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV. Available: {df.columns.tolist()}")

        video_col = "video_path" if "video_path" in df.columns else "video"
        if video_col not in df.columns:
            raise ValueError(f"CSV must contain 'video_path' or 'video' column. Available: {df.columns.tolist()}")

        entries = []
        for _, row in df.iterrows():
            entry = self._process_entry(str(row[video_col]), str(row[column]))
            if entry is not None:
                entries.append(entry)
        return entries

    def _read_json(self, column: str) -> list[dict[str, Any]]:
        """Read entries from JSON."""
        with open(self.dataset_file, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")

        entries = []
        for item in data:
            if "video_path" not in item or column not in item:
                continue
            entry = self._process_entry(item["video_path"], item[column])
            if entry is not None:
                entries.append(entry)
        return entries

    def _read_jsonl(self, column: str) -> list[dict[str, Any]]:
        """Read entries from JSONL."""
        entries = []
        with open(self.dataset_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "video_path" not in item or column not in item:
                    continue
                entry = self._process_entry(item["video_path"], item[column])
                if entry is not None:
                    entries.append(entry)
        return entries

    def _process_entry(self, video_path: str, keyframe_spec: str) -> dict[str, Any] | None:
        """Process a single entry: match with video latent and validate keyframes.
        Args:
            video_path: Path to the source video
            keyframe_spec: Keyframe specification string ("0:path1;60:path2;...")
        Returns:
            Sample dict or None if invalid
        """
        # Find corresponding video latent
        latent_path = self.latents_dir / f"{Path(video_path).stem}.pt"
        if not latent_path.is_file():
            logger.warning(f"No video latent found for: {video_path}")
            return None

        # Read resolution from video latent (latent space dimensions)
        try:
            latent_data = torch.load(latent_path, map_location="cpu", weights_only=False)
            latent_height = latent_data["height"]
            latent_width = latent_data["width"]
        except Exception as e:
            logger.warning(f"Failed to load video latent for {video_path}: {e}")
            return None

        # Convert latent dimensions to pixel space for VAE encoding
        pixel_height = latent_height * VAE_SPATIAL_FACTOR
        pixel_width = latent_width * VAE_SPATIAL_FACTOR

        # Parse and validate keyframe paths
        keyframe_entries = parse_keyframe_paths(keyframe_spec)
        if not keyframe_entries:
            return None

        valid_keyframes = []
        for frame_idx, rel_path in keyframe_entries:
            full_path = self.data_root / rel_path
            if not full_path.is_file():
                logger.warning(f"Keyframe image not found: {full_path}")
                continue
            valid_keyframes.append({
                "frame_idx": frame_idx,
                "path": full_path,
                "relative_path": rel_path,
            })

        if not valid_keyframes:
            return None

        return {
            "video_path": video_path,
            "keyframes": valid_keyframes,
            "base_name": Path(video_path).stem,
            "target_height": pixel_height,
            "target_width": pixel_width,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


@torch.inference_mode()
def compute_keyframe_latents(
    dataset: KeyframeDataset,
    model_path: str,
    output_dir: str,
    device: str = "cuda",
    vae_tiling: bool = False,
) -> None:
    """Process keyframe images and save their latent representations.
    Args:
        dataset: KeyframeDataset with validated samples
        model_path: Path to LTX-2 checkpoint
        output_dir: Directory to save keyframe latents
        device: Device to use
        vae_tiling: Enable VAE tiling for large resolutions
    """
    console = Console()
    torch_device = torch.device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load VAE encoder
    with console.status(f"[bold]Loading VAE encoder from [cyan]{model_path}[/]...", spinner="dots"):
        vae = load_video_vae_encoder(model_path, device=torch_device, dtype=torch.bfloat16)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing keyframes", total=len(dataset))

        for sample in dataset:
            encoded_keyframes = []
            for kf_info in sample["keyframes"]:
                keyframe_tensor = load_and_preprocess_keyframe(
                    image_path=str(kf_info["path"]),
                    height=sample["target_height"],
                    width=sample["target_width"],
                    dtype=torch.bfloat16,
                    device=torch_device,
                )

                latent_data = encode_video(vae=vae, video=keyframe_tensor, use_tiling=vae_tiling)

                encoded_keyframes.append({
                    "latent": latent_data["latents"][0].cpu().contiguous(),  # [C, F', H', W']
                    "frame_idx": kf_info["frame_idx"],
                    "path": kf_info["relative_path"],
                })

            output_file = output_path / f"{sample['base_name']}.pt"
            torch.save(
                {
                    "keyframes": encoded_keyframes,
                    "num_keyframes": len(encoded_keyframes),
                    "height": sample["target_height"] // VAE_SPATIAL_FACTOR,
                    "width": sample["target_width"] // VAE_SPATIAL_FACTOR,
                },
                output_file,
            )
            progress.advance(task)

    logger.info(f"Processed {len(dataset)} samples. Keyframe latents saved to {output_path}")


@app.command()
def main(
    dataset_path: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) with keyframe specifications",
    ),
    keyframe_column: str = typer.Option(
        default="keyframe_paths",
        help="Column name containing keyframe specifications ('0:path1;60:path2;...')",
    ),
    latents_dir: str = typer.Option(
        ...,
        help="Directory containing preprocessed video latents (to get target resolution)",
    ),
    model_path: str = typer.Option(
        ...,
        help="Path to LTX-2 checkpoint (.safetensors)",
    ),
    output_dir: str = typer.Option(
        ...,
        help="Output directory for keyframe latents",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for large resolutions",
    ),
) -> None:
    """
    Preprocess keyframe images for keyframe-conditioned video generation training.

    The dataset file must contain:
    - 'video_path' (or 'video') column: path to the video file (used to find corresponding latent)
    - Keyframe column with format: "0:path/to/frame0.png;60:path/to/frame60.png;..."

    Example CSV:
        video_path,keyframe_paths,caption
        videos/clip1.mp4,"0:keyframes/f0.png;120:keyframes/f120.png","A scene..."

    Example workflow:
        # Step 1: Process video latents first
        python scripts/process_dataset.py dataset.csv \\
            --resolution-buckets "544x960x121" \\
            --model-path /path/to/ltx2.safetensors \\
            --text-encoder-path /path/to/gemma \\
            --output-dir /path/to/preprocessed

        # Step 2: Process keyframes (reads resolution from video latents)
        python scripts/process_keyframes.py dataset.csv \\
            --keyframe-column keyframe_paths \\
            --latents-dir /path/to/preprocessed/latents \\
            --model-path /path/to/ltx2.safetensors \\
            --output-dir /path/to/preprocessed/keyframes
    """
    if not Path(latents_dir).exists():
        raise typer.BadParameter(
            f"Latents directory not found: {latents_dir}. "
            "Please run process_dataset.py first to generate video latents."
        )

    dataset = KeyframeDataset(
        dataset_file=dataset_path,
        keyframe_column=keyframe_column,
        latents_dir=latents_dir,
    )

    compute_keyframe_latents(
        dataset=dataset,
        model_path=model_path,
        output_dir=output_dir,
        device=device,
        vae_tiling=vae_tiling,
    )


if __name__ == "__main__":
    app()
