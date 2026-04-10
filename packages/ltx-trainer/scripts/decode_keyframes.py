#!/usr/bin/env python3

"""
Decode keyframe latents back to images for verification.
Allows visual inspection of the preprocessing pipeline (resize, crop, VAE encode/decode).

Usage:
    python scripts/decode_keyframes.py /path/to/keyframes \
        /path/to/ltx2.safetensors --output-dir /path/to/decoded
"""

from pathlib import Path

import torch
import typer
from PIL import Image
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
from transformers.utils.logging import disable_progress_bar

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_decoder

disable_progress_bar()
console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Decode keyframe latents back to images for verification.",
)


@torch.inference_mode()
def decode_keyframes(
    keyframes_dir: Path,
    model_path: str,
    output_dir: Path,
    device: str = "cuda",
) -> None:
    """Decode keyframe latents back to images.
    Args:
        keyframes_dir: Directory containing keyframe .pt files
        model_path: Path to LTX-2 checkpoint
        output_dir: Directory to save decoded images
        device: Device to use
    """
    torch_device = torch.device(device)

    with console.status(f"[bold]Loading VAE decoder from [cyan]{model_path}[/]...", spinner="dots"):
        vae = load_video_vae_decoder(model_path, device=torch_device, dtype=torch.bfloat16)

    keyframe_files = list(Path(keyframes_dir).rglob("*.pt"))
    if not keyframe_files:
        logger.warning(f"No .pt files found in {keyframes_dir}")
        return

    logger.info(f"Found {len(keyframe_files)} keyframe latent files")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
        task = progress.add_task("Decoding keyframes", total=len(keyframe_files))

        for kf_file in keyframe_files:
            try:
                kf_data = torch.load(kf_file, map_location="cpu", weights_only=False)

                if "keyframes" not in kf_data:
                    logger.warning(f"Skipping {kf_file.name}: not a keyframe latent file")
                    progress.advance(task)
                    continue

                for kf_entry in kf_data["keyframes"]:
                    latent = kf_entry["latent"]  # [C, F', H', W']
                    frame_idx = kf_entry["frame_idx"]

                    # Decode: [C, F', H', W'] -> [1, 3, 1, H, W]
                    decoded = vae(latent.unsqueeze(0).to(torch_device, dtype=torch.bfloat16))

                    # Convert to image: [1, 3, 1, H, W] -> [H, W, 3] uint8
                    decoded = decoded.squeeze(0).squeeze(1).permute(1, 2, 0)
                    decoded = ((decoded + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()

                    output_file = output_path / f"{kf_file.stem}_frame{frame_idx}.png"
                    Image.fromarray(decoded).save(output_file)


            except Exception as e:
                logger.error(f"Failed to decode {kf_file}: {e}")

            progress.advance(task)

    logger.info(f"Decoded keyframes saved to {output_path}")


@app.command()
def main(
    keyframes_dir: str = typer.Option(
        ...,
        help="Directory containing keyframe latent .pt files",
    ),
    model_path: str = typer.Option(
        ...,
        help="Path to LTX-2 checkpoint (.safetensors)",
    ),
    output_dir: str = typer.Option(
        ...,
        help="Output directory for decoded images",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use",
    ),
) -> None:
    """
    Decode keyframe latents back to images for verification.

    Example:
        python scripts/decode_keyframes.py /path/to/preprocessed/keyframes \\
            /path/to/ltx2.safetensors \\
            --output-dir /path/to/decoded_keyframes
    """
    decode_keyframes(
        keyframes_dir=Path(keyframes_dir),
        model_path=model_path,
        output_dir=Path(output_dir),
        device=device,
    )


if __name__ == "__main__":
    app()
