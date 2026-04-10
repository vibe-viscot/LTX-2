"""Text-to-video training strategy.
This strategy implements standard text-to-video generation training where:
- Only target latents are used (no reference videos)
- Standard noise application and loss computation
- Supports first frame conditioning
- Supports keyframe conditioning via token concatenation (matching inference behavior)
- Optionally supports joint audio-video training
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import VideoLatentShape
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    VIDEO_SCALE_FACTORS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class TextToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for text-to-video training strategy."""

    name: Literal["text_to_video"] = "text_to_video"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    keyframe_conditioning_p: float = Field(
        default=0.0,
        description="Probability of conditioning on keyframes during training. "
        "Frame indices are read from preprocessed keyframe data.",
        ge=0.0,
        le=1.0,
    )

    with_audio: bool = Field(
        default=False,
        description="Whether to include audio in training (joint audio-video generation)",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents when with_audio is True",
    )


class TextToVideoStrategy(TrainingStrategy):
    """Text-to-video training strategy.
    This strategy implements regular video generation training where:
    - Only target latents are used (no reference videos)
    - Standard noise application and loss computation
    - Supports first frame conditioning
    - Supports keyframe conditioning via concatenation (matching inference behavior)
    - Optionally supports joint audio-video training when with_audio=True
    """

    config: TextToVideoConfig

    def __init__(self, config: TextToVideoConfig):
        """Initialize strategy with configuration.
        Args:
            config: Text-to-video configuration
        """
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        """Whether this training strategy requires audio components."""
        return self.config.with_audio

    def get_data_sources(self) -> list[str] | dict[str, str]:
        """
        Text-to-video training requires latents and text conditions.
        When with_audio is True, also requires audio latents.
        When keyframe_conditioning_p > 0, also requires keyframes.
        """
        sources = {
            "latents": "latents",
            "conditions": "conditions",
        }

        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"

        if self.config.keyframe_conditioning_p > 0:
            sources["keyframes"] = "keyframes"

        return sources

    def prepare_training_inputs(  # noqa: PLR0915
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for text-to-video training."""
        # Get pre-encoded latents - dataset provides uniform non-patchified format [B, C, F, H, W]
        latents = batch["latents"]
        video_latents = latents["latents"]

        # Get video dimensions (assume same for all batch elements)
        latent_num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        video_latents = self._video_patchifier.patchify(video_latents)

        # Handle FPS with backward compatibility
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (already processed by embedding connectors in trainer)
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        # Create conditioning mask (first frame conditioning)
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Generate video positions
        video_positions = self._get_video_positions(
            num_frames=latent_num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Prepare keyframe conditioning tokens (if enabled)
        keyframe_result = self._prepare_keyframe_tokens(
            batch=batch,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Sample noise and sigmas
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)

        # Apply noise: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        # For first-frame conditioning tokens, use clean latents
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # Compute video targets (velocity prediction)
        video_targets = video_noise - video_latents

        # Create per-token timesteps
        video_timesteps = self._create_per_token_timesteps(video_conditioning_mask, sigmas.squeeze())

        # Concatenate keyframe tokens if present
        if keyframe_result is not None:
            kf_tokens, kf_positions, kf_seq_len = keyframe_result

            final_latent = torch.cat([noisy_video, kf_tokens], dim=1)
            final_positions = torch.cat([video_positions, kf_positions], dim=2)

            # Keyframe tokens use timestep=0 (clean conditioning)
            kf_timesteps = torch.zeros(batch_size, kf_seq_len, device=device, dtype=sigmas.dtype)
            final_timesteps = torch.cat([video_timesteps, kf_timesteps], dim=1)
        else:
            final_latent = noisy_video
            final_positions = video_positions
            final_timesteps = video_timesteps

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=final_latent,
            timesteps=final_timesteps,
            positions=final_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Video loss mask: True for tokens we want to compute loss on
        # First-frame conditioning tokens and keyframe tokens are excluded
        video_loss_mask = ~video_conditioning_mask
        if keyframe_result is not None:
            kf_mask = torch.zeros(batch_size, keyframe_result[2], dtype=torch.bool, device=device)
            video_loss_mask = torch.cat([video_loss_mask, kf_mask], dim=1)

        # Handle audio if enabled
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        if self.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            ref_seq_len=video_seq_len if keyframe_result is not None else None,
        )

    def _prepare_keyframe_tokens(
        self,
        batch: dict[str, Any],
        height: int,
        width: int,
        batch_size: int,
        fps: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, int] | None:
        """Prepare keyframe conditioning tokens and positions for concatenation.
        Keyframe latents are patchified and positioned at their original pixel-space
        frame indices, matching the inference-time behavior.
        Args:
            batch: Raw batch data containing keyframe information
            height: Latent height
            width: Latent width
            batch_size: Batch size
            fps: Frames per second
            device: Target device
            dtype: Target dtype
        Returns:
            Tuple of (keyframe_tokens, keyframe_positions, keyframe_seq_len) or None
        """
        if self.config.keyframe_conditioning_p <= 0 or "keyframes" not in batch:
            return None

        if torch.rand(1).item() >= self.config.keyframe_conditioning_p:
            return None

        keyframe_data = batch["keyframes"]
        if "keyframes" not in keyframe_data or len(keyframe_data["keyframes"]) == 0:
            return None

        all_tokens = []
        all_positions = []

        for kf_item in keyframe_data["keyframes"]:
            frame_idx = kf_item.get("frame_idx")
            if frame_idx is None:
                logger.warning("Keyframe item missing frame_idx, skipping")
                continue

            # DataLoader collates scalars into tensors; extract the scalar value
            if isinstance(frame_idx, torch.Tensor):
                frame_idx = frame_idx[0].item()

            # Get keyframe latent: [B, C, 1, H, W]
            kf_latent = kf_item["latent"].to(device=device, dtype=dtype)
            if kf_latent.dim() == 4:
                kf_latent = kf_latent.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

            # Patchify: [B, C, 1, H, W] -> [B, tokens_per_frame, C]
            kf_tokens = self._video_patchifier.patchify(kf_latent)

            # Compute positions with frame_idx offset (matching inference behavior)
            kf_positions = self._get_keyframe_positions(
                frame_idx=frame_idx,
                height=height,
                width=width,
                batch_size=batch_size,
                fps=fps,
                device=device,
                dtype=dtype,
            )

            all_tokens.append(kf_tokens)
            all_positions.append(kf_positions)

        if not all_tokens:
            return None

        combined_tokens = torch.cat(all_tokens, dim=1)
        combined_positions = torch.cat(all_positions, dim=2)
        return combined_tokens, combined_positions, combined_tokens.shape[1]

    def _get_keyframe_positions(
        self,
        frame_idx: int,
        height: int,
        width: int,
        batch_size: int,
        fps: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Generate position embeddings for a single keyframe at a given frame index.
        Args:
            frame_idx: Pixel-space frame index for this keyframe
            height: Latent height
            width: Latent width
            batch_size: Batch size
            fps: Frames per second
            device: Target device
            dtype: Target dtype
        Returns:
            Position tensor of shape [B, 3, tokens_per_frame, 2]
        """
        latent_coords = self._video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=1,
                height=height,
                width=width,
                batch=batch_size,
                channels=128,
            ),
            device=device,
        )

        # Only apply causal_fix for frame 0 (matching inference behavior)
        pixel_coords = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=VIDEO_SCALE_FACTORS,
            causal_fix=(frame_idx == 0),
        ).to(dtype)

        # Apply frame offset and fps scaling
        pixel_coords = pixel_coords.clone()
        pixel_coords[:, 0, ...] += frame_idx
        pixel_coords[:, 0, ...] /= fps

        return pixel_coords

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor]:
        """Prepare audio inputs for joint audio-video training.
        Args:
            batch: Raw batch data containing audio_latents
            sigmas: Sampled sigma values (same as video)
            audio_prompt_embeds: Audio context embeddings
            prompt_attention_mask: Attention mask for context
            batch_size: Batch size
            device: Target device
            dtype: Target dtype
        Returns:
            Tuple of (audio_modality, audio_targets, audio_loss_mask)
        """
        # Get audio latents - dataset provides uniform non-patchified format [B, C, T, F]
        audio_data = batch["audio_latents"]
        audio_latents = audio_data["latents"]

        # Patchify audio latents: [B, C, T, F] -> [B, T, C*F]
        audio_latents = self._audio_patchifier.patchify(audio_latents)

        audio_seq_len = audio_latents.shape[1]

        # Sample audio noise
        audio_noise = torch.randn_like(audio_latents)

        # Apply noise to audio (same sigma as video)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise

        # Compute audio targets
        audio_targets = audio_noise - audio_latents

        # Audio timesteps: all tokens use the sampled sigma (no conditioning mask)
        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len)

        # Generate audio positions
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # Create audio Modality
        audio_modality = Modality(
            enabled=True,
            latent=noisy_audio,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Audio loss mask: all tokens contribute to loss (no conditioning)
        audio_loss_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.bool, device=device)

        return audio_modality, audio_targets, audio_loss_mask

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for video and optionally audio.
        When keyframes are present, video_pred contains both video and keyframe tokens.
        Loss is computed only on the video portion, excluding first-frame conditioning
        and keyframe tokens.
        """
        # Slice predictions to video-only portion when keyframes are concatenated
        if inputs.ref_seq_len is not None and video_pred.shape[1] > inputs.ref_seq_len:
            video_pred = video_pred[:, : inputs.ref_seq_len, :]
            video_loss_mask = inputs.video_loss_mask[:, : inputs.ref_seq_len]
        else:
            video_loss_mask = inputs.video_loss_mask

        # Video loss
        video_loss = (video_pred - inputs.video_targets).pow(2)
        video_loss_mask = video_loss_mask.unsqueeze(-1).float()
        video_loss = video_loss.mul(video_loss_mask).div(video_loss_mask.mean())
        video_loss = video_loss.mean()

        # If no audio, return video loss only
        if not self.config.with_audio or audio_pred is None or inputs.audio_targets is None:
            return video_loss

        # Audio loss (no conditioning mask)
        audio_loss = (audio_pred - inputs.audio_targets).pow(2).mean()

        # Combined loss
        return video_loss + audio_loss
