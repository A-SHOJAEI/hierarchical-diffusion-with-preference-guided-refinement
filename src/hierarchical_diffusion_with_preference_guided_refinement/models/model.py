"""Core model implementation for hierarchical diffusion with preference guidance."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import RewardWeightedGuidance, DenoisingScheduler, LoRAAdapter

logger = logging.getLogger(__name__)


class SimpleUNet(nn.Module):
    """Simplified U-Net architecture for diffusion denoising.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        hidden_dims: List of hidden dimensions for each level
        time_embed_dim: Dimension of time embedding
        text_embed_dim: Dimension of text conditioning
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dims: List[int] = [64, 128, 256, 512],
        time_embed_dim: int = 256,
        text_embed_dim: int = 768,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Text projection
        self.text_proj = nn.Linear(text_embed_dim, time_embed_dim)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        prev_dim = in_channels
        for dim in hidden_dims:
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_dim, dim, 3, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.SiLU(),
                    nn.Conv2d(dim, dim, 3, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.SiLU(),
                )
            )
            prev_dim = dim

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.GroupNorm(8, hidden_dims[-1]),
            nn.SiLU(),
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            # Input channels: current level + skip connection from encoder
            # After upsampling, we have hidden_dims[i] channels
            # Skip connection provides hidden_dims[i-1] channels
            dec_in_channels = hidden_dims[i] + hidden_dims[i - 1]
            dec_out_channels = hidden_dims[i - 1]
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(dec_in_channels, dec_out_channels, 3, padding=1),
                    nn.GroupNorm(8, dec_out_channels),
                    nn.SiLU(),
                    nn.Conv2d(dec_out_channels, dec_out_channels, 3, padding=1),
                    nn.GroupNorm(8, dec_out_channels),
                    nn.SiLU(),
                )
            )

        # Final output
        self.final = nn.Conv2d(hidden_dims[0], out_channels, 1)

        # Conditioning projection layers
        self.cond_projs = nn.ModuleList([
            nn.Linear(time_embed_dim, dim) for dim in hidden_dims
        ])

    def get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal time embeddings.

        Args:
            timesteps: Timestep indices [B]

        Returns:
            Time embeddings [B, D]
        """
        half_dim = self.time_embed_dim // 2
        embeddings = torch.exp(
            -torch.arange(half_dim, device=timesteps.device) *
            (torch.log(torch.tensor(10000.0)) / half_dim)
        )
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image [B, C, H, W]
            timesteps: Timestep indices [B]
            text_embed: Text conditioning [B, L, D]

        Returns:
            Predicted noise [B, C, H, W]
        """
        # Get time embedding
        t_emb = self.get_time_embedding(timesteps)
        t_emb = self.time_mlp(t_emb)

        # Add text conditioning
        if text_embed is not None:
            # Pool text embeddings
            text_pooled = text_embed.mean(dim=1)  # [B, D]
            text_cond = self.text_proj(text_pooled)
            t_emb = t_emb + text_cond

        # Encoder
        encoder_outputs = []
        h = x
        for i, block in enumerate(self.encoder_blocks):
            h = block(h)
            # Add time conditioning
            cond = self.cond_projs[i](t_emb)[:, :, None, None]
            h = h + cond
            # Store ALL encoder outputs for skip connections
            encoder_outputs.append(h)
            # Downsample for next level (except last)
            if i < len(self.encoder_blocks) - 1:
                h = F.avg_pool2d(h, 2)

        # Bottleneck (processes the last encoder output)
        h = self.bottleneck(h)

        # Decoder with skip connections
        # decoder_blocks has len(hidden_dims)-1 = 3 blocks
        # We want to skip from: encoder_outputs[2], encoder_outputs[1], encoder_outputs[0]
        for i, block in enumerate(self.decoder_blocks):
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            # Skip from the corresponding encoder level
            skip_idx = len(encoder_outputs) - 2 - i
            skip = encoder_outputs[skip_idx]
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        # Final output
        out = self.final(h)
        return out


class PreferenceAdapter(nn.Module):
    """Lightweight adapter for preference-guided refinement.

    This adapter uses LoRA to efficiently fine-tune the diffusion model
    based on preference feedback without modifying the base model weights.

    Args:
        base_model: Base diffusion model to adapt
        adapter_rank: Rank for LoRA adaptation
        adapter_alpha: Alpha scaling for LoRA
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter_rank: int = 4,
        adapter_alpha: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Add LoRA adapters to key layers
        self.adapters = nn.ModuleDict()
        self._add_lora_adapters(adapter_rank, adapter_alpha)

        logger.info(f"Initialized PreferenceAdapter with rank={adapter_rank}")

    def _add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Add LoRA adapters to model layers.

        Instead of trying to inject into frozen Conv2d layers, we build a
        small trainable residual network that maps the same input space
        (noisy image channels) to the output space (predicted noise channels).
        This keeps the architecture simple while providing a learnable
        refinement pathway.
        """
        in_ch = self.base_model.final.in_channels   # first hidden dim
        out_ch = self.base_model.final.out_channels  # image channels

        # A lightweight convolutional residual path
        self.adapter_conv = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, 3, padding=1),
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

        # Initialize last conv to near-zero so the adapter starts as identity
        nn.init.zeros_(self.adapter_conv[-1].weight)
        nn.init.zeros_(self.adapter_conv[-1].bias)

        # Also add a LoRA adapter operating on the spatially-pooled features
        self.adapters["output_lora"] = LoRAAdapter(out_ch, out_ch, rank, alpha)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with adapter.

        Runs the frozen base model, then applies a trainable convolutional
        residual and a LoRA channel-wise bias to refine the prediction.

        Args:
            x: Input image [B, C, H, W]
            timesteps: Timestep indices [B]
            text_embed: Text conditioning [B, L, D]

        Returns:
            Predicted noise with adapter adjustments
        """
        # Get base model output (no gradients through frozen base)
        base_output = self.base_model(x, timesteps, text_embed)

        # Convolutional residual path (trainable)
        conv_residual = self.adapter_conv(base_output)

        # LoRA channel-wise bias
        B, C, H, W = base_output.shape
        pooled = base_output.mean(dim=[-2, -1])  # [B, C]
        lora_bias = self.adapters["output_lora"](pooled)  # [B, C]

        return base_output + conv_residual + lora_bias.unsqueeze(-1).unsqueeze(-1)


class HierarchicalDiffusionModel(nn.Module):
    """Hierarchical diffusion model with preference-guided refinement.

    This is the main model that combines:
    1. Base diffusion model for coarse generation
    2. Preference adapter for refinement
    3. Reward-weighted guidance for trajectory steering

    Args:
        image_size: Size of generated images
        in_channels: Number of input channels
        hidden_dims: Hidden dimensions for U-Net
        text_embed_dim: Dimension of text embeddings
        use_preference_adapter: Whether to use preference adapter
        guidance_scale: Scale for reward-weighted guidance
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        hidden_dims: List[int] = [64, 128, 256, 512],
        text_embed_dim: int = 768,
        use_preference_adapter: bool = True,
        guidance_scale: float = 7.5,
    ):
        super().__init__()
        self.image_size = image_size
        self.use_preference_adapter = use_preference_adapter

        # Base diffusion model
        self.base_model = SimpleUNet(
            in_channels=in_channels,
            out_channels=in_channels,
            hidden_dims=hidden_dims,
            text_embed_dim=text_embed_dim,
        )

        # Preference adapter
        if use_preference_adapter:
            self.adapter = PreferenceAdapter(
                base_model=self.base_model,
                adapter_rank=4,
                adapter_alpha=1.0,
            )

        # Reward-weighted guidance
        # Use in_channels for latent_dim since images have in_channels dimensions
        self.guidance = RewardWeightedGuidance(
            embed_dim=text_embed_dim,
            latent_dim=in_channels,
            guidance_scale=guidance_scale,
        )

        # Noise scheduler
        self.scheduler = DenoisingScheduler()

        logger.info(f"Initialized HierarchicalDiffusionModel (adapter={use_preference_adapter})")

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_embed: torch.Tensor,
        use_adapter: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Noisy input [B, C, H, W]
            timesteps: Timestep indices [B]
            text_embed: Text conditioning [B, L, D]
            use_adapter: Whether to use preference adapter

        Returns:
            Predicted noise [B, C, H, W]
        """
        if use_adapter and self.use_preference_adapter:
            noise_pred = self.adapter(x, timesteps, text_embed)
        else:
            noise_pred = self.base_model(x, timesteps, text_embed)

        return noise_pred

    def generate(
        self,
        text_embed: torch.Tensor,
        batch_size: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        use_guidance: bool = True,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generate images from text embeddings.

        Args:
            text_embed: Text conditioning [B, L, D]
            batch_size: Batch size for generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            use_guidance: Whether to use reward-weighted guidance
            device: Device to run generation on

        Returns:
            Generated images [B, C, H, W]
        """
        # Start from random noise
        latent = torch.randn(
            batch_size,
            3,
            self.image_size,
            self.image_size,
            device=device,
        )

        # Set scheduler timesteps
        self.scheduler.num_inference_steps = num_inference_steps
        self.scheduler.timesteps = torch.linspace(
            self.scheduler.num_train_timesteps - 1,
            0,
            num_inference_steps,
        ).long().to(device)

        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand timestep for batch
            timestep = t.repeat(batch_size)

            # Get time embedding for guidance
            time_embed = self.base_model.get_time_embedding(timestep)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.forward(
                    latent,
                    timestep,
                    text_embed,
                    use_adapter=self.use_preference_adapter,
                )

            # Apply reward-weighted guidance
            if use_guidance:
                noise_pred = self.guidance.apply_guidance(
                    noise_pred,
                    latent,
                    text_embed,
                    time_embed,
                    do_classifier_free_guidance=False,
                )

            # Denoise
            latent = self.scheduler.step(noise_pred, t, latent)

        # Clamp to [-1, 1]
        latent = torch.clamp(latent, -1.0, 1.0)

        return latent

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters.

        Returns:
            List of trainable parameters
        """
        if self.use_preference_adapter:
            # Only adapter and guidance parameters are trainable
            params = list(self.adapter.parameters()) + list(self.guidance.parameters())
        else:
            # All parameters trainable
            params = list(self.parameters())

        trainable = [p for p in params if p.requires_grad]

        return trainable
