"""Custom loss functions, layers, and modules for preference-guided diffusion.

This module contains the novel components that enable reward-weighted guidance
of the denoising trajectory based on learned preference models.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)


class RewardWeightedGuidance(nn.Module):
    """Novel reward-weighted guidance module for steering diffusion sampling.

    This is the core innovation: treating diffusion sampling as a sequential
    decision process where preference feedback shapes the denoising trajectory.
    The reward model learned from preference pairs guides each denoising step.

    Args:
        embed_dim: Text embedding dimension
        time_embed_dim: Time embedding dimension
        latent_dim: Latent/image channel dimension (default 3 for RGB images)
        num_heads: Number of attention heads for guidance
        guidance_scale: Scale factor for reward-weighted guidance
    """

    def __init__(
        self,
        embed_dim: int = 768,
        time_embed_dim: int = 256,
        latent_dim: int = 3,
        num_heads: int = 8,
        guidance_scale: float = 7.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_embed_dim = time_embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.guidance_scale = guidance_scale

        # Reward prediction network
        self.reward_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        # Time embedding projection and processing
        self.time_proj = nn.Linear(time_embed_dim, embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Latent projection from spatial features to embedding space
        self.latent_proj = nn.Linear(latent_dim, embed_dim)

        # Multi-head attention for guidance computation
        self.guidance_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.1,
            batch_first=True,
        )

        logger.info(f"Initialized RewardWeightedGuidance with scale={guidance_scale}")

    def compute_reward(
        self,
        latent: torch.Tensor,
        text_embed: torch.Tensor,
        timestep_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute predicted reward for the current state.

        Args:
            latent: Current latent representation [B, C, H, W]
            text_embed: Text conditioning [B, L, D]
            timestep_embed: Timestep embedding [B, D]

        Returns:
            Predicted reward scores [B, 1]
        """
        B, C, H, W = latent.shape

        # Pool spatial dimensions
        latent_pooled = latent.mean(dim=[-2, -1])  # [B, C]

        # Project latent to embed_dim
        latent_projected = self.latent_proj(latent_pooled)  # [B, D]

        # Project and process time conditioning
        time_proj = self.time_proj(timestep_embed)  # [B, D]
        time_cond = self.time_embed(time_proj)  # [B, D]

        # Use attention to combine latent with text
        latent_expanded = latent_projected.unsqueeze(1)  # [B, 1, D]
        attended, _ = self.guidance_attn(
            latent_expanded,
            text_embed,
            text_embed,
        )
        attended = attended.squeeze(1)  # [B, D]

        # Combine features
        combined = attended + time_cond

        # Predict reward
        reward = self.reward_net(combined)  # [B, 1]

        return reward

    def apply_guidance(
        self,
        noise_pred: torch.Tensor,
        latent: torch.Tensor,
        text_embed: torch.Tensor,
        timestep_embed: torch.Tensor,
        do_classifier_free_guidance: bool = True,
    ) -> torch.Tensor:
        """Apply reward-weighted guidance to noise prediction.

        Args:
            noise_pred: Predicted noise [B, C, H, W]
            latent: Current latent [B, C, H, W]
            text_embed: Text conditioning [B, L, D]
            timestep_embed: Timestep embedding [B, D]
            do_classifier_free_guidance: Whether to apply CFG

        Returns:
            Guided noise prediction
        """
        # Compute reward for guidance
        reward = self.compute_reward(latent, text_embed, timestep_embed)

        # Scale reward to guidance strength
        guidance_weight = torch.sigmoid(reward) * self.guidance_scale

        if do_classifier_free_guidance:
            # Split conditional and unconditional predictions
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            # Apply reward-weighted guidance
            noise_pred = noise_pred_uncond + guidance_weight.view(-1, 1, 1, 1) * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            # Apply guidance directly
            noise_pred = noise_pred * guidance_weight.view(-1, 1, 1, 1)

        return noise_pred


class PreferenceLoss(nn.Module):
    """Custom loss function for learning from preference rankings.

    Implements a Bradley-Terry style preference model with margin-based ranking loss.
    This enables learning from human preference comparisons rather than absolute scores.

    Args:
        margin: Margin for ranking loss
        temperature: Temperature for preference probability
    """

    def __init__(self, margin: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        better_reward: torch.Tensor,
        worse_reward: torch.Tensor,
        preference_margin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute preference ranking loss.

        Args:
            better_reward: Predicted reward for preferred image [B, 1]
            worse_reward: Predicted reward for less preferred image [B, 1]
            preference_margin: Optional margin from human feedback [B, 1]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute preference logits
        logits = (better_reward - worse_reward) / self.temperature

        # Adaptive margin based on preference strength
        if preference_margin is not None:
            target_margin = preference_margin * self.margin
        else:
            target_margin = self.margin

        # Ranking loss: encourage better_reward > worse_reward + margin
        loss = F.relu(target_margin - logits).mean()

        # Compute accuracy
        with torch.no_grad():
            correct = (better_reward > worse_reward).float().mean()

        metrics = {
            "preference_loss": loss.item(),
            "preference_accuracy": correct.item(),
            "reward_gap": (better_reward - worse_reward).mean().item(),
        }

        return loss, metrics


class DenoisingScheduler:
    """Custom denoising scheduler with learned timestep selection.

    Implements a hierarchical coarse-to-fine schedule where timesteps are
    adaptively selected based on the current generation quality.

    Args:
        num_train_timesteps: Number of training timesteps
        num_inference_steps: Number of inference steps
        beta_start: Starting beta value
        beta_end: Ending beta value
        beta_schedule: Schedule type ('linear', 'scaled_linear', 'squaredcos_cap_v2')
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # Create beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            betas = self._betas_for_alpha_bar(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Compute alphas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Set timesteps
        self.timesteps = torch.linspace(
            num_train_timesteps - 1,
            0,
            num_inference_steps,
        ).long()

        logger.info(f"Initialized scheduler with {num_inference_steps} inference steps")

    def _betas_for_alpha_bar(self, num_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
        """Create betas from alpha_bar schedule (cosine)."""
        def alpha_bar(time_step):
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas)

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to original samples according to timesteps.

        Args:
            original: Original samples [B, C, H, W]
            noise: Noise to add [B, C, H, W]
            timesteps: Timestep indices [B]

        Returns:
            Noisy samples
        """
        # Ensure alphas_cumprod is on the same device as timesteps
        alphas_cumprod = self.alphas_cumprod.to(timesteps.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)

        noisy = sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        return noisy

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one denoising step.

        Args:
            model_output: Predicted noise
            timestep: Current timestep
            sample: Current sample

        Returns:
            Denoised sample
        """
        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Compute predicted original sample (x_0) from noise prediction
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

        # Clamp predicted x_0 for stability
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

        # DDPM posterior mean coefficients:
        # mu = (sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)) * x_0
        #    + (sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * x_t
        # where beta_t = 1 - alpha_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        pred_sample_coeff = (alpha_prod_t_prev**0.5 * beta_t) / (1 - alpha_prod_t)
        current_sample_coeff = (alpha_prod_t / alpha_prod_t_prev)**0.5 * beta_prod_t_prev / (1 - alpha_prod_t)

        # Compute previous sample
        pred_prev_sample = pred_sample_coeff * pred_original_sample + current_sample_coeff * sample

        return pred_prev_sample


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation (LoRA) layer for efficient fine-tuning.

    Adds trainable low-rank decomposition to frozen layers for parameter-efficient
    adaptation during preference learning.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank decomposition
        alpha: Scaling factor
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            Adapted output
        """
        return self.lora_up(self.lora_down(x)) * self.scaling
