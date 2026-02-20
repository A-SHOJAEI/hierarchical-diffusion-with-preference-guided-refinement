"""Training loop with learning rate scheduling and early stopping."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np

from ..models.components import PreferenceLoss

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """Trainer for base diffusion model.

    Args:
        model: Diffusion model to train
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Device to train on
        mixed_precision: Whether to use mixed precision training
        gradient_clip: Gradient clipping value (None to disable)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        mixed_precision: bool = True,
        gradient_clip: Optional[float] = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip

        # Mixed precision training
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Metrics tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        logger.info(f"Initialized DiffusionTrainer (device={device}, mixed_precision={mixed_precision})")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch["image"].to(self.device)
            captions = batch["caption"]

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                loss = self._compute_diffusion_loss(images, captions)

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return {"train_loss": avg_loss}

    def _compute_diffusion_loss(
        self,
        images: torch.Tensor,
        captions: List[str],
    ) -> torch.Tensor:
        """Compute diffusion training loss.

        Args:
            images: Clean images [B, C, H, W]
            captions: List of text captions

        Returns:
            Loss value
        """
        batch_size = images.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.model.scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device,
        )

        # Sample noise
        noise = torch.randn_like(images)

        # Add noise to images
        noisy_images = self.model.scheduler.add_noise(images, noise, timesteps)

        # Get text embeddings (placeholder - in practice use CLIP)
        # For now, use dummy embeddings
        text_embed_dim = self.model.base_model.text_proj.in_features
        text_embed = torch.randn(batch_size, 77, text_embed_dim, device=self.device)

        # Predict noise
        noise_pred = self.model(noisy_images, timesteps, text_embed)

        # Compute MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                captions = batch["caption"]

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    loss = self._compute_diffusion_loss(images, captions)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)

        logger.info(f"Epoch {epoch} - Validation Loss: {avg_loss:.4f}")

        return {"val_loss": avg_loss}

    def save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save model checkpoint.

        Args:
            save_path: Path to save checkpoint
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with checkpoint info
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return checkpoint


class PreferenceTrainer:
    """Trainer for preference-guided refinement stage.

    This implements the novel RLHF-inspired preference optimization.

    Args:
        model: Model with preference adapter
        optimizer: Optimizer instance
        preference_loss: Preference loss function
        scheduler: Learning rate scheduler
        device: Device to train on
        mixed_precision: Whether to use mixed precision
        gradient_clip: Gradient clipping value
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        preference_loss: PreferenceLoss,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        mixed_precision: bool = True,
        gradient_clip: Optional[float] = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.preference_loss = preference_loss
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip

        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        self.train_metrics: List[Dict[str, float]] = []
        self.val_metrics: List[Dict[str, float]] = []

        logger.info("Initialized PreferenceTrainer")

    def train_epoch(
        self,
        preference_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train preference model for one epoch.

        Args:
            preference_loader: Preference data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(preference_loader):
            # Get preference pairs
            better_img = batch["better_image"].to(self.device)
            worse_img = batch["worse_image"].to(self.device)
            captions = batch["caption"]
            margin = batch["margin"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                loss, metrics = self._compute_preference_loss(
                    better_img, worse_img, captions, margin
                )

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_trainable_parameters(),
                        self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_trainable_parameters(),
                        self.gradient_clip
                    )
                self.optimizer.step()

            total_loss += metrics["preference_loss"]
            total_accuracy += metrics["preference_accuracy"]
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx + 1}/{len(preference_loader)}] "
                    f"Loss: {metrics['preference_loss']:.4f}, "
                    f"Acc: {metrics['preference_accuracy']:.4f}"
                )

        avg_metrics = {
            "train_loss": total_loss / num_batches,
            "train_accuracy": total_accuracy / num_batches,
        }
        self.train_metrics.append(avg_metrics)

        return avg_metrics

    def _compute_preference_loss(
        self,
        better_img: torch.Tensor,
        worse_img: torch.Tensor,
        captions: List[str],
        margin: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute preference ranking loss.

        Args:
            better_img: Preferred images [B, C, H, W]
            worse_img: Less preferred images [B, C, H, W]
            captions: Text captions
            margin: Preference margins [B, 1]

        Returns:
            Tuple of (loss, metrics)
        """
        batch_size = better_img.shape[0]

        # Generate text embeddings (placeholder)
        text_embed_dim = self.model.base_model.text_proj.in_features
        text_embed = torch.randn(batch_size, 77, text_embed_dim, device=self.device)

        # Sample timesteps
        timesteps = torch.randint(
            0,
            self.model.scheduler.num_train_timesteps // 2,  # Focus on later steps
            (batch_size,),
            device=self.device,
        )

        # Get time embeddings
        time_embed = self.model.base_model.get_time_embedding(timesteps)

        # Compute rewards for both images
        better_reward = self.model.guidance.compute_reward(
            better_img, text_embed, time_embed
        )
        worse_reward = self.model.guidance.compute_reward(
            worse_img, text_embed, time_embed
        )

        # Compute preference loss
        loss, metrics = self.preference_loss(better_reward, worse_reward, margin)

        return loss, metrics

    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Validate preference model.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                better_img = batch["better_image"].to(self.device)
                worse_img = batch["worse_image"].to(self.device)
                captions = batch["caption"]
                margin = batch["margin"].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    loss, metrics = self._compute_preference_loss(
                        better_img, worse_img, captions, margin
                    )

                total_loss += metrics["preference_loss"]
                total_accuracy += metrics["preference_accuracy"]
                num_batches += 1

        avg_metrics = {
            "val_loss": total_loss / num_batches,
            "val_accuracy": total_accuracy / num_batches,
        }
        self.val_metrics.append(avg_metrics)

        logger.info(
            f"Epoch {epoch} - Val Loss: {avg_metrics['val_loss']:.4f}, "
            f"Val Acc: {avg_metrics['val_accuracy']:.4f}"
        )

        return avg_metrics

    def save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric: float) -> bool:
        """Check if training should stop.

        Args:
            val_metric: Validation metric value

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_metric
            return False

        if self.mode == "min":
            improved = val_metric < self.best_score - self.min_delta
        else:
            improved = val_metric > self.best_score + self.min_delta

        if improved:
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                logger.info("Early stopping triggered")
                self.early_stop = True
                return True

        return False
