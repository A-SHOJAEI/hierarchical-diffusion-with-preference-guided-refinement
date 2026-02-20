"""Tests for training components."""

import pytest
import torch
from torch.optim import Adam

from hierarchical_diffusion_with_preference_guided_refinement.models import (
    HierarchicalDiffusionModel,
    PreferenceLoss,
)
from hierarchical_diffusion_with_preference_guided_refinement.training import (
    DiffusionTrainer,
    PreferenceTrainer,
    EarlyStopping,
)
from hierarchical_diffusion_with_preference_guided_refinement.data import (
    ConceptualCaptionsDataset,
    PreferenceDataset,
)


class TestDiffusionTrainer:
    """Test DiffusionTrainer."""

    def test_trainer_creation(self, device, sample_config):
        """Test trainer can be created."""
        model = HierarchicalDiffusionModel(
            image_size=sample_config["image_size"],
            hidden_dims=sample_config["hidden_dims"],
            text_embed_dim=sample_config["text_embed_dim"],
            use_preference_adapter=False,
        )
        model.to(device)

        optimizer = Adam(model.parameters(), lr=0.001)

        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            mixed_precision=False,
        )

        assert trainer is not None

    def test_compute_diffusion_loss(self, device, sample_config):
        """Test diffusion loss computation."""
        model = HierarchicalDiffusionModel(
            image_size=sample_config["image_size"],
            hidden_dims=sample_config["hidden_dims"],
            text_embed_dim=sample_config["text_embed_dim"],
            use_preference_adapter=False,
        )
        model.to(device)

        optimizer = Adam(model.parameters(), lr=0.001)

        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            mixed_precision=False,
        )

        images = torch.randn(2, 3, sample_config["image_size"],
                           sample_config["image_size"], device=device)
        captions = ["test caption 1", "test caption 2"]

        loss = trainer._compute_diffusion_loss(images, captions)

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1

    def test_train_epoch(self, device, sample_config):
        """Test training for one epoch."""
        model = HierarchicalDiffusionModel(
            image_size=sample_config["image_size"],
            hidden_dims=sample_config["hidden_dims"],
            text_embed_dim=sample_config["text_embed_dim"],
            use_preference_adapter=False,
        )
        model.to(device)

        optimizer = Adam(model.parameters(), lr=0.001)

        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            mixed_precision=False,
        )

        # Create small dataset
        dataset = ConceptualCaptionsDataset(
            split="train",
            max_samples=4,
            image_size=sample_config["image_size"],
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
        )

        metrics = trainer.train_epoch(dataloader, epoch=1)

        assert "train_loss" in metrics
        assert isinstance(metrics["train_loss"], float)

    def test_checkpoint_save_load(self, device, sample_config, temp_dir):
        """Test checkpoint saving and loading."""
        model = HierarchicalDiffusionModel(
            image_size=sample_config["image_size"],
            hidden_dims=sample_config["hidden_dims"],
            text_embed_dim=sample_config["text_embed_dim"],
            use_preference_adapter=False,
        )
        model.to(device)

        optimizer = Adam(model.parameters(), lr=0.001)

        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            mixed_precision=False,
        )

        # Save checkpoint
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=1, metrics={"loss": 0.5})

        assert checkpoint_path.exists()

        # Load checkpoint
        checkpoint = trainer.load_checkpoint(str(checkpoint_path))

        assert "epoch" in checkpoint
        assert checkpoint["epoch"] == 1


class TestPreferenceTrainer:
    """Test PreferenceTrainer."""

    def test_preference_trainer_creation(self, device, sample_config):
        """Test preference trainer can be created."""
        model = HierarchicalDiffusionModel(
            image_size=sample_config["image_size"],
            hidden_dims=sample_config["hidden_dims"],
            text_embed_dim=sample_config["text_embed_dim"],
            use_preference_adapter=True,
        )
        model.to(device)

        trainable_params = model.get_trainable_parameters()
        optimizer = Adam(trainable_params, lr=0.001)
        preference_loss = PreferenceLoss(margin=0.5)

        trainer = PreferenceTrainer(
            model=model,
            optimizer=optimizer,
            preference_loss=preference_loss,
            device=device,
            mixed_precision=False,
        )

        assert trainer is not None

    def test_compute_preference_loss(self, device, sample_config):
        """Test preference loss computation."""
        model = HierarchicalDiffusionModel(
            image_size=sample_config["image_size"],
            hidden_dims=sample_config["hidden_dims"],
            text_embed_dim=sample_config["text_embed_dim"],
            use_preference_adapter=True,
        )
        model.to(device)

        trainable_params = model.get_trainable_parameters()
        optimizer = Adam(trainable_params, lr=0.001)
        preference_loss = PreferenceLoss(margin=0.5)

        trainer = PreferenceTrainer(
            model=model,
            optimizer=optimizer,
            preference_loss=preference_loss,
            device=device,
            mixed_precision=False,
        )

        better_img = torch.randn(2, 3, sample_config["image_size"],
                                sample_config["image_size"], device=device)
        worse_img = torch.randn(2, 3, sample_config["image_size"],
                               sample_config["image_size"], device=device)
        captions = ["test caption 1", "test caption 2"]
        margin = torch.ones(2, 1, device=device) * 0.5

        loss, metrics = trainer._compute_preference_loss(
            better_img, worse_img, captions, margin
        )

        assert isinstance(loss, torch.Tensor)
        assert "preference_loss" in metrics
        assert "preference_accuracy" in metrics


class TestEarlyStopping:
    """Test EarlyStopping."""

    def test_early_stopping_creation(self):
        """Test early stopping can be created."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        assert early_stopping is not None

    def test_early_stopping_improvement(self):
        """Test early stopping with improvement."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.001, mode="min")

        # Should not stop when improving
        assert not early_stopping(1.0)
        assert not early_stopping(0.9)
        assert not early_stopping(0.8)

    def test_early_stopping_no_improvement(self):
        """Test early stopping without improvement."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.001, mode="min")

        # Initial value
        early_stopping(1.0)

        # No improvement
        assert not early_stopping(1.1)
        assert not early_stopping(1.1)

        # Should stop after patience
        assert early_stopping(1.1)

    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.001, mode="max")

        # Should not stop when improving
        assert not early_stopping(0.8)
        assert not early_stopping(0.9)
        assert not early_stopping(1.0)
