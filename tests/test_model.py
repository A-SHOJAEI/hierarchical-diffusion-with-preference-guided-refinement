"""Tests for model components."""

import pytest
import torch

from hierarchical_diffusion_with_preference_guided_refinement.models import (
    HierarchicalDiffusionModel,
    PreferenceAdapter,
    RewardWeightedGuidance,
    PreferenceLoss,
    DenoisingScheduler,
    LoRAAdapter,
)


class TestRewardWeightedGuidance:
    """Test RewardWeightedGuidance module."""

    def test_guidance_creation(self):
        """Test guidance module can be created."""
        guidance = RewardWeightedGuidance(
            embed_dim=128,
            num_heads=4,
            guidance_scale=7.5,
        )
        assert guidance is not None

    def test_compute_reward(self, device, sample_images, sample_text_embeddings):
        """Test reward computation."""
        time_embed_dim = 128
        guidance = RewardWeightedGuidance(
            embed_dim=128,
            time_embed_dim=time_embed_dim,
            num_heads=4,
            guidance_scale=7.5,
        )
        guidance.to(device)

        batch_size = sample_images.shape[0]
        timestep_embed = torch.randn(batch_size, time_embed_dim, device=device)

        reward = guidance.compute_reward(
            sample_images,
            sample_text_embeddings,
            timestep_embed,
        )

        assert reward.shape == (batch_size, 1)

    def test_apply_guidance(self, device, sample_images, sample_text_embeddings):
        """Test guidance application."""
        time_embed_dim = 128
        guidance = RewardWeightedGuidance(
            embed_dim=128,
            time_embed_dim=time_embed_dim,
            num_heads=4,
            guidance_scale=7.5,
        )
        guidance.to(device)

        batch_size = sample_images.shape[0]
        noise_pred = torch.randn_like(sample_images)
        timestep_embed = torch.randn(batch_size, time_embed_dim, device=device)

        guided = guidance.apply_guidance(
            noise_pred,
            sample_images,
            sample_text_embeddings,
            timestep_embed,
            do_classifier_free_guidance=False,
        )

        assert guided.shape == noise_pred.shape


class TestPreferenceLoss:
    """Test PreferenceLoss module."""

    def test_preference_loss_creation(self):
        """Test preference loss can be created."""
        loss_fn = PreferenceLoss(margin=0.5, temperature=1.0)
        assert loss_fn is not None

    def test_preference_loss_computation(self, device):
        """Test preference loss computation."""
        loss_fn = PreferenceLoss(margin=0.5, temperature=1.0)

        better_reward = torch.randn(4, 1, device=device)
        worse_reward = torch.randn(4, 1, device=device)

        loss, metrics = loss_fn(better_reward, worse_reward)

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert "preference_loss" in metrics
        assert "preference_accuracy" in metrics

    def test_preference_loss_with_margin(self, device):
        """Test preference loss with custom margin."""
        loss_fn = PreferenceLoss(margin=0.5, temperature=1.0)

        better_reward = torch.randn(4, 1, device=device)
        worse_reward = torch.randn(4, 1, device=device)
        margin = torch.ones(4, 1, device=device) * 0.3

        loss, metrics = loss_fn(better_reward, worse_reward, margin)

        assert isinstance(loss, torch.Tensor)


class TestDenoisingScheduler:
    """Test DenoisingScheduler."""

    def test_scheduler_creation(self):
        """Test scheduler can be created."""
        scheduler = DenoisingScheduler(
            num_train_timesteps=100,
            num_inference_steps=10,
        )
        assert scheduler is not None
        assert len(scheduler.timesteps) == 10

    def test_add_noise(self, device):
        """Test noise addition."""
        scheduler = DenoisingScheduler(
            num_train_timesteps=100,
            num_inference_steps=10,
        )

        original = torch.randn(2, 3, 32, 32, device=device)
        noise = torch.randn_like(original)
        timesteps = torch.tensor([10, 20], device=device)

        noisy = scheduler.add_noise(original, noise, timesteps)

        assert noisy.shape == original.shape

    def test_step(self, device):
        """Test denoising step."""
        scheduler = DenoisingScheduler(
            num_train_timesteps=100,
            num_inference_steps=10,
        )

        sample = torch.randn(2, 3, 32, 32, device=device)
        model_output = torch.randn_like(sample)
        timestep = 50

        prev_sample = scheduler.step(model_output, timestep, sample)

        assert prev_sample.shape == sample.shape


class TestLoRAAdapter:
    """Test LoRAAdapter."""

    def test_lora_creation(self):
        """Test LoRA adapter can be created."""
        adapter = LoRAAdapter(
            in_features=128,
            out_features=128,
            rank=4,
            alpha=1.0,
        )
        assert adapter is not None

    def test_lora_forward(self, device):
        """Test LoRA forward pass."""
        adapter = LoRAAdapter(
            in_features=128,
            out_features=128,
            rank=4,
            alpha=1.0,
        )
        adapter.to(device)

        x = torch.randn(2, 128, device=device)
        output = adapter(x)

        assert output.shape == x.shape


class TestHierarchicalDiffusionModel:
    """Test HierarchicalDiffusionModel."""

    def test_model_creation(self, sample_config):
        """Test model can be created."""
        model = HierarchicalDiffusionModel(
            image_size=sample_config["image_size"],
            hidden_dims=sample_config["hidden_dims"],
            text_embed_dim=sample_config["text_embed_dim"],
            use_preference_adapter=True,
        )
        assert model is not None

    def test_model_forward(self, device, sample_config, sample_images,
                          sample_text_embeddings, sample_timesteps):
        """Test model forward pass."""
        model = HierarchicalDiffusionModel(
            image_size=sample_config["image_size"],
            hidden_dims=sample_config["hidden_dims"],
            text_embed_dim=sample_config["text_embed_dim"],
            use_preference_adapter=False,
        )
        model.to(device)

        output = model(
            sample_images,
            sample_timesteps,
            sample_text_embeddings,
        )

        assert output.shape == sample_images.shape

    def test_model_with_adapter(self, device, sample_config, sample_images,
                               sample_text_embeddings, sample_timesteps):
        """Test model with preference adapter."""
        model = HierarchicalDiffusionModel(
            image_size=sample_config["image_size"],
            hidden_dims=sample_config["hidden_dims"],
            text_embed_dim=sample_config["text_embed_dim"],
            use_preference_adapter=True,
        )
        model.to(device)

        output = model(
            sample_images,
            sample_timesteps,
            sample_text_embeddings,
            use_adapter=True,
        )

        assert output.shape == sample_images.shape

    def test_model_generation(self, device, sample_config):
        """Test model image generation."""
        model = HierarchicalDiffusionModel(
            image_size=sample_config["image_size"],
            hidden_dims=sample_config["hidden_dims"],
            text_embed_dim=sample_config["text_embed_dim"],
            use_preference_adapter=False,
        )
        model.to(device)

        text_embed = torch.randn(1, 77, sample_config["text_embed_dim"], device=device)

        generated = model.generate(
            text_embed=text_embed,
            batch_size=1,
            num_inference_steps=5,
            device=device,
        )

        assert generated.shape == (1, 3, sample_config["image_size"], sample_config["image_size"])

    def test_get_trainable_parameters(self, sample_config):
        """Test getting trainable parameters."""
        model = HierarchicalDiffusionModel(
            image_size=sample_config["image_size"],
            hidden_dims=sample_config["hidden_dims"],
            text_embed_dim=sample_config["text_embed_dim"],
            use_preference_adapter=True,
        )

        params = model.get_trainable_parameters()
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
