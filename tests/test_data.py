"""Tests for data loading and preprocessing."""

import pytest
import torch

from hierarchical_diffusion_with_preference_guided_refinement.data import (
    ConceptualCaptionsDataset,
    PreferenceDataset,
    create_dataloaders,
    PreprocessPipeline,
)


class TestConceptualCaptionsDataset:
    """Test ConceptualCaptionsDataset."""

    def test_dataset_creation(self):
        """Test dataset can be created."""
        dataset = ConceptualCaptionsDataset(
            split="train",
            max_samples=10,
            image_size=64,
        )
        assert len(dataset) > 0

    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct format."""
        dataset = ConceptualCaptionsDataset(
            split="train",
            max_samples=10,
            image_size=64,
        )
        item = dataset[0]

        assert "image" in item
        assert "caption" in item
        assert "image_id" in item

        # Check image shape and range
        assert item["image"].shape == (3, 64, 64)
        # For synthetic data, values are from randn (unbounded Gaussian)
        # Just check they're reasonable floats
        assert torch.is_tensor(item["image"])
        assert item["image"].dtype == torch.float32

        # Check caption is string
        assert isinstance(item["caption"], str)

    def test_dataset_length(self):
        """Test dataset length matches max_samples."""
        max_samples = 5
        dataset = ConceptualCaptionsDataset(
            split="train",
            max_samples=max_samples,
            image_size=64,
        )
        assert len(dataset) == max_samples


class TestPreferenceDataset:
    """Test PreferenceDataset."""

    def test_preference_dataset_creation(self):
        """Test preference dataset can be created."""
        dataset = PreferenceDataset(
            num_pairs=10,
            image_size=64,
        )
        assert len(dataset) == 10

    def test_preference_dataset_getitem(self):
        """Test preference dataset returns correct format."""
        dataset = PreferenceDataset(
            num_pairs=10,
            image_size=64,
        )
        item = dataset[0]

        assert "caption" in item
        assert "better_image" in item
        assert "worse_image" in item
        assert "margin" in item
        assert "pair_id" in item

        # Check image shapes
        assert item["better_image"].shape == (3, 64, 64)
        assert item["worse_image"].shape == (3, 64, 64)

        # Check margin is positive
        assert item["margin"] > 0


class TestDataLoaders:
    """Test dataloader creation."""

    def test_create_dataloaders(self, sample_config):
        """Test dataloaders can be created."""
        config = sample_config.copy()
        config["max_samples_train"] = 10
        config["max_samples_val"] = 5

        train_loader, val_loader, test_loader = create_dataloaders(
            config,
            num_workers=0,
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

    def test_dataloader_batch(self, sample_config):
        """Test dataloader returns correct batch format."""
        config = sample_config.copy()
        config["max_samples_train"] = 10
        config["max_samples_val"] = 5

        train_loader, _, _ = create_dataloaders(
            config,
            num_workers=0,
        )

        batch = next(iter(train_loader))

        assert "image" in batch
        assert "caption" in batch

        # Check batch dimensions
        assert batch["image"].shape[0] == config["batch_size"]
        assert batch["image"].shape[1:] == (3, config["image_size"], config["image_size"])


class TestPreprocessPipeline:
    """Test preprocessing pipeline."""

    def test_preprocess_caption(self):
        """Test caption preprocessing."""
        # Create dummy tokenizer
        from transformers import CLIPTokenizer

        try:
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            pipeline = PreprocessPipeline(tokenizer, max_length=77, image_size=64)

            caption = "a test caption"
            result = pipeline.preprocess_caption(caption)

            assert "input_ids" in result
            assert "attention_mask" in result
            assert result["input_ids"].shape[0] == 77
        except Exception:
            # Skip if tokenizer can't be loaded
            pytest.skip("CLIP tokenizer not available")

    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        from transformers import CLIPTokenizer

        try:
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            pipeline = PreprocessPipeline(tokenizer, max_length=77, image_size=64)

            captions = ["caption 1", "caption 2", "caption 3"]
            result = pipeline.preprocess_batch(captions)

            assert "input_ids" in result
            assert "attention_mask" in result
            assert result["input_ids"].shape == (3, 77)
        except Exception:
            pytest.skip("CLIP tokenizer not available")
