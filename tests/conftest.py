"""Pytest fixtures for testing."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "seed": 42,
        "image_size": 64,
        "batch_size": 2,
        "hidden_dims": [32, 64],
        "text_embed_dim": 128,
        "learning_rate": 0.001,
        "guidance_scale": 7.5,
        "num_train_timesteps": 100,
        "num_inference_steps": 10,
    }


@pytest.fixture
def sample_images(device):
    """Generate sample images for testing."""
    images = torch.randn(2, 3, 64, 64, device=device)
    return images


@pytest.fixture
def sample_text_embeddings(device):
    """Generate sample text embeddings for testing."""
    embeddings = torch.randn(2, 77, 128, device=device)
    return embeddings


@pytest.fixture
def sample_timesteps(device):
    """Generate sample timesteps for testing."""
    timesteps = torch.randint(0, 100, (2,), device=device)
    return timesteps


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for testing."""
    return tmp_path


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed before each test."""
    torch.manual_seed(42)
    np.random.seed(42)
