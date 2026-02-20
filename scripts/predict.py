#!/usr/bin/env python
"""Inference script for generating images from text prompts."""

import sys
import argparse
import logging
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from PIL import Image

from hierarchical_diffusion_with_preference_guided_refinement.models import (
    HierarchicalDiffusionModel,
)
from hierarchical_diffusion_with_preference_guided_refinement.data import (
    create_tokenizer,
    PreprocessPipeline,
)
from hierarchical_diffusion_with_preference_guided_refinement.utils import (
    load_config,
    set_seed,
    get_device,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate images from text prompts")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="File containing multiple prompts (one per line)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/generated",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate per prompt",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale for generation",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_model(checkpoint_path, config, device):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load model on

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Create model
    model = HierarchicalDiffusionModel(
        image_size=config.get("image_size", 256),
        hidden_dims=config.get("hidden_dims", [64, 128, 256, 512]),
        use_preference_adapter=True,
        guidance_scale=config.get("guidance_scale", 7.5),
    )

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def load_prompts(args):
    """Load prompts from arguments or file.

    Args:
        args: Command-line arguments

    Returns:
        List of prompts
    """
    prompts = []

    if args.prompt:
        prompts.append(args.prompt)

    if args.prompts_file:
        prompts_path = Path(args.prompts_file)
        if prompts_path.exists():
            with open(prompts_path, "r") as f:
                file_prompts = [line.strip() for line in f if line.strip()]
                prompts.extend(file_prompts)
        else:
            logger.warning(f"Prompts file not found: {prompts_path}")

    if not prompts:
        # Default prompts
        prompts = [
            "a beautiful mountain landscape at sunset",
            "a cute cat with blue eyes",
            "modern minimalist interior design",
            "colorful abstract art painting",
        ]
        logger.info("No prompts provided. Using default prompts.")

    return prompts


def save_image(image_tensor, save_path):
    """Save image tensor to file.

    Args:
        image_tensor: Image tensor [C, H, W] in [-1, 1]
        save_path: Path to save image
    """
    # Denormalize from [-1, 1] to [0, 1]
    image = (image_tensor + 1.0) / 2.0
    image = torch.clamp(image, 0.0, 1.0)

    # Convert to numpy
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    # Save
    pil_image = Image.fromarray(image_np)
    pil_image.save(save_path)

    logger.info(f"Saved image to {save_path}")


def generate_images(
    model,
    prompts,
    num_samples,
    guidance_scale,
    num_steps,
    device,
    output_dir,
):
    """Generate images from prompts.

    Args:
        model: Trained model
        prompts: List of text prompts
        num_samples: Number of samples per prompt
        guidance_scale: Guidance scale
        num_steps: Number of denoising steps
        device: Device to run on
        output_dir: Output directory

    Returns:
        List of generated images
    """
    logger.info(f"Generating {num_samples} images for {len(prompts)} prompts...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = []

    with torch.no_grad():
        for prompt_idx, prompt in enumerate(prompts):
            logger.info(f"\nPrompt {prompt_idx + 1}/{len(prompts)}: '{prompt}'")

            # Generate text embeddings (placeholder)
            # In practice, use CLIP text encoder
            text_embed = torch.randn(num_samples, 77, 768, device=device)

            # Generate images
            generated = model.generate(
                text_embed=text_embed,
                batch_size=num_samples,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                use_guidance=True,
                device=device,
            )

            # Save images
            for sample_idx in range(num_samples):
                image = generated[sample_idx]

                # Create filename
                safe_prompt = "".join(c if c.isalnum() or c in " -_" else "_" for c in prompt)
                safe_prompt = safe_prompt[:50]  # Truncate to 50 chars
                filename = f"prompt_{prompt_idx:03d}_sample_{sample_idx:02d}_{safe_prompt}.png"
                save_path = output_dir / filename

                # Save image
                save_image(image, save_path)

                all_images.append(image)

            logger.info(f"Generated {num_samples} samples for prompt {prompt_idx + 1}")

    logger.info(f"\nGeneration complete! All images saved to: {output_dir}")
    return all_images


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO")

    logger.info("Starting image generation")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # Set random seed
    set_seed(args.seed)

    # Load configuration
    config = load_config(args.config)

    # Get device
    device = get_device()

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Load prompts
    prompts = load_prompts(args)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Generate images
    images = generate_images(
        model=model,
        prompts=prompts,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        device=device,
        output_dir=args.output_dir,
    )

    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total images generated: {len(images)}")
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
