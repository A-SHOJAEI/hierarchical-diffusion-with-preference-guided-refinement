#!/usr/bin/env python
"""Evaluation script with multiple metrics."""

import sys
import argparse
import logging
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from hierarchical_diffusion_with_preference_guided_refinement.data import (
    create_dataloaders,
    create_tokenizer,
)
from hierarchical_diffusion_with_preference_guided_refinement.models import (
    HierarchicalDiffusionModel,
)
from hierarchical_diffusion_with_preference_guided_refinement.evaluation import (
    compute_all_metrics,
    ResultsAnalyzer,
    save_metrics,
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
    parser = argparse.ArgumentParser(description="Evaluate trained diffusion model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default=None,
        help="Path to baseline model for comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate",
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def generate_samples(model, test_loader, num_samples, device):
    """Generate samples for evaluation.

    Args:
        model: Trained model
        test_loader: Test data loader
        num_samples: Number of samples to generate
        device: Device to run on

    Returns:
        Tuple of (real_images, generated_images, captions)
    """
    logger.info(f"Generating {num_samples} samples for evaluation...")

    real_images = []
    generated_images = []
    captions = []

    num_generated = 0

    with torch.no_grad():
        for batch in test_loader:
            if num_generated >= num_samples:
                break

            batch_size = batch["image"].shape[0]
            images = batch["image"].to(device)
            batch_captions = batch["caption"]

            # Generate text embeddings (placeholder)
            text_embed = torch.randn(batch_size, 77, 768, device=device)

            # Generate images
            gen_images = model.generate(
                text_embed=text_embed,
                batch_size=batch_size,
                num_inference_steps=50,
                device=device,
            )

            real_images.append(images)
            generated_images.append(gen_images)
            captions.extend(batch_captions)

            num_generated += batch_size
            logger.info(f"Generated {num_generated}/{num_samples} samples")

    # Concatenate batches
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    generated_images = torch.cat(generated_images, dim=0)[:num_samples]
    captions = captions[:num_samples]

    logger.info(f"Sample generation complete")
    return real_images, generated_images, captions


def evaluate_model(
    model,
    test_loader,
    num_samples,
    device,
    baseline_model=None,
):
    """Evaluate model with multiple metrics.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        num_samples: Number of samples
        device: Device to run on
        baseline_model: Optional baseline model for comparison

    Returns:
        Dictionary of metrics
    """
    logger.info("=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)

    # Generate samples
    real_images, generated_images, captions = generate_samples(
        model, test_loader, num_samples, device
    )

    # Generate baseline samples if provided
    baseline_images = None
    if baseline_model is not None:
        logger.info("Generating baseline samples...")
        _, baseline_images, _ = generate_samples(
            baseline_model, test_loader, num_samples, device
        )

    # Compute all metrics
    logger.info("\nComputing metrics...")
    metrics = compute_all_metrics(
        model=model,
        real_images=real_images,
        generated_images=generated_images,
        captions=captions,
        baseline_images=baseline_images,
        device=device,
    )

    return metrics


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO", log_file=f"{args.output_dir}/evaluation.log")

    logger.info("Starting model evaluation")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_seed(config.get("seed", 42))

    # Get device
    device = get_device()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Load baseline model if provided
    baseline_model = None
    if args.baseline_checkpoint:
        logger.info(f"Loading baseline model from {args.baseline_checkpoint}")
        baseline_model = load_model(args.baseline_checkpoint, config, device)

    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        config,
        num_workers=config.get("num_workers", 4),
    )

    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        num_samples=args.num_samples,
        device=device,
        baseline_model=baseline_model,
    )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name:25s}: {value:10.4f}")

    # Save metrics
    save_metrics(metrics, output_dir / "metrics.json", format="json")
    save_metrics(metrics, output_dir / "metrics.csv", format="csv")

    # Create analysis
    analyzer = ResultsAnalyzer(results_dir=output_dir)

    # Create summary table
    analyzer.create_summary_table(
        {"Current Model": metrics},
        save_name="evaluation_summary.txt",
    )

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
