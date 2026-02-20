#!/usr/bin/env python
"""Full training pipeline for hierarchical diffusion with preference guidance.

This script trains the model in two stages:
1. Base diffusion model training on image-caption pairs
2. Preference-guided refinement with RLHF-inspired optimization
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from hierarchical_diffusion_with_preference_guided_refinement.data import (
    create_dataloaders,
    PreferenceDataset,
    create_tokenizer,
)
from hierarchical_diffusion_with_preference_guided_refinement.models import (
    HierarchicalDiffusionModel,
    PreferenceLoss,
)
from hierarchical_diffusion_with_preference_guided_refinement.training import (
    DiffusionTrainer,
    PreferenceTrainer,
    EarlyStopping,
)
from hierarchical_diffusion_with_preference_guided_refinement.utils import (
    load_config,
    save_config,
    set_seed,
    get_device,
    setup_logging,
    count_parameters,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train hierarchical diffusion model with preference guidance"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["base", "preference", "both"],
        default="both",
        help="Training stage to run",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for checkpoints",
    )
    return parser.parse_args()


def train_base_model(config, device, output_dir):
    """Train base diffusion model.

    Args:
        config: Configuration dictionary
        device: Device to train on
        output_dir: Output directory for checkpoints

    Returns:
        Trained model
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: Training Base Diffusion Model")
    logger.info("=" * 80)

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        config,
        num_workers=config.get("num_workers", 4),
    )

    # Create model
    model = HierarchicalDiffusionModel(
        image_size=config.get("image_size", 256),
        hidden_dims=config.get("hidden_dims", [64, 128, 256, 512]),
        use_preference_adapter=False,  # Train base model first
        guidance_scale=config.get("guidance_scale", 7.5),
    )
    model.to(device)

    # Count parameters
    count_parameters(model)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 0.0001),
        weight_decay=config.get("weight_decay", 0.01),
    )

    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.get("epochs", 10),
        eta_min=config.get("min_lr", 0.000001),
    )

    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=config.get("mixed_precision", True),
        gradient_clip=config.get("gradient_clip", 1.0),
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get("patience", 5),
        min_delta=config.get("min_delta", 0.0001),
        mode="min",
    )

    # Training loop
    best_val_loss = float("inf")
    epochs = config.get("epochs", 10)

    try:
        # MLflow tracking (optional)
        try:
            import mlflow
            mlflow.start_run(run_name="base_diffusion")
            mlflow.log_params({
                "model": "base_diffusion",
                "learning_rate": config.get("learning_rate"),
                "batch_size": config.get("batch_size"),
                "epochs": epochs,
            })
            use_mlflow = True
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
            use_mlflow = False

        for epoch in range(1, epochs + 1):
            logger.info(f"\nEpoch {epoch}/{epochs}")
            logger.info("-" * 40)

            # Train
            train_metrics = trainer.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = trainer.validate(val_loader, epoch)

            # Update scheduler
            if scheduler is not None:
                scheduler.step()

            # Log metrics
            if use_mlflow:
                try:
                    mlflow.log_metrics(
                        {**train_metrics, **val_metrics},
                        step=epoch,
                    )
                except:
                    pass

            # Save best model
            val_loss = val_metrics["val_loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = Path(output_dir) / "best_base_model.pt"
                trainer.save_checkpoint(save_path, epoch, val_metrics)
                logger.info(f"New best model saved (val_loss={val_loss:.4f})")

            # Check early stopping
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Save final model
        final_path = Path(output_dir) / "final_base_model.pt"
        trainer.save_checkpoint(final_path, epoch, val_metrics)

        if use_mlflow:
            try:
                mlflow.end_run()
            except:
                pass

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        save_path = Path(output_dir) / "interrupted_base_model.pt"
        trainer.save_checkpoint(save_path, epoch, val_metrics)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("\nBase model training completed!")
    return model


def train_preference_model(config, device, output_dir, base_model=None):
    """Train preference-guided refinement model.

    Args:
        config: Configuration dictionary
        device: Device to train on
        output_dir: Output directory for checkpoints
        base_model: Pre-trained base model (optional)

    Returns:
        Trained model with preference adapter
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: Training Preference-Guided Refinement")
    logger.info("=" * 80)

    # Create or load model
    if base_model is None:
        # Load from checkpoint
        base_checkpoint = Path(output_dir) / "best_base_model.pt"
        if base_checkpoint.exists():
            logger.info(f"Loading base model from {base_checkpoint}")
            model = HierarchicalDiffusionModel(
                image_size=config.get("image_size", 256),
                hidden_dims=config.get("hidden_dims", [64, 128, 256, 512]),
                use_preference_adapter=True,
                guidance_scale=config.get("guidance_scale", 7.5),
            )
            checkpoint = torch.load(base_checkpoint, map_location=device)
            # Load only base model weights
            model.base_model.load_state_dict(
                {k.replace("base_model.", ""): v for k, v in checkpoint["model_state_dict"].items()
                 if k.startswith("base_model.")},
                strict=False
            )
            model.to(device)
        else:
            logger.warning("No base model checkpoint found. Training from scratch.")
            model = HierarchicalDiffusionModel(
                image_size=config.get("image_size", 256),
                hidden_dims=config.get("hidden_dims", [64, 128, 256, 512]),
                use_preference_adapter=True,
                guidance_scale=config.get("guidance_scale", 7.5),
            )
            model.to(device)
    else:
        # Add preference adapter to base model
        model = HierarchicalDiffusionModel(
            image_size=config.get("image_size", 256),
            hidden_dims=config.get("hidden_dims", [64, 128, 256, 512]),
            use_preference_adapter=True,
            guidance_scale=config.get("guidance_scale", 7.5),
        )
        model.base_model.load_state_dict(base_model.base_model.state_dict())
        model.to(device)

    # Count trainable parameters
    count_parameters(model)

    # Create preference dataset
    preference_dataset = PreferenceDataset(
        num_pairs=config.get("preference_pairs", 5000),
        image_size=config.get("image_size", 256),
    )

    preference_loader = torch.utils.data.DataLoader(
        preference_dataset,
        batch_size=config.get("preference_batch_size", 8),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    # Validation split (use 10% for validation)
    val_size = len(preference_dataset) // 10
    train_size = len(preference_dataset) - val_size
    train_pref, val_pref = torch.utils.data.random_split(
        preference_dataset,
        [train_size, val_size],
    )

    train_pref_loader = torch.utils.data.DataLoader(
        train_pref,
        batch_size=config.get("preference_batch_size", 8),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
    )

    val_pref_loader = torch.utils.data.DataLoader(
        val_pref,
        batch_size=config.get("preference_batch_size", 8),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
    )

    # Get trainable parameters (only adapter and guidance)
    trainable_params = model.get_trainable_parameters()

    # Create optimizer (only for trainable parameters)
    optimizer = AdamW(
        trainable_params,
        lr=config.get("preference_lr", 0.00005),
        weight_decay=config.get("weight_decay", 0.01),
    )

    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.get("preference_epochs", 5),
        eta_min=config.get("min_lr", 0.000001),
    )

    # Create preference loss
    preference_loss = PreferenceLoss(
        margin=config.get("preference_margin", 0.5),
        temperature=config.get("preference_temperature", 1.0),
    )

    # Create trainer
    trainer = PreferenceTrainer(
        model=model,
        optimizer=optimizer,
        preference_loss=preference_loss,
        scheduler=scheduler,
        device=device,
        mixed_precision=config.get("mixed_precision", True),
        gradient_clip=config.get("gradient_clip", 1.0),
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get("patience", 3),
        min_delta=config.get("min_delta", 0.001),
        mode="min",
    )

    # Training loop
    best_val_loss = float("inf")
    epochs = config.get("preference_epochs", 5)

    try:
        # MLflow tracking
        try:
            import mlflow
            mlflow.start_run(run_name="preference_refinement")
            mlflow.log_params({
                "model": "preference_guided",
                "learning_rate": config.get("preference_lr"),
                "batch_size": config.get("preference_batch_size"),
                "epochs": epochs,
            })
            use_mlflow = True
        except:
            use_mlflow = False

        for epoch in range(1, epochs + 1):
            logger.info(f"\nEpoch {epoch}/{epochs}")
            logger.info("-" * 40)

            # Train
            train_metrics = trainer.train_epoch(train_pref_loader, epoch)

            # Validate
            val_metrics = trainer.validate(val_pref_loader, epoch)

            # Update scheduler
            if scheduler is not None:
                scheduler.step()

            # Log metrics
            if use_mlflow:
                try:
                    mlflow.log_metrics(
                        {**train_metrics, **val_metrics},
                        step=epoch,
                    )
                except:
                    pass

            # Save best model
            val_loss = val_metrics["val_loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = Path(output_dir) / "best_preference_model.pt"
                trainer.save_checkpoint(save_path, epoch, val_metrics)
                logger.info(f"New best model saved (val_loss={val_loss:.4f})")

            # Check early stopping
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Save final model
        final_path = Path(output_dir) / "final_preference_model.pt"
        trainer.save_checkpoint(final_path, epoch, val_metrics)

        if use_mlflow:
            try:
                mlflow.end_run()
            except:
                pass

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        save_path = Path(output_dir) / "interrupted_preference_model.pt"
        trainer.save_checkpoint(save_path, epoch, val_metrics)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("\nPreference model training completed!")
    return model


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO", log_file="results/training.log")

    logger.info("Starting hierarchical diffusion training")
    logger.info(f"Configuration file: {args.config}")

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_seed(config.get("seed", 42))

    # Get device
    device = get_device()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(config, output_dir / "config.yaml")

    # Train based on stage
    if args.stage in ["base", "both"]:
        base_model = train_base_model(config, device, output_dir)
    else:
        base_model = None

    if args.stage in ["preference", "both"]:
        train_preference_model(config, device, output_dir, base_model)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Models saved to: {output_dir}")

    # Save final training results as JSON for downstream evaluation
    import json
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    final_results = {
        "stage": args.stage,
        "config": args.config,
        "output_dir": str(output_dir),
        "status": "completed",
    }

    # Load metrics from best checkpoints if they exist
    for ckpt_name in ["best_base_model.pt", "best_preference_model.pt"]:
        ckpt_path = output_dir / ckpt_name
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                metrics = ckpt.get("metrics", {})
                stage_key = "base" if "base" in ckpt_name else "preference"
                final_results[f"{stage_key}_metrics"] = metrics
                final_results[f"{stage_key}_epoch"] = ckpt.get("epoch", -1)
            except Exception as e:
                logger.warning(f"Could not load checkpoint {ckpt_name}: {e}")

    results_path = results_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Saved training results to {results_path}")


if __name__ == "__main__":
    main()
