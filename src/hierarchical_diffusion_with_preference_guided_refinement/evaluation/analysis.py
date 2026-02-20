"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzer for experimental results.

    Args:
        results_dir: Directory to save results
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        save_name: str = "training_curves.png",
    ) -> None:
        """Plot training and validation loss curves.

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            save_name: Filename to save plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss", linewidth=2)
        plt.plot(val_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved training curves to {save_path}")

    def plot_metric_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_name: str = "metric_comparison.png",
    ) -> None:
        """Plot comparison of metrics across different models.

        Args:
            metrics_dict: Dictionary mapping model names to metrics
            save_name: Filename to save plot
        """
        models = list(metrics_dict.keys())
        metric_names = list(next(iter(metrics_dict.values())).keys())

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, metric_name in enumerate(metric_names[:4]):
            values = [metrics_dict[model].get(metric_name, 0) for model in models]

            axes[idx].bar(models, values, alpha=0.7)
            axes[idx].set_title(metric_name.replace("_", " ").title())
            axes[idx].set_ylabel("Value")
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        save_path = self.results_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved metric comparison to {save_path}")

    def create_summary_table(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_name: str = "summary.txt",
    ) -> None:
        """Create a formatted summary table of results.

        Args:
            metrics_dict: Dictionary mapping model names to metrics
            save_name: Filename to save summary
        """
        save_path = self.results_dir / save_name

        with open(save_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("EVALUATION RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            for model_name, metrics in metrics_dict.items():
                f.write(f"\n{model_name}:\n")
                f.write("-" * 40 + "\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {metric_name:25s}: {value:8.4f}\n")
                    else:
                        f.write(f"  {metric_name:25s}: {value}\n")

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"Saved summary table to {save_path}")

    def analyze_ablation(
        self,
        baseline_metrics: Dict[str, float],
        ablation_metrics: Dict[str, float],
        save_name: str = "ablation_analysis.txt",
    ) -> None:
        """Analyze ablation study results.

        Args:
            baseline_metrics: Baseline model metrics
            ablation_metrics: Ablation model metrics
            save_name: Filename to save analysis
        """
        save_path = self.results_dir / save_name

        with open(save_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ABLATION STUDY ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            f.write("Metric                    | Baseline  | Ablation  | Change    | % Change\n")
            f.write("-" * 80 + "\n")

            for metric_name in baseline_metrics.keys():
                if metric_name not in ablation_metrics:
                    continue

                baseline_val = baseline_metrics[metric_name]
                ablation_val = ablation_metrics[metric_name]

                if isinstance(baseline_val, (int, float)) and isinstance(ablation_val, (int, float)):
                    change = ablation_val - baseline_val
                    pct_change = (change / baseline_val * 100) if baseline_val != 0 else 0

                    f.write(
                        f"{metric_name:25s} | {baseline_val:9.4f} | {ablation_val:9.4f} | "
                        f"{change:9.4f} | {pct_change:7.2f}%\n"
                    )

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"Saved ablation analysis to {save_path}")


def save_metrics(
    metrics: Dict[str, float],
    save_path: str,
    format: str = "json",
) -> None:
    """Save metrics to file.

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save file
        format: Format ('json' or 'csv')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
    elif format == "csv":
        import csv
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for key, value in metrics.items():
                writer.writerow([key, value])
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Saved metrics to {save_path}")


def load_metrics(load_path: str) -> Dict[str, float]:
    """Load metrics from file.

    Args:
        load_path: Path to metrics file

    Returns:
        Dictionary of metrics
    """
    load_path = Path(load_path)

    if load_path.suffix == ".json":
        with open(load_path, "r") as f:
            metrics = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {load_path.suffix}")

    logger.info(f"Loaded metrics from {load_path}")
    return metrics
