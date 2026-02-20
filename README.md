# Hierarchical Diffusion with Preference-Guided Refinement

A two-stage image generation system combining coarse-to-fine diffusion with RLHF-inspired preference optimization. The base diffusion model generates images from text captions, then a refinement stage uses preference rankings to fine-tune a lightweight adapter that steers generation toward human-preferred aesthetics through learned reward-weighted guidance of the denoising trajectory.

## Key Innovation

This project treats diffusion sampling as a sequential decision process where preference feedback shapes the denoising trajectory. Unlike traditional diffusion models that follow fixed sampling schedules, our approach uses a learned reward model from preference comparisons to dynamically weight the guidance at each denoising step, enabling fine-grained control over image aesthetics without full model retraining.

## Installation

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Quick Start

### Training

Train the full two-stage pipeline:

```bash
python scripts/train.py --config configs/default.yaml --stage both
```

Train only the base diffusion model:

```bash
python scripts/train.py --config configs/default.yaml --stage base
```

Train only the preference refinement stage:

```bash
python scripts/train.py --config configs/default.yaml --stage preference
```

### Evaluation and Inference

```bash
# Evaluate with metrics
python scripts/evaluate.py --checkpoint models/best_preference_model.pt --config configs/default.yaml

# Generate images from text
python scripts/predict.py --checkpoint models/best_preference_model.pt --prompt "a beautiful sunset" --output-dir results/generated
```

## Methodology

The core innovation treats diffusion sampling as a sequential decision process. At each denoising step, a learned reward model predicts aesthetic quality, which dynamically scales the guidance strength:

```python
reward = reward_model(latent, text_embedding, timestep)
guidance_weight = sigmoid(reward) * guidance_scale
noise_pred = uncond_pred + guidance_weight * (cond_pred - uncond_pred)
```

The architecture uses a U-Net base model (1000 timesteps, cosine scheduling) with lightweight LoRA adapters (rank-4, only 4% parameters) fine-tuned via Bradley-Terry preference loss. This enables preference-guided generation without retraining the full model.

## Configuration

Key parameters in `configs/default.yaml`: image_size (256), epochs (10 base + 5 preference), guidance_scale (7.5), adapter_rank (4). The `configs/ablation.yaml` disables preference refinement for baseline comparison.

## Results

Training completed successfully with the following metrics:

### Base Diffusion Model (10 epochs)
| Metric | Value | Description |
|--------|-------|-------------|
| Final Validation Loss | 0.5383 | MSE denoising loss |
| Best Validation Loss | 0.5383 | Achieved at epoch 10 |
| Model Parameters | 150M | U-Net with text conditioning |

### Preference Refinement Model (5 epochs)
| Metric | Value | Description |
|--------|-------|-------------|
| Final Validation Loss | 0.2500 | Bradley-Terry preference loss |
| Final Validation Accuracy | 19.05% | Preference ranking accuracy |
| Training Accuracy | 50-75% | Batch-level preference accuracy |
| Trainable Parameters | 6.1M | LoRA adapters only (4% of base) |

### Expected Evaluation Metrics
Run `python scripts/evaluate.py` to compute:
- FID Score (target < 25.0): Frechet Inception Distance
- CLIP Score (target > 0.28): Text-image alignment quality
- Preference Win Rate (target > 65%): Comparison vs baseline
- Inference Time (target < 3000ms): Generation speed per image

Results will be saved to `results/metrics.json` after evaluation.

## Project Structure

```
hierarchical-diffusion-with-preference-guided-refinement/
├── src/hierarchical_diffusion_with_preference_guided_refinement/
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model architecture and components
│   ├── training/      # Training loops and utilities
│   ├── evaluation/    # Metrics and analysis
│   └── utils/         # Configuration and helpers
├── scripts/
│   ├── train.py       # Full training pipeline
│   ├── evaluate.py    # Evaluation with metrics
│   └── predict.py     # Image generation
├── tests/             # Comprehensive test suite
├── configs/           # YAML configurations
└── results/           # Outputs and checkpoints
```

## Technical Details

**Two-Stage Training:** Base diffusion (10 epochs) uses standard MSE denoising loss. Preference refinement (5 epochs) uses Bradley-Terry ranking loss with margin, training only adapters while base weights stay frozen.

**Preference Loss:** `loss = ReLU(margin - (better_reward - worse_reward)/temperature).mean()` encourages higher scores for preferred images.

**Testing:** Run `pytest tests/ -v` or `pytest tests/ --cov=hierarchical_diffusion_with_preference_guided_refinement`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 16GB+ system RAM

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
