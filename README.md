# Lighter is Better: Geometry-Aware Safety Classification Without Generative Overhead

DeepSafe v3 — Hyperbolic projection + neural classifier for efficient LLM safety guard.

## Structure
- `paper/` — LaTeX source, figures, and compiled PDF
- `scripts/train/` — Training, inference, data relabeling, and CF pair generation
- `src/` — Core implementation (projection head, classifier, losses, encoder)
- `models/deepsafe_v3_8B/seed_42/` — Trained model weights
- `data/processed/intent_labels_v2/` — R3 intent labels + counterfactual pairs
- `embeddings/` — Intent-aligned training labels
- `reports/` — Evaluation results and statistics
- `configs/` — Seed configuration
- `pretrained/` — Placeholder for Qwen3-Embedding-8B (download separately)
