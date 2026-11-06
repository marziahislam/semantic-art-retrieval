# Milestone evidence: projection_90b55b2d

| Method | Trainable parameters | R@1 (%) | R@5 (%) | R@10 (%) | MRR |
| --- | --- | --- | --- | --- | --- |
| Frozen CLIP | 0 | 17.12 | 38.35 | 49.20 | 0.2787 |
| Projection-only CLIP | 655360 | 22.36 | 47.90 | 58.09 | 0.3459 |

Protocol: ViT-B/32; float32; comment-only text-to-image retrieval;
1069 validation queries and 1069 candidate paintings; truncate=True.
Trained on 19244 pairs with batch size 32,
AdamW learning rate 1e-05, weight decay 0.01, seed 42.
Selected epoch 3 using validation MRR. Restored checkpoint reproduced its metrics.
The parameter count assumes the verified ViT-B/32 projection-only implementation.

R@1 changed by +5.24 percentage points;
MRR changed by +0.0672.
Correct-image rank improved for 620 queries, stayed the same for
214, and worsened for 235.

## Limits on interpretation

- Single-seed preliminary validation results; checkpoint selection uses this validation set.
- These are not held-out test results or evidence of artist-disjoint generalization.
- Shared descriptions can produce ambiguous exact-image targets. Artist overlap is present.
- No measured truncation percentage is available in the saved run; do not invent one.
- No timings, multi-seed error bars, or claims of statistical significance are inferred.
- TYPE summaries are descriptive and include group counts; they are not extra model-selection tests.
- Qualitative examples are selected examples, not an estimate of overall performance.

## Figure use

Main-text candidates: teaser.png and metrics_comparison.png.
Training and paired-rank plots can support the methods/results section or figure appendix.
Use figure_captions.md for the evaluation protocol and qualitative selection rule.
LaTeX table uses booktabs commands. PNG and SVG versions of each figure are available.
