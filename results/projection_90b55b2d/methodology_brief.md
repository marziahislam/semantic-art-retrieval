# Brief methodology draft

We evaluate text-to-painting retrieval on SemArt using pretrained OpenAI CLIP
ViT-B/32. We retain the official training partition
(19,244 pairs) and evaluate on
1,069 validation comments against the same number of paintings.
Queries contain artistic comments alone, truncated to CLIP's 77-token context.
Images use the supplied RGB, resize, center-crop, and normalization pipeline.
Normalized 512-dimensional image and text embeddings are ranked by cosine similarity.

We compare frozen CLIP with projection-only adaptation, updating 655,360 parameters
while freezing both backbones and the logit scale. Adaptation uses bidirectional
image-text contrastive cross-entropy, AdamW (learning rate 1e-05,
weight decay 0.01), batch size 32, gradient-norm
clipping at 1.0, and seed 42 for 3 epochs on PACE. Both comparisons
use float32. The checkpoint is selected by validation MRR (epoch 3)
and its restored metrics are verified. We report Recall@1/5/10 and MRR on the full
validation gallery. These are single-seed validation findings; held-out test,
artist-disjoint, and metadata-ablation results are not yet available.
