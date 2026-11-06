# Draft figure captions

## metrics_comparison
Frozen and projection-only CLIP on the official SemArt validation split: 1069 comment queries
ranked against the same 1069 paintings. Recall is reported as a percentage; MRR is unitless.
Both evaluations use float32 ViT-B/32 and tokenizer-truncated comments. Epoch 3 was
selected using validation MRR. Results describe one seed; no test-set or multi-seed claim is made.

## training_progress
Training contrastive loss and full-gallery validation MRR by epoch. Epoch 0 is the matching
frozen baseline and has no training loss. The vertical marker identifies the checkpoint selected
by validation MRR. Training loss and retrieval metrics measure different objectives.

## paired_rank_analysis
Cumulative correct-image ranks and paired per-query ranks before/after adaptation. Each point
represents one validation query; points below the diagonal have improved rank. Both rank axes
in the scatter use logarithmic scales. Identical points can overlap.

## teaser / qualitative examples
Same-query top-three retrievals from frozen and adapted CLIP; the paired painting is shown
separately and marked in green when retrieved. Ranks use the full gallery, not the three displayed
images. Text is an excerpt for display, while model input follows the saved truncation policy.
Examples are selected deterministically at the median rank gain within their category, with
query-ID ordering for ties. The improvement category prefers queries newly correct at rank 1.
These selected cases illustrate behavior and do not replace aggregate results.

Selected query IDs: {"improved": "43477-mancini2.jpg", "unchanged_failure": "19658-campovac.jpg", "worsened": "03767-034madon.jpg"}
