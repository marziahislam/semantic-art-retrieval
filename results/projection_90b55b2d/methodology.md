# Methodology and experimental conditions — draft

## Task and data

We study text-to-image retrieval on SemArt: an artistic comment is the query, and
the output is a ranked gallery of paintings. The official training partition contains
19,244 image-comment pairs. Evaluation uses the official
validation partition with 1,069 queries and the same number of candidate
paintings. Training and validation image filenames are disjoint. No held-out test
results are included in this experiment. Each paired filename defines the designated
positive image; semantically plausible alternative paintings count as incorrect.

## Model and preprocessing

The baseline is pretrained OpenAI CLIP ViT-B/32. The image pipeline converts
to RGB, resizes the shorter side to 224 pixels, center-crops to 224 by 224, converts
pixels to tensors, and applies CLIP's supplied normalization. The text input is the
artistic comment alone; no metadata fields are appended. The CLIP tokenizer uses
77 positions including special tokens, with longer input truncated. Outputs are
512-dimensional image/text vectors, normalized to unit length. Cosine similarity
is their dot product. Both comparison models are evaluated in float32.

## Projection-only adaptation

Only visual.proj (768 by 512) and text_projection (512 by 512) are optimized:
655,360 trainable parameters. Both encoder backbones and the pretrained logit scale
remain frozen. Evaluation-mode backbone behavior is retained while gradients are
enabled for the projections. Paired records are shuffled together. For a batch of B
matched pairs, scaled cosine scores form a B by B matrix. Diagonal pairs are positives;
other in-batch pairs are negatives. The objective averages image-to-text and
text-to-image cross-entropy. It uses the pretrained frozen exponential logit scale.

The run uses AdamW, learning rate 1e-05, weight decay
0.01, batch size 32, gradient-norm clipping at 1.0,
and seed 42. AdamW otherwise uses the library defaults. It completed
3 epochs. Images use CLIP's deterministic evaluation preprocessing, without
additional random augmentation. A one-pair final batch would be skipped because it
has no in-batch negatives; the saved epoch logs record examples seen/skipped.
The saved run's training environment was Python 3.11.18,
PyTorch 2.15.0, and torchvision
0.29.2 on Georgia Tech PACE.
The exact GPU model, elapsed runtime, and peak memory were not recorded and should
not be inferred. The configured epochs are a fixed training budget, not early stopping.

## Evaluation and checkpoint selection

At epoch 0 and after each training epoch, both validation modalities are re-encoded.
Each query is ranked against the entire validation gallery, regardless of encoding
batch size. Rankings use descending cosine scores, with stable ties following gallery
order. Recall@K is the fraction of queries whose paired painting has one-based rank
at most K. MRR is the mean of 1/rank. The primary metrics are Recall@1, Recall@5,
Recall@10, and MRR; median rank is an additional descriptive statistic.

The checkpoint is selected by highest validation MRR, retaining the earlier checkpoint
on an exact tie and allowing epoch 0 to win. Selected epoch: 3.
Restoring that checkpoint reproduced the saved metrics. Thus the before/after
comparison shares preprocessing, precision, input fields, query IDs, gallery, and
ranking rules. This is validation-based model selection rather than an independent
test estimate. It is a single-seed preliminary experiment; no significance or
multi-seed robustness claim is supported.

## Observations and limitations

The standard split has 547 author values shared between
training and validation. There are 78 distinct
description texts shared across them, and 11
validation rows share a description with another validation row. These properties
can introduce shortcuts, ambiguous targets, and false negatives. They were retained
to preserve the official evaluation protocol. Exact duplicate-image content and near
duplicates have not been audited. Artist mentions inside comments have not been masked.
The fraction of comments truncated has not been measured in the saved experiment.

Completed: frozen baseline and projection-only adaptation on the standard split.
Planned: artist-disjoint splits, metadata/name-masking comparisons, and partial/full
encoder adaptation. Artist-disjoint means absent during this project's fine-tuning;
exposure during CLIP pretraining is unknown. These planned experiments must not be
described as completed results.

## References and reuse

Cite the SemArt dataset paper (Garcia and Vogiatzis, 2018) and CLIP (Radford et al.,
2021), using the report's bibliography. The full related-work section still needs
the milestone rubric's required breadth and connections to the proposed experiments.
This is draft wording to edit into the ICLR report, not a complete report section set.
