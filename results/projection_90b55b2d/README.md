# Results guide

This folder compares **frozen CLIP** with **projection-only fine-tuned CLIP**.
Both are evaluated on the same **1,069 SemArt validation descriptions and 1,069
candidate paintings**. Training used 19,244 pairs; epoch 3 was selected by validation
MRR. These are single-seed validation results, not held-out test results.

## Start here: the essential report files

| File | What it means | Where to use it |
| --- | --- | --- |
| [teaser.png](teaser.png) | One query and the top-three retrieved paintings before/after fine-tuning. Green borders identify the paired painting. | Introduction: teaser/draft figure |
| [metrics.tex](metrics.tex) | Main results table: Recall@1/5/10 and MRR for both models. Requires LaTeX `booktabs`. | Experiments and Results |
| [training_progress.png](training_progress.png) | Left: training loss by epoch. Right: validation MRR, including the frozen baseline and selected epoch. | Optional supporting results figure |
| [methodology_brief.md](methodology_brief.md) | Short explanation of the model, training conditions, and evaluation. | Starting point for methods prose |
| [figure_captions.md](figure_captions.md) | Draft captions explaining each figure and its limitations. | Adapt when inserting figures |

**Minimum figure/table upload:** `teaser.png` and `metrics.tex`.
Add `training_progress.png` if space permits. The Markdown notes are writing aids.

## Other figures: optional supporting material

| File | What it shows |
| --- | --- |
| [metrics_comparison.png](metrics_comparison.png) | A bar-chart version of the main metrics table. Usually use the table or this chart to avoid repeating the same information. |
| [paired_rank_analysis.png](paired_rank_analysis.png) | Left: percentage of queries retrieved as the allowed rank increases. Right: each query's frozen rank versus adapted rank; points below the diagonal improved. |
| [qualitative_worsened.png](qualitative_worsened.png) | A selected query whose correct painting moved lower after fine-tuning. |
| [qualitative_unchanged_failure.png](qualitative_unchanged_failure.png) | A selected query whose correct painting retained the same rank below first place. “Failure” here means a top-one miss. |

Each figure also has an `.svg` version. PNG is convenient for uploading; SVG
preserves vector labels/plots and embeds the painting images. The two versions
contain the same result, so upload whichever format your report editor supports.
Qualitative examples illustrate individual behavior; they do not measure overall
performance. Query excerpts are shortened for display; model inputs used tokenizer
truncation. All shown ranks refer to the full gallery, even when only three images
are displayed.

## Tables and statistics: for checking or further analysis

| File | What it contains |
| --- | --- |
| [metrics.md](metrics.md) | Readable Markdown version of the main results table. |
| [metrics.csv](metrics.csv) | Spreadsheet-friendly version of the same table. Recall columns are percentages; MRR is a decimal. |
| [rank_statistics.json](rank_statistics.json) | Counts of queries with improved/unchanged/worsened correct-image ranks, median ranks, and overall metric changes. Positive `rank_gain` means improvement in the per-query file. Recall changes are percentage points. |
| [retrieval_transitions.csv](retrieval_transitions.csv) | For each cutoff K: baseline hits, adapted hits, gained hits, and lost hits. `adapted_hits = baseline_hits + gained_hits - lost_hits`. |
| [per_query_comparison.csv](per_query_comparison.csv) | One row per description: before/after correct-image ranks, top-five filenames/similarities, rank changes, and original metadata. Positive reciprocal-rank gain means improvement. |
| [training_history.csv](training_history.csv) | Training loss and validation metrics by epoch. Epoch 0 is the frozen baseline, so its training loss is blank. Recall values here are fractions, not percentages. |
| [descriptive_results_by_type.csv](descriptive_results_by_type.csv) | Results grouped by SemArt TYPE/genre, with group sizes. Recall values are fractions. Small groups should be interpreted cautiously. |
| [selected_examples.json](selected_examples.json) | Query IDs and the deterministic selection rule used for the qualitative figures. |

Metric meanings:

- **Recall@K:** fraction of descriptions whose paired painting appears in the top K.
- **MRR:** average of `1 / correct rank`; higher is better. It is not average rank.
- **Correct rank:** one-based position of the paired painting; lower is better.
- **Cosine similarity:** an image/text compatibility score, not a probability.
  Absolute scores across models should not be interpreted as calibrated confidence.

## Writing notes and reproducibility

| File or folder | Purpose |
| --- | --- |
| [methodology.md](methodology.md) | Detailed task formulation, preprocessing, loss, optimizer settings, checkpoint selection, and limitations. |
| [report_notes.md](report_notes.md) | Main findings and caveats to help write the results discussion. |
| [report_checklist.md](report_checklist.md) | Maps outputs to the milestone rubric and identifies remaining prose/experiments. |
| [experiment_evidence.json](experiment_evidence.json) | Configuration, selected epoch, metrics, update checks, and dataset audit. |
| `source_evidence/` | Small original run files used to regenerate the report outputs: metrics, history, rankings, split IDs/hashes, versions, and update checks. No model weights are included. |

The main notebook's **Milestone report artifacts** section regenerates these files
without training or a GPU. It explicitly selects `RUN_NAME = "projection_90b55b2d"`.
Generated outputs for this run are replaced on regeneration; this README is not.
Full model/optimizer checkpoints remain separately in `artifacts/` and are ignored
by Git.

## Claims to keep separate

Completed: frozen baseline and projection-only adaptation on the official split.
Artist-disjoint evaluation, metadata ablations, and partial/full encoder fine-tuning
are planned. The current results do not establish performance on unseen artists,
statistical significance, or multi-seed robustness. Shared descriptions and author
overlap are documented in the methodology note.
