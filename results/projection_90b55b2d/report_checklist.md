# Milestone coverage checklist

| Required component | Available evidence | Remaining author work |
| --- | --- | --- |
| Introduction, scope, input/output | methodology.md; teaser.png | Explain motivation and research goals |
| Teaser/draft figure | teaser.png / teaser.svg; figure_captions.md | Insert and edit caption |
| Related work | Proposal bibliography; method references | At least 5 relevant citations per rubric; explain connections |
| Technical approach and conditions | methodology.md; experiment_evidence.json; source_evidence/ | Adapt wording; distinguish projection adaptation from full fine-tuning |
| CS7643 H1-H3 | Proposal research questions | Write three specific, testable hypotheses; mark untested ones planned |
| Dataset/splits and metrics | methodology.md; source_evidence/split_manifest.json | Explain standard-split overlap and paired-image relevance |
| Baseline and method results | metrics.tex / metrics.csv; metrics_comparison.png | Discuss matched-protocol changes; label validation selection |
| Optimization evidence | training_progress.png; training_history.csv | Describe loss and retrieval trends without equating them |
| Qualitative analysis | teaser and qualitative_*.png; selected_examples.json | Interpret visible versus contextual failures with human review |
| Additional descriptive results | paired_rank_analysis.png; retrieval_transitions.csv; per_query_comparison.csv; descriptive_results_by_type.csv | Choose only useful material for the page limit |
| Next steps | Unfinished experiments listed in methodology.md | Assign dates, milestones, risks, mitigations, and success criteria |
| Format | Exported figures/tables | ICLR template, all authors, 4–5 main-text pages, references excluded; appendix figures only |

This package supports completed experiments. It does not replace the introduction,
related-work analysis, H1-H3 formulation, human interpretation, or project timeline.
GPU model, runtime, peak memory, truncation rate, artist-disjoint/metadata results,
and multi-seed uncertainty were not available in the downloaded evidence.

## Candidate hypotheses to refine

- H1: Updating only CLIP's image/text projections increases validation MRR over
  frozen CLIP with identical query fields, truncation, and gallery. Current evidence:
  one completed standard-split experiment; broader robustness remains untested.
- H2: The adaptation gain in MRR is smaller on an artist-disjoint split than on
  the standard split, holding training/evaluation settings fixed. Status: planned.
- H3: Adding artist metadata yields a larger MRR gain on the standard split than
  on the artist-disjoint split under the same adaptation strategy. Status: planned.

Use methodology_brief.md as a short prose starting point and methodology.md for
the complete experimental conditions and limitations.
