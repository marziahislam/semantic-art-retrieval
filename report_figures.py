"""Report artifacts from saved SemArt runs; no model or GPU required."""
from pathlib import Path
import hashlib
import json
import shutil
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

METRICS = ('recall@1', 'recall@5', 'recall@10', 'mrr')
COLORS = ('#64748b', '#087f8c')


def load_report_run(project_root, run_name):
    """Validate source files and align results by artwork ID before plotting."""
    root = Path(project_root)
    run = root / 'artifacts' / run_name
    if not (run / 'summary.json').is_file():
        run = root / 'results' / run_name / 'source_evidence'
    assert (run / 'summary.json').is_file(), 'Download the run or retain its source_evidence folder.'
    summary = json.loads((run / 'summary.json').read_text())
    history = pd.DataFrame(json.loads((run / 'history.json').read_text()))
    manifest = json.loads((run / 'split_manifest.json').read_text())
    provenance = json.loads((run / 'provenance.json').read_text())
    assert summary['config'] == json.loads((run / 'config.json').read_text())
    assert summary['config']['checkpoint'] == 'ViT-B/32'
    assert summary['config']['adaptation'] == 'projection_only'
    assert set(manifest['train_ids']).isdisjoint(manifest['val_ids'])
    for split, expected in manifest['metadata_sha256'].items():
        source = root / 'SemArt' / f'semart_{split}.csv'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, f'Metadata changed: {split}'
    assert summary['checkpoint_reproduced'], 'Checkpoint was not verified.'
    assert not summary['config']['smoke_test'], 'Select a complete run for this report.'
    assert not history['epoch'].duplicated().any()
    history = history.sort_values('epoch').reset_index(drop=True)
    assert history['epoch'].tolist() == list(range(int(history['epoch'].max()) + 1))
    assert np.isfinite(history[list(METRICS)].to_numpy()).all()
    assert history['num_queries'].eq(len(manifest['val_ids'])).all()
    assert history['gallery_size'].eq(len(manifest['val_ids'])).all()
    assert (history[list(METRICS)].ge(0) & history[list(METRICS)].le(1)).all().all()
    assert summary['best_epoch'] == int(history.loc[history['mrr'].idxmax(), 'epoch'])
    for key, epoch in [('baseline_metrics', 0), ('best_metrics', summary['best_epoch'])]:
        matching = history[history['epoch'] == epoch]
        assert len(matching) == 1
        for metric in METRICS:
            assert abs(float(matching.iloc[0][metric]) - summary[key][metric]) < 1e-6

    metadata = pd.read_csv(root / 'SemArt' / 'semart_val.csv', sep='\t', encoding='cp1252')
    assert metadata['IMAGE_FILE'].is_unique
    tables = {}
    for name, filename, metric_key in (
        ('baseline', 'baseline_rankings.csv', 'baseline_metrics'),
        ('adapted', 'best_rankings.csv', 'best_metrics'),
    ):
        table = pd.read_csv(run / filename)
        assert table['query_id'].is_unique and table['query_id'].notna().all()
        assert set(table['query_id']) == set(manifest['val_ids'])
        ranks = table['correct_rank'].to_numpy(dtype=float)
        count = summary[metric_key]['num_queries']
        gallery_size = summary[metric_key]['gallery_size']
        assert len(table) == count == gallery_size == len(manifest['val_ids'])
        assert np.isfinite(ranks).all() and (ranks == np.floor(ranks)).all()
        assert ((ranks >= 1) & (ranks <= gallery_size)).all()
        calculated = {f'recall@{k}': float(np.mean(ranks <= k)) for k in (1, 5, 10)}
        calculated['mrr'] = float(np.mean(1 / ranks))
        for metric, value in calculated.items():
            assert abs(value - summary[metric_key][metric]) < 1e-6, (name, metric)
        for row in table.itertuples():
            ids = json.loads(row.top5_image_ids)
            scores = json.loads(row.top5_similarities)
            assert len(ids) == len(scores) == min(5, gallery_size)
            assert len(set(ids)) == len(ids) and set(ids) <= set(manifest['val_ids'])
            assert np.isfinite(scores).all()
            assert all(a >= b for a, b in zip(scores, scores[1:]))
            if row.correct_rank <= len(ids):
                assert ids[int(row.correct_rank) - 1] == row.query_id
            else:
                assert row.query_id not in ids
        tables[name] = table

    paired = tables['baseline'].merge(
        tables['adapted'], on='query_id', suffixes=('_baseline', '_adapted'),
        validate='one_to_one',
    ).merge(metadata, left_on='query_id', right_on='IMAGE_FILE', validate='one_to_one')
    assert len(paired) == len(manifest['val_ids']) == len(metadata)
    paired['rank_gain'] = paired['correct_rank_baseline'] - paired['correct_rank_adapted']
    paired['reciprocal_rank_gain'] = (
        1 / paired['correct_rank_adapted'] - 1 / paired['correct_rank_baseline']
    )
    images_root = root / 'SemArt' / 'Images'
    assert all((images_root / filename).is_file() for filename in manifest['val_ids'])
    out = root / 'results' / run_name
    out.mkdir(parents=True, exist_ok=True)
    return dict(root=root, run=run, run_name=run_name, out=out, summary=summary, history=history,
                paired=paired, metadata=metadata.set_index('IMAGE_FILE'),
                images_root=images_root, provenance=provenance, manifest=manifest)


def export_report_tables(report):
    """Save numerical evidence, a LaTeX table, and report-ready factual notes."""
    s, paired, out = report['summary'], report['paired'], report['out']
    rows = []
    for method, key, count in [('Frozen CLIP', 'baseline_metrics', 0),
                               ('Projection-only CLIP', 'best_metrics', 655360)]:
        m = s[key]
        rows.append({'Method': method, 'Trainable parameters': count,
                     'R@1 (%)': 100*m['recall@1'], 'R@5 (%)': 100*m['recall@5'],
                     'R@10 (%)': 100*m['recall@10'], 'MRR': m['mrr']})
    table = pd.DataFrame(rows)
    table.to_csv(out / 'metrics.csv', index=False)
    table.to_latex(out / 'metrics.tex', index=False, escape=True, formatters={
        'R@1 (%)': '{:.2f}'.format, 'R@5 (%)': '{:.2f}'.format,
        'R@10 (%)': '{:.2f}'.format, 'MRR': '{:.4f}'.format,
    })
    formatted = table.copy()
    for col in ['R@1 (%)', 'R@5 (%)', 'R@10 (%)']:
        formatted[col] = formatted[col].map(lambda value: f'{value:.2f}')
    formatted['MRR'] = formatted['MRR'].map(lambda value: f'{value:.4f}')
    header = '| ' + ' | '.join(formatted.columns) + ' |\n'
    markdown = header + '| ' + ' | '.join(['---']*len(formatted.columns)) + ' |\n'
    markdown += '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in formatted.itertuples(index=False, name=None))
    (out / 'metrics.md').write_text(markdown + '\n')

    baseline, adapted = paired['correct_rank_baseline'], paired['correct_rank_adapted']
    counts = {'queries': len(paired),
              'improved_rank': int((adapted < baseline).sum()),
              'unchanged_rank': int((adapted == baseline).sum()),
              'worsened_rank': int((adapted > baseline).sum()),
              'median_rank_baseline': float(baseline.median()),
              'median_rank_adapted': float(adapted.median())}
    transitions = []
    for k in (1, 5, 10):
        before, after = baseline <= k, adapted <= k
        transitions.append({'K': k, 'baseline_hits': int(before.sum()), 'adapted_hits': int(after.sum()),
                            'gained_hits': int((~before & after).sum()),
                            'lost_hits': int((before & ~after).sum())})
        counts[f'recall@{k}_change_percentage_points'] = 100*float(after.mean()-before.mean())
    counts['mrr_change'] = float(paired['reciprocal_rank_gain'].mean())
    pd.DataFrame(transitions).to_csv(out / 'retrieval_transitions.csv', index=False)
    (out / 'rank_statistics.json').write_text(json.dumps(counts, indent=2))
    paired.to_csv(out / 'per_query_comparison.csv', index=False)
    report['history'].to_csv(out / 'training_history.csv', index=False)

    # Group summaries are descriptive; small groups remain visible with their counts.
    by_type = []
    for group, frame in paired.groupby('TYPE', dropna=False):
        before, after = frame['correct_rank_baseline'], frame['correct_rank_adapted']
        by_type.append({'type': str(group), 'n': len(frame),
                        'baseline_r1': float((before<=1).mean()), 'adapted_r1': float((after<=1).mean()),
                        'baseline_mrr': float((1/before).mean()), 'adapted_mrr': float((1/after).mean())})
    pd.DataFrame(by_type).to_csv(out / 'descriptive_results_by_type.csv', index=False)
    cfg = s['config']
    notes = f'''# Milestone evidence: {report['run_name']}

{markdown}

Protocol: {cfg['checkpoint']}; float32; comment-only text-to-image retrieval;
{len(paired)} validation queries and {len(paired)} candidate paintings; truncate={cfg['truncate']}.
Trained on {len(report['manifest']['train_ids'])} pairs with batch size {cfg['batch_size']},
AdamW learning rate {cfg['learning_rate']}, weight decay {cfg['weight_decay']}, seed {cfg['seed']}.
Selected epoch {s['best_epoch']} using validation MRR. Restored checkpoint reproduced its metrics.
The parameter count assumes the verified ViT-B/32 projection-only implementation.

R@1 changed by {counts['recall@1_change_percentage_points']:+.2f} percentage points;
MRR changed by {counts['mrr_change']:+.4f}.
Correct-image rank improved for {counts['improved_rank']} queries, stayed the same for
{counts['unchanged_rank']}, and worsened for {counts['worsened_rank']}.

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
'''
    (out / 'report_notes.md').write_text(notes)
    evidence = {key: s[key] for key in ('config', 'best_epoch', 'baseline_metrics', 'best_metrics', 'checkpoint_reproduced', 'update_checks')}
    evidence['source_run'] = report['run_name']
    evidence['dataset_audit'] = report['provenance']['dataset_audit']
    (out / 'experiment_evidence.json').write_text(json.dumps(evidence, indent=2))
    # Preserve the small raw evidence in Git; large model checkpoints remain ignored.
    evidence_root = out / 'source_evidence'
    evidence_root.mkdir(exist_ok=True)
    for filename in ('summary.json', 'config.json', 'history.json', 'split_manifest.json',
                     'provenance.json', 'update_checks.json', 'baseline_rankings.csv', 'best_rankings.csv'):
        source, destination = report['run'] / filename, evidence_root / filename
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    write_methodology(report)
    return table, counts


def _save_figure(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(out / f'{name}.svg', bbox_inches='tight', facecolor='white')
    return fig


def plot_report_metrics(report):
    s = report['summary']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained', gridspec_kw={'width_ratios':[3,1]})
    x = np.arange(3)
    for j, (name, key) in enumerate([('Frozen', 'baseline_metrics'), ('Projection-only', 'best_metrics')]):
        values = [100*s[key][metric] for metric in METRICS[:3]]
        bars = axes[0].bar(x+(j-.5)*.35, values, .35, color=COLORS[j], label=name)
        axes[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
        bars = axes[1].bar(j, s[key]['mrr'], color=COLORS[j], width=.65)
        axes[1].bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
    axes[0].set(xticks=x, xticklabels=['Recall@1','Recall@5','Recall@10'], ylabel='Queries retrieved (%)', ylim=(0,100))
    axes[0].legend(frameon=False, loc='upper left')
    axes[1].set(xticks=[0,1], xticklabels=['Frozen','Adapted'], ylabel='MRR', ylim=(0,1))
    fig.suptitle(f"SemArt validation | {s['best_metrics']['gallery_size']:,}-painting gallery")
    return _save_figure(fig, report['out'], 'metrics_comparison')


def plot_report_training(report):
    h = report['history']
    fig, axes = plt.subplots(1,2,figsize=(9,3.4), layout='constrained')
    trained = h[h['train_loss'].notna()]
    axes[0].plot(trained['epoch'], trained['train_loss'], 'o-', color=COLORS[1])
    axes[0].set(xlabel='Epoch', ylabel='Training contrastive loss', title='Optimization', xticks=h['epoch'])
    axes[1].plot(h['epoch'], h['mrr'], 'o-', color=COLORS[1])
    axes[1].axhline(h.iloc[0]['mrr'], color=COLORS[0], ls='--', label='Epoch 0: frozen')
    axes[1].axvline(report['summary']['best_epoch'], color='#bd6b21', ls=':', label='Selected epoch')
    axes[1].set(xlabel='Epoch', ylabel='Validation MRR', title='Full-gallery retrieval', xticks=h['epoch'], ylim=(0,1))
    axes[1].legend(frameon=False, fontsize=8)
    return _save_figure(fig, report['out'], 'training_progress')


def plot_report_ranks(report):
    p = report['paired']; n = len(p)
    fig, axes = plt.subplots(1,2,figsize=(9,3.7), layout='constrained')
    for column,label,color in [('correct_rank_baseline','Frozen',COLORS[0]),('correct_rank_adapted','Projection-only',COLORS[1])]:
        values = np.sort(p[column].to_numpy())
        axes[0].step(values, 100*np.arange(1,n+1)/n, where='post', label=label, color=color)
    axes[0].set(xscale='log', xlabel='Correct painting rank (log scale)', ylabel='Cumulative queries (%)', ylim=(0,100), title='Retrieval across all rank cutoffs')
    axes[0].legend(frameon=False)
    axes[1].scatter(p['correct_rank_baseline'],p['correct_rank_adapted'],s=9,alpha=.3,color=COLORS[1])
    axes[1].plot([1,n],[1,n],ls='--',color=COLORS[0])
    axes[1].set(xscale='log',yscale='log',xlabel='Frozen rank (log scale)',ylabel='Adapted rank (log scale)',title='Below diagonal = improved rank',xlim=(.8,n*1.1),ylim=(.8,n*1.1))
    return _save_figure(fig, report['out'], 'paired_rank_analysis')


def select_report_examples(report):
    """Deterministic, median-gain examples rather than the most extreme wins."""
    p = report['paired']
    groups = {
        'improved': p[(p.correct_rank_baseline>1)&(p.correct_rank_adapted==1)],
        'worsened': p[p.correct_rank_adapted>p.correct_rank_baseline],
        'unchanged_failure': p[(p.correct_rank_adapted==p.correct_rank_baseline)&(p.correct_rank_adapted>1)],
    }
    if groups['improved'].empty:
        groups['improved'] = p[p.rank_gain>0]
    selected = {}
    for category, frame in groups.items():
        if len(frame):
            ordered = frame.sort_values(['rank_gain','query_id'])
            selected[category] = str(ordered.iloc[len(ordered)//2]['query_id'])
    if not selected:
        # A no-change run still needs an honestly labeled teaser.
        selected['unchanged_success'] = str(p.sort_values('query_id').iloc[len(p)//2]['query_id'])
    (report['out']/'selected_examples.json').write_text(json.dumps({
        'rule':'Median rank gain within each category; ties ordered by query ID. Improved prefers newly correct top-1 queries.',
        'queries':selected,
    },indent=2))
    return selected


def write_methodology(report):
    """Draft method/conditions from the saved run, with unfinished work explicit."""
    s, h, out = report['summary'], report['history'], report['out']
    cfg, audit = s['config'], report['provenance']['dataset_audit']
    epochs = int(h['epoch'].max())
    brief = f'''# Brief methodology draft

We evaluate text-to-painting retrieval on SemArt using pretrained OpenAI CLIP
{cfg['checkpoint']}. We retain the official training partition
({len(report['manifest']['train_ids']):,} pairs) and evaluate on
{len(report['paired']):,} validation comments against the same number of paintings.
Queries contain artistic comments alone, truncated to CLIP's 77-token context.
Images use the supplied RGB, resize, center-crop, and normalization pipeline.
Normalized 512-dimensional image and text embeddings are ranked by cosine similarity.

We compare frozen CLIP with projection-only adaptation, updating 655,360 parameters
while freezing both backbones and the logit scale. Adaptation uses bidirectional
image-text contrastive cross-entropy, AdamW (learning rate {cfg['learning_rate']},
weight decay {cfg['weight_decay']}), batch size {cfg['batch_size']}, gradient-norm
clipping at 1.0, and seed {cfg['seed']} for {epochs} epochs on PACE. Both comparisons
use float32. The checkpoint is selected by validation MRR (epoch {s['best_epoch']})
and its restored metrics are verified. We report Recall@1/5/10 and MRR on the full
validation gallery. These are single-seed validation findings; held-out test,
artist-disjoint, and metadata-ablation results are not yet available.
'''
    (out / 'methodology_brief.md').write_text(brief)
    methods = f'''# Methodology and experimental conditions — draft

## Task and data

We study text-to-image retrieval on SemArt: an artistic comment is the query, and
the output is a ranked gallery of paintings. The official training partition contains
{len(report['manifest']['train_ids']):,} image-comment pairs. Evaluation uses the official
validation partition with {len(report['paired']):,} queries and the same number of candidate
paintings. Training and validation image filenames are disjoint. No held-out test
results are included in this experiment. Each paired filename defines the designated
positive image; semantically plausible alternative paintings count as incorrect.

## Model and preprocessing

The baseline is pretrained OpenAI CLIP {cfg['checkpoint']}. The image pipeline converts
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

The run uses AdamW, learning rate {cfg['learning_rate']}, weight decay
{cfg['weight_decay']}, batch size {cfg['batch_size']}, gradient-norm clipping at 1.0,
and seed {cfg['seed']}. AdamW otherwise uses the library defaults. It completed
{epochs} epochs. Images use CLIP's deterministic evaluation preprocessing, without
additional random augmentation. A one-pair final batch would be skipped because it
has no in-batch negatives; the saved epoch logs record examples seen/skipped.
The saved run's training environment was Python {report['provenance']['python'].split()[0]},
PyTorch {report['provenance']['packages']['torch']}, and torchvision
{report['provenance']['packages']['torchvision']} on Georgia Tech PACE.
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
on an exact tie and allowing epoch 0 to win. Selected epoch: {s['best_epoch']}.
Restoring that checkpoint reproduced the saved metrics. Thus the before/after
comparison shares preprocessing, precision, input fields, query IDs, gallery, and
ranking rules. This is validation-based model selection rather than an independent
test estimate. It is a single-seed preliminary experiment; no significance or
multi-seed robustness claim is supported.

## Observations and limitations

The standard split has {audit['shared_author_values']} author values shared between
training and validation. There are {audit['shared_description_values']} distinct
description texts shared across them, and {audit['val_rows_with_shared_description']}
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
'''
    (out / 'methodology.md').write_text(methods)
    coverage = '''# Milestone coverage checklist

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
'''
    (out / 'report_checklist.md').write_text(coverage)


def plot_retrieval_example(report, query_id, name='teaser', category='selected example'):
    row = report['paired'].set_index('query_id').loc[query_id]
    metadata = report['metadata']
    fig = plt.figure(figsize=(12,7.5))
    grid = fig.add_gridspec(2,4,left=.07,right=.99,bottom=.10,top=.70,wspace=.12,hspace=.5)
    fig.suptitle('Semantic art retrieval: frozen CLIP and projection-only adaptation',y=.99,fontsize=15)
    fig.text(.07,.945,'Description → CLIP text embedding → cosine ranking of painting embeddings',fontsize=10,color='#334155')
    excerpt = textwrap.shorten(' '.join(str(row['DESCRIPTION']).split()),width=370,placeholder=' …')
    fig.text(.07,.895,'Query excerpt (display only):\n'+textwrap.fill(excerpt,width=125),va='top',fontsize=10,linespacing=1.4)

    def show_image(ax, filename, title, correct=False):
        with Image.open(report['images_root']/filename) as source:
            image=source.convert('RGB')
        ax.imshow(image)
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True);spine.set_color('#15803d' if correct else '#cbd5e1');spine.set_linewidth(3 if correct else 1)
        ax.set_title(title,fontsize=9,pad=7)
    ax=fig.add_subplot(grid[:,0])
    show_image(ax,query_id,'Paired painting\n'+textwrap.fill(str(row['TITLE']),26),True)
    for i,(label,suffix) in enumerate([('Frozen','baseline'),('Adapted','adapted')]):
        ids=json.loads(row[f'top5_image_ids_{suffix}'])[:3]
        for j,filename in enumerate(ids):
            ax=fig.add_subplot(grid[i,j+1])
            title=textwrap.shorten(str(metadata.loc[filename,'TITLE']),width=48,placeholder='…')
            method_label = f'{label} #{j+1} | paired rank {int(row[f"correct_rank_{suffix}"])}'
            show_image(ax,filename,method_label+'\n'+textwrap.fill(title,25),filename==query_id)
    fig.text(.07,.025,f'{category}; query {query_id}. Green border marks the paired painting. Full-gallery ranks; top 3 shown.\nText shown is a readability excerpt; model input used tokenizer truncation. Example selection is documented separately.',fontsize=8,color='#334155')
    return _save_figure(fig,report['out'],name)


def write_figure_captions(report, selected):
    n=len(report['paired']);epoch=report['summary']['best_epoch']
    text=f'''# Draft figure captions

## metrics_comparison
Frozen and projection-only CLIP on the official SemArt validation split: {n} comment queries
ranked against the same {n} paintings. Recall is reported as a percentage; MRR is unitless.
Both evaluations use float32 ViT-B/32 and tokenizer-truncated comments. Epoch {epoch} was
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

Selected query IDs: {json.dumps(selected,sort_keys=True)}
'''
    (report['out']/'figure_captions.md').write_text(text)
