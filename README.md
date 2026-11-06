# Deep Learning: Semantic Art Retrieval
CS 7643 / 4644

## Research questions

1. Does adaptation of pretrained CLIP improve text-to-painting retrieval on SemArt?
2. Do improvements persist when evaluation artists are absent from fine-tuning data?
3. Does artwork metadata supply useful context or encourage shortcuts?

An artist-disjoint split controls exposure during our SemArt fine-tuning. It does
not establish that those artists were absent from CLIP's original pretraining.

## First milestone: frozen CLIP baseline

- Suggested starting checkpoint: OpenAI CLIP `ViT-B/32`.
- Query: artistic comment only; no added metadata.
- Similarity: cosine similarity between normalized text and image embeddings.
- Development: official validation split; rank against its complete image gallery.
- Final standard evaluation: official test split after development choices are fixed.
- Metrics: Recall@1, Recall@5, Recall@10, and mean reciprocal rank (MRR).
- Initial long-text policy: tokenizer truncation; measure the fraction affected.
- Preserve raw comments and metadata for later controlled experiments.

Checkpoint and truncation are initial implementation choices, not completed
experimental findings. Never compare results from different gallery sizes as
though they used the same retrieval task.

## Where files live

| Location | Contents |
| --- | --- |
| Local Git checkout | Notebook, dependency list, documentation, small results |
| ICE `~/scratch/semantic-art-retrieval` | Working directory, data, caches, embeddings, checkpoints |
| ICE `~/scratch/envs/semantic-art` | Persistent Python environment |

The local notebook sends cells to an ICE kernel. Imports, relative paths, and
files created by Python refer to ICE. Local `.py` files do not synchronize
automatically. Save and back up remote results explicitly; scratch is working storage.

Suggested remote layout (create once the download location is known):

```text
semantic-art-retrieval/
  data/             # original SemArt files; retain original split definitions
  cache/            # downloaded model weights
  artifacts/        # embeddings, IDs, model checkpoints
  results/          # small metrics/configuration summaries to copy back to Git
```

## ICE setup

Run these inside an allocated compute shell. Reuse an existing environment if
it is already installed; do not recreate it each session.

```bash
module spider anaconda3
module load anaconda3
```

If needed, follow the module system's version/prerequisite instructions.
First-time environment creation:

```bash
mkdir -p ~/scratch/envs
conda create --prefix ~/scratch/envs/semantic-art python=3.11 pip -y
conda activate ~/scratch/envs/semantic-art
```

Install CUDA-enabled PyTorch and torchvision using the Linux/Pip command from
https://pytorch.org/get-started/locally/ compatible with the allocated GPU and
driver. Then copy this dependency file to the remote workspace and run:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name semantic-art --display-name "Python (semantic-art ICE)"
```

`requirements.txt` is intentionally an initial dependency list. After the setup
works, record installed versions and the exact CLIP revision before comparing runs.

## Each interactive session

From the ICE login node, request resources:

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:1 --time=02:00:00 --pty bash
```

Course-specific account, partition, or QoS options may be required. Once granted:

```bash
hostname
nvidia-smi
module load anaconda3
conda activate ~/scratch/envs/semantic-art
mkdir -p ~/scratch/semantic-art-retrieval
cd ~/scratch/semantic-art-retrieval
```

Start Jupyter from this directory:

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.port_retries=0
```

Keep that terminal open. In a second **local** terminal, substitute your username
and the hostname returned by `hostname`:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -J GTUSERNAME@login-ice.pace.gatech.edu \
  -L 127.0.0.1:8888:127.0.0.1:8888 GTUSERNAME@COMPUTE_NODE
```

This requires SSH access to your allocated compute node. If ICE rejects it,
inspect that error before continuing; do not assume the tunnel is established.
In local VS Code, open `semart.ipynb`, select an existing Jupyter server, enter
`http://127.0.0.1:8888/?token=YOUR_TOKEN`, and choose the registered ICE kernel.
Keep tokens out of Git. End Jupyter with Ctrl+C and leave the compute shell with
`exit` when finished.

## Work possible before SemArt finishes downloading

1. Verify the notebook kernel's hostname, Python path, and CUDA access.
2. Load frozen CLIP with its supplied preprocessing and tokenizer.
3. Test one permitted image with two short descriptions; check finite outputs,
   matching embedding dimensions, and normalized similarities. This is only a
   smoke test, not an accuracy measurement.
4. Specify artwork-ID alignment and retrieval conventions before dataset code.
5. Hand-check metrics on correct-match ranks `[1, 2, 10, 11]`: R@1 = 0.25,
   R@5 = 0.50, R@10 = 0.75, MRR approximately 0.422727.

## Later experiments

- Add a deterministic artist-grouped split with artist-disjoint train/validation/test
  groups; save a split manifest, seed, counts, and unknown-artist handling.
- Compare frozen CLIP with projection-only adaptation, then partial fine-tuning;
  consider full fine-tuning only if resources and validation evidence justify it.
- Use bidirectional contrastive loss for fine-tuning; keep text-to-image retrieval
  as the primary reported task.
- Compare comment-only queries with selected metadata subsets under the same
  checkpoint/split protocol. Audit artist mentions already present in comments.
- Develop preprocessing and model-selection choices on validation data.
- Save qualitative failures involving visible content versus historical context.

For each run, record the Git commit, model/checkpoint, library versions, split and
gallery IDs, query fields, truncation policy, exclusions, and metrics. Report
dataset counts from the actual downloaded files rather than copying inconsistent
approximate counts from the proposal. An initial baseline on the official split
should pass basic checks before building artist-disjoint experiments.

## Rerunning training versus regenerating the report

Do not use **Run All** just to regenerate figures. The notebook contains ICE
training cells and a separate local report section.

| Goal | What to run |
| --- | --- |
| Regenerate figures/tables from the completed run | Local kernel; run only **Milestone report artifacts**. Set `RUN_NAME` explicitly. No training or checkpoint loading occurs. |
| Start a new adaptation experiment | ICE kernel; `RESUME_FROM = None`; run the fine-tuning section from its configuration cell. This starts from pretrained CLIP weights in a new directory. |
| Resume interrupted training | ICE kernel; set `RESUME_FROM` to the run's `last_training.pt`; rerun the fine-tuning section from the configuration cell. |
| Re-evaluate a finished run | Resume from `last_training.pt` with the original total epoch budget. No epochs run if the saved epoch already reaches that budget; the selected checkpoint is restored and evaluated. |
| Continue past three completed epochs | Resume and increase the total `config["epochs"]` budget, for example to 5. Other settings and dataset contents must match. |

Resumption starts at the next epoch boundary; an interrupted partial epoch is
repeated. Checkpoints include optimizer and random states. New checkpoint files
embed the best projection state to recover consistently if saving the separate
best file was interrupted. Older checkpoint pairs remain supported when consistent.
This does not guarantee identical GPU results across hardware/library changes.

For the downloaded run, report outputs are in `results/projection_90b55b2d/`.
`methodology.md` drafts the completed method/conditions; `report_checklist.md`
maps artifacts to the milestone sections and identifies remaining writing.
`source_evidence/` retains small raw JSON/CSV evidence for regeneration; model
weights and optimizer checkpoints remain in ignored `artifacts/`. If the artifact
folder is absent, report generation can use `source_evidence/` plus the original
local SemArt files. Back up checkpoints separately from Git.

The report section overwrites generated files for the explicitly selected run.
For a new training run, download its folder and change `RUN_NAME` before plotting.
