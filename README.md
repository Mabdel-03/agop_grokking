# AGOP-Grokking

Code for the paper **"Can AGOP Explain Grokking? A Spectral-Geometric Account
of Delayed Generalization"** by Mahmoud Abdelmoneum and Theo Chen.

The paper unifies four accounts of grokking (Fourier circuit formation,
lazy-to-rich NTK movement, neural collapse, AGOP-based feature learning) by
asking whether all four are different shadows of one transition in the average
gradient outer product (AGOP) of the trained classifier. We test this through
three aims on modular arithmetic.

## Paper-to-code map

| Paper item | Produced by | Config |
|---|---|---|
| Table 1, Figure 1 (preliminary modular-addition runs, 3 seeds) | `scripts/preliminary/slurm_preliminary_runs.sbatch` | `configs/preliminary.yaml` |
| Table 2 Aim 1 rows, Figure 2 left (block-NTK vs AGOP-Fourier) | `scripts/aim_coverage/slurm_run_aims.sbatch` | `configs/aim1_block_ntk.yaml` |
| Table 2 Aim 2 rows, Figure 2 right (projected VCR) | same | `configs/aim2_vcr.yaml` |
| Table 2 Aim 3 rows (modular multiplication MLP) | same | `configs/aim3_mod_mult.yaml` |
| Table 2 Aim 3 RFM row (matched RFM is degenerate) | `scripts/aim_coverage/run_rfm.py` | `configs/aim3_rfm.yaml` |
| Table 2 LaTeX snippet | `scripts/figures/build_tables.py` | (reads `results/`) |
| Figure 2 PDF | `scripts/figures/build_figures.py` | (reads `results/`) |

## Source layout (`src/`)

| File | Role in the paper |
|---|---|
| `data.py` | Modular-addition dataset; modular-multiplication dataset in discrete-log coordinates (Aim 3). |
| `models.py` | Two-layer quadratic MLP (paper Section "Methods"); ReLU MLP fallback; frozen-feature ablation. |
| `fourier.py` | Real Fourier basis on Z_m and the normalized-alignment estimator (Aim 3). |
| `agop.py` | Input AGOP via vmap'd Jacobians + AGOP spectral summaries. |
| `ntk.py` | Full correct-logit NTK drift; block-wise forward Gram G_l and backward Gram B_l (Aim 1). |
| `neural_collapse.py` | Held-out projected variation collapse ratio VCR_Q with AGOP, PCA, random, and full-space projectors (Aim 2). |
| `rfm.py` | Mahalanobis-Gaussian Recursive Feature Machine baseline (Aim 3 intervention). |
| `metrics.py` | Checkpoint schedule, effective rank, eigengap, first-crossing logic. |
| `train.py` | Training loop and per-checkpoint metric emission. |
| `plotting.py` | Per-run accuracy/loss, AGOP spectrum, AGOP-Fourier alignment plots produced inside each result directory. |
| `utils.py` | YAML/JSON I/O, device selection, RNG seeding. |

## Setup

```bash
pip install -r requirements.txt
```

CUDA is used when available and the harness falls back to CPU. Python 3.10+ is expected.

## Reproducing the paper

### 1. Preliminary modular-addition runs (Table 1, Figure 1)

```bash
sbatch scripts/preliminary/slurm_preliminary_runs.sbatch
```

Runs the proven recipe (p=31, train_fraction=0.5, hidden width 512, lr 3e-3,
weight decay 1.0, init scale 1.0, 50,000 steps) on seeds 0, 1, 2. Outputs
land in `results/preliminary/`.

### 2. Aim-coverage runs (Table 2, Figure 2)

```bash
sbatch scripts/aim_coverage/slurm_run_aims.sbatch
```

Runs the four aim-coverage experiments serially in one Slurm allocation
(~3–4 GPU-hours on `ou_bcs_low`):

1. **Aim 1**: block-NTK decomposition under `configs/aim1_block_ntk.yaml`.
2. **Aim 2**: held-out projected variation collapse under `configs/aim2_vcr.yaml`.
3. **Aim 3 (MLP)**: modular multiplication in discrete-log coordinates under `configs/aim3_mod_mult.yaml`.
4. **Aim 3 (RFM)**: matched-capacity Mahalanobis-Gaussian RFM under `configs/aim3_rfm.yaml`.

Outputs land in `results/aim1_block_ntk/`, `results/aim2_vcr/`,
`results/aim3_mod_mult/`, `results/aim3_rfm/`. Each directory contains
`metrics.csv`, `run_summaries.csv`, `lead_times.csv`, `summary.md`, plus
auto-generated plots under `plots/`.

### 3. Build paper-ready tables and figures

```bash
python scripts/figures/build_tables.py   # writes paper_out/tables/aim*.tex
python scripts/figures/build_figures.py  # writes paper_out/figures/aim*.pdf
```

Both scripts take `--results-root` and an output directory; defaults read from
`results/` and write to `paper_out/` under the repo root. The LaTeX snippets
mirror the rows of Table 2 and the two-panel Figure 2 in the paper.

## Reading the results that ship with this repo

Each `results/*/summary.md` reports the per-seed grokking summary and the
median relative-50% lead times. Headline numbers from the paper:

- **Preliminary**: `t_grok ∈ {21000, 30000, 26000}` for seeds 0/1/2 at `t_fit=40`.
- **Aim 1** (p=31, width 512, single seed): `t_grok=20000`. AGOP-Fourier
  alignment at K=2 first-crosses at step 9,000; block backward Gram
  `B_linear2` at 1,700; `B_linear1` at 4,000; full correct-logit NTK at 3,000.
  Block-AGOP signals lead the full kernel, consistent with Aim 1's lazy-to-rich
  prediction.
- **Aim 2**: AGOP-projected VCR (r=8) crosses at 450; PCA-projected at 50;
  random-projected (r=8) at 950; full-space VCR at 450. The AGOP-specific
  advantage exists but is small at one seed.
- **Aim 3**: modular multiplication MLP reaches 99.3% test accuracy with
  `t_grok=36000`; AGOP-Fourier K=4 crosses at step 100, K=2 at 450 on the
  discrete-log basis. RFM with the Mahalanobis-Gaussian kernel does not
  generalize on one-hot inputs.

## Compute environment

The Slurm scripts assume the MIT Engaging cluster with the `ou_bcs_low`
partition and a single A100 80GB GPU per job. Replace the `#SBATCH` headers
to retarget another cluster. All training is full-batch and fits on one GPU.

## Citation

If you build on this work, please cite the paper. Bibliography is in
`paper_out/` (or alongside the published PDF).
