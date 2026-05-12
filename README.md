# AGOP Grokking

Minimal PyTorch experiment package for the project:

**Can AGOP Explain Grokking? A Spectral-Geometric Account of Delayed Generalization**

The core experiment trains small networks on modular addition and tests whether
input AGOP-Fourier alignment rises before the test-accuracy jump.

## Setup

```bash
pip install -r requirements.txt
```

The code uses CUDA when available and falls back to CPU. Python 3.10+ is expected.

## Commands

Smoke test:

```bash
python run_experiment.py --config configs/smoke.yaml
python analyze_results.py --results_dir results/smoke
```

Adaptive pilot search:

```bash
python run_experiment.py --config configs/pilot.yaml
```

Main experiment after pilot:

```bash
python run_experiment.py --config results/pilot/best_config.yaml --seeds 0 1 2 3 4 --results_dir results/main
python analyze_results.py --results_dir results/main
```

Or use the default main config:

```bash
python run_experiment.py --config configs/main.yaml
python analyze_results.py --results_dir results/main
```

SLURM GPU smoke test:

```bash
sbatch scripts/slurm_smoke_gpu.sbatch
```

## Outputs

Each run writes:

- `metrics.csv`
- `agop_spectra.csv`
- `config_resolved.yaml`
- `split_indices.npz`
- `checks.log`
- optional `final_model.pt`

Analysis writes:

- `metrics_all.csv`
- `agop_spectra_all.csv`
- `lead_times.csv`
- `summary.md`
- per-run and aggregate plots under `plots/`

## Scientific Caution

The code does not fabricate positive results. If no configuration groks, the
analysis reports a partial or negative result using the best available pilot
diagnostics.
