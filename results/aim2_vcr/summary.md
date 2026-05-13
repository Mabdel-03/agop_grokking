# AGOP Grokking Summary

- Runs analyzed: 1
- Grokking-like runs: 1
- Grokked seeds: seed 0 (fit 50, grok 20000)

## Relative 50% Lead Times

| metric_name             | median  | count | fraction_positive |
| ----------------------- | ------- | ----- | ----------------- |
| random_align_K1         | 19950.0 | 1     | 1.0               |
| log_weight_norm         | 19950.0 | 1     | 1.0               |
| vcr_pca_r4              | 19950.0 | 1     | 1.0               |
| vcr_pca_r8              | 19950.0 | 1     | 1.0               |
| random_align_K8         | 19950.0 | 1     | 1.0               |
| train_loss              | 19950.0 | 1     | 1.0               |
| random_align_K4         | 19900.0 | 1     | 1.0               |
| align_K1                | 19550.0 | 1     | 1.0               |
| vcr_full                | 19550.0 | 1     | 1.0               |
| align_K8                | 19550.0 | 1     | 1.0               |
| vcr_random_r4           | 19550.0 | 1     | 1.0               |
| vcr_agop_r8             | 19550.0 | 1     | 1.0               |
| vcr_agop_r4             | 19550.0 | 1     | 1.0               |
| vcr_random_r8           | 19050.0 | 1     | 1.0               |
| align_K4                | 18550.0 | 1     | 1.0               |
| random_align_K2         | 17000.0 | 1     | 1.0               |
| ntk_drift_correct_logit | 17000.0 | 1     | 1.0               |
| align_K2                | 11000.0 | 1     | 1.0               |

Best Fourier-alignment median lead time: `align_K1` = 19550.0 steps.
This exceeds 1 of 3 baseline median lead times.

## Output Files
- `metrics_all.csv`
- `agop_spectra_all.csv`
- `lead_times.csv`
- per-run plot folders named by `run_id`
- aggregate `lead_time_bar.*`, `lead_time_table_heatmap.*`, and `final_summary.*`

Claims should remain cautious: positive lead time is evidence for the AGOP-Fourier hypothesis only when delayed generalization is present.
