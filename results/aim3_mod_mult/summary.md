# AGOP Grokking Summary

- Runs analyzed: 1
- Grokking-like runs: 1
- Grokked seeds: seed 0 (fit 50, grok 36000)

## Relative 50% Lead Times

| metric_name             | median  | count | fraction_positive |
| ----------------------- | ------- | ----- | ----------------- |
| log_weight_norm         | 35950.0 | 1     | 1.0               |
| random_align_K4         | 35950.0 | 1     | 1.0               |
| train_loss              | 35950.0 | 1     | 1.0               |
| random_align_K1         | 35950.0 | 1     | 1.0               |
| align_K4                | 35900.0 | 1     | 1.0               |
| random_align_K2         | 35900.0 | 1     | 1.0               |
| align_K1                | 35850.0 | 1     | 1.0               |
| align_K2                | 35550.0 | 1     | 1.0               |
| ntk_drift_correct_logit | 34050.0 | 1     | 1.0               |

Best Fourier-alignment median lead time: `align_K4` = 35900.0 steps.
This exceeds 1 of 3 baseline median lead times.

## Output Files
- `metrics_all.csv`
- `agop_spectra_all.csv`
- `lead_times.csv`
- per-run plot folders named by `run_id`
- aggregate `lead_time_bar.*`, `lead_time_table_heatmap.*`, and `final_summary.*`

Claims should remain cautious: positive lead time is evidence for the AGOP-Fourier hypothesis only when delayed generalization is present.
