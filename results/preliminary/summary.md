# AGOP Grokking Summary

- Runs analyzed: 3
- Grokking-like runs: 3
- Grokked seeds: seed 0 (fit 40, grok 21000), seed 1 (fit 40, grok 30000), seed 2 (fit 40, grok 26000)

## Relative 50% Lead Times

| metric_name             | median  | count | fraction_positive |
| ----------------------- | ------- | ----- | ----------------- |
| align_K2                | 25990.0 | 3     | 1.0               |
| align_K4                | 25990.0 | 3     | 1.0               |
| align_K8                | 25990.0 | 3     | 1.0               |
| random_align_K8         | 25990.0 | 3     | 1.0               |
| random_align_K2         | 25990.0 | 3     | 1.0               |
| random_align_K4         | 25980.0 | 3     | 1.0               |
| log_weight_norm         | 25970.0 | 3     | 1.0               |
| train_loss              | 25960.0 | 3     | 1.0               |
| align_K1                | 25400.0 | 3     | 1.0               |
| random_align_K1         | 23600.0 | 3     | 1.0               |
| ntk_drift_correct_logit | 23600.0 | 3     | 1.0               |

Best Fourier-alignment median lead time: `align_K2` = 25990.0 steps.
This exceeds 3 of 3 baseline median lead times.

## Output Files
- `metrics_all.csv`
- `agop_spectra_all.csv`
- `lead_times.csv`
- per-run plot folders named by `run_id`
- aggregate `lead_time_bar.*`, `lead_time_table_heatmap.*`, and `final_summary.*`

Claims should remain cautious: positive lead time is evidence for the AGOP-Fourier hypothesis only when delayed generalization is present.
