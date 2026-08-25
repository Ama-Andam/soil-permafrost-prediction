# Results

All CSVs and figures are generated on UND Talon HPC and stored at:

  ~/results_v4/   — all model results, scaling tables, senior experiments
  ~/figures_v4/   — all figures (V4, SCALE, RECOV, SENIOR, DETAIL, COMP)

Key files:
  v4_results_all.csv                    — full model performance (33 models)
  scaling_results.csv                   — GPU scaling wall times + speedup
  scaling_per_model_results.csv         — per-model timing at each GPU count
  scaling_training_master_table.csv     — senior format training table
  scaling_tuning_master_table.csv       — senior format tuning table
  stability_summary.csv                 — 10-run stability mean ± std
  uncertainty_results.csv               — 95% prediction interval coverage
  pruning_results.csv                   — pruning efficiency frontier
  tuning_results.csv                    — hyperparameter tuning results
