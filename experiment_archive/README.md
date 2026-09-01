# Experiment archive

This post-submission archive records the final remote experiments completed after the main codebase
was established. It contains lightweight, auditable tables and the exact queue/evaluation scripts;
raw medical data, model checkpoints, full run directories, virtual environments, and credentials
are intentionally excluded from Git.

## Contents

- `results/swinunetr_multiseed.csv`: four-seed matched SSL-vs-random SwinUNETR fine-tuning.
- `results/capacity_control_n50.csv`: the matched 2×2 encoder/head capacity control at N=50.
- `results/capacity_control_n20_additional.csv`: additional N=20 cells, with the cohort mismatch
  explicitly marked.
- `results/heldout_test_a2.csv`: valid held-out-test results for the A2 frozen and fine-tuned arms.
- `results/compute_cost.csv`: measured wall-clock training and inference costs.
- `../scripts/test_inference/`: source-only held-out-test evaluation harness.

All Dice values are mean foreground Dice. Standard deviations use the four seed-level results.
The run logs/checkpoints on the experiment hosts remain the primary source; these CSV files are a
compact archival export.

## Validity boundaries

The N=50 capacity table is the controlled factorial: the four cells use the matched N=50 protocol.
The two additional N=20 cells were drawn from different stored patient-list constructions and must
not be interpreted as a single controlled 2×2 factorial.

Only A2 test-set values that passed the locked-checkpoint audit are included. Frozen-random test
outputs are excluded because the evaluator did not reproduce the training-time random encoder
initialization order. Random fine-tune checkpoints were unavailable. Test-set values were never
used for model or checkpoint selection.

Inference time is amortized over 89 preprocessed CT volumes (25,578 slices). It includes model
construction, checkpoint/array loading, and forward passes, but excludes raw-NIfTI preprocessing.
