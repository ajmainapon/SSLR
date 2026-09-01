# Submitted-paper naming

The repository preserves historical experiment identifiers so that existing checkpoints and run
logs remain addressable. Reader-facing documentation, tables, and figures use the submitted-paper
names below.

| Paper name | Masking | Slice gap | Historical identifier |
|---|---|---|---|
| I-JEPA-CT | multi-block, 40% | wide, k=8–20 | `v2`, `vit_ep039.pt` |
| A1 | random per-patch, 75% | wide, k=8–20 | `loo_a1_random_mask` |
| **A2 (selected)** | **multi-block, 40%** | **narrow, k=3–7** | `loo_a3_narrow_k`, run prefix `a3` |
| A3 | random per-patch, 75% | narrow, k=3–7 | `loo_a5_dense_narrow`, run prefix `a5` |

Historical identifiers are implementation aliases only. They should not be used as configuration
names in new prose, tables, figure legends, or result summaries.
