# AMOS-CT cross-dataset experiment — runbook

**Goal:** convert the "single dataset" limitation in `paper_draft.tex` into a
result. Show that A3 SSL features (pretrained on TotalSegmentator) transfer to
a different abdominal CT distribution (AMOS22) under matched linear-probe
protocol.

**Expected outcome:** A3 SSL beats Random init on AMOS at every N tested
(parallel to the TotalSeg headline). If the AMOS gap is substantially smaller
than the TotalSeg gap, that's still a positive cross-dataset result; we
discuss the gap shrinkage as a data-distribution effect, not a backbone
failure.

**Compute budget:** ~1.5 hr GPU for 6 linear-probe runs. AMOS download is
~40 GB; preprocessing ~1 hr CPU on the RTX box.

---

## 0. Run from Mac → execute on RTX box

The Mac has the scripts (already written and verified). The RTX box has
the GPU + existing TotalSeg data + A3 checkpoint. Sync the new scripts and
run there.

```bash
# From Mac (this repo). On the Mac the scripts live under SSLR/scripts/;
# on the RTX box the project is flat at ~/SSLP/, so they land in ~/SSLP/scripts/.
rsync -avz SSLR/scripts/preprocess_amos.py \
           SSLR/scripts/prepare_labels_amos.py \
           SSLR/scripts/queue_amos_linprobe.sh \
           SSLR/scripts/AMOS_RUNBOOK.md \
           rmedu-04@100.101.70.68:~/SSLP/scripts/
```

---

## 1. Download AMOS22 Task-1 (on the RTX box)

```bash
ssh rmedu-04@100.101.70.68
cd ~/SSLP
mkdir -p amos && cd amos

# AMOS22 challenge data is hosted on Zenodo (link from https://amos22.grand-challenge.org/data/)
# Direct Zenodo record: https://zenodo.org/records/7155725
# Download Task-1 archives (CT only). Approx 40 GB total.
wget https://zenodo.org/records/7155725/files/amos22.zip
unzip amos22.zip
# Verify:
ls imagesTr/ | wc -l   # expect 200+ CT volumes (IDs amos_0001..0500)
ls labelsTr/ | wc -l   # expect 200+ label volumes
```

*Note*: AMOS IDs 0001-0500 are CT, 0501-0600 are MRI. Our scripts use only the CT range.

---

## 2. Preprocess AMOS CT volumes + labels (on the RTX box)

```bash
cd ~/SSLP
# Convert raw NIfTI -> uint8 (Z, 224, 224) numpy slices
python scripts/preprocess_amos.py --split train      # ~30 min for 400 vols
python scripts/preprocess_amos.py --split val        # ~5 min for 100 vols

# Convert AMOS multi-organ labels -> our 7-class scheme, same shape
python scripts/prepare_labels_amos.py --split train  # ~20 min
python scripts/prepare_labels_amos.py --split val    # ~5 min

# Sanity check
ls data_amos/slices/train/ | wc -l   # ~400 .npy files
ls data_amos/labels/train/ | wc -l   # ~400 .npy files

# Optional: spot-check alignment with check_organ.py
python scripts/check_organ.py \
    --slices data_amos/slices/val/amos_0401.npy \
    --labels data_amos/labels/val/amos_0401.npy \
    --out alignment_check/amos_0401.png
```

---

## 3. Launch the linear-probe queue (on the RTX box)

```bash
cd ~/SSLP
nohup bash scripts/queue_amos_linprobe.sh > runs/queue_amos_linprobe.out 2>&1 &
disown
# Monitor:
tail -f runs/queue_amos_linprobe.log
```

Wall-clock: ~1.5 hr (6 runs × ~15 min).

---

## 4. Pull results back to Mac

```bash
# From Mac
rsync -avz \
    rmedu-04@100.101.70.68:~/SSLP/runs/lin_amos_*/log.json \
    runs_synced/
# Or just pull the summary table from the tail of the log:
ssh rmedu-04@100.101.70.68 'tail -50 ~/SSLP/runs/queue_amos_linprobe.log'
```

---

## 5. Add a cross-dataset row to `paper_draft.tex`

Expected new section: a new `\subsection{Cross-dataset transfer: AMOS}` after
the Comparison-to-SwinUNETR subsection, with a small table:

```
N    AMOS A3 SSL    AMOS Random    Δ rel
20   0.????         0.????         +??%
50   0.????         0.????         +??%
100  0.????         0.????         +??%
```

And update Limitation #4 ("Single dataset") to reference this result —
either remove it entirely or downgrade to "single anatomy region (abdominal+thoracic),
cross-modality (MRI) not yet tested".

---

## Decision: do we also run fine-tune on AMOS?

The cheap version above is linear probe only. To match the TotalSeg headline
(fine-tune at lr=1e-4), add a `queue_amos_finetune.sh` later. Single-seed
fine-tune at N=20/50/100 × 2 inits = ~10 GPU-hr. Multi-seed (4-seed) = ~40 GPU-hr.

**Recommendation:** start with linear probe only. If AMOS linear-probe gap is
positive (A3 > Random at every N), that's enough to defuse the single-dataset
limitation for the WACV R1 submission. Defer fine-tune to R2 if time permits.

---

## Honest caveats to disclose in the paper

1. **7-class subset, not 10-class.** AMOS is abdominal-only; we cannot test
   our headline finding on lungs+heart in AMOS. The cross-dataset claim is
   restricted to the abdominal-organ subset.

2. **Single seed AMOS probes** (vs 4-seed TotalSeg fine-tune). Cross-dataset
   transfer measures something different from the headline; multi-seed at
   N=50 for the headline AMOS comparison is a worthwhile follow-up.

3. **Same backbone, different downstream data.** This validates that the SSL
   *features* generalize, not that the *pretraining* is robust to source-data
   shift (a stronger but separate claim that would require pretraining on
   AMOS+TotalSeg mix).
