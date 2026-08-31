# Release validation

Validation was performed on 31 August 2026 using the local `rlqp` Conda
environment (Python 3.13.9) on an Apple M4 CPU. Wall-clock time was not used as
an equality criterion because it is machine- and load-dependent.

## Checkpoint selection

The selected files are exactly the MLP checkpoints referenced by the two paper
evaluation scripts:

- Table 1: `learned_osqp/checkpoints_arc/0324_feat_unscaled_alpha_1.25_1.95_loss_softplus/`
- Table 2: `learned_osqp/checkpoints_arc/0324_control_fixed/`

For adaptive-rho models, both the lowest-validation-iteration checkpoint
(`.pt`) and the lowest-validation-rho-update checkpoint (`_best_rho.pt`) are
included. For fixed-rho models, only the iteration-selected checkpoint is used.
The release contains 30 Table 1 checkpoints and 6 Table 2 checkpoints. SHA-256
hashes and checkpoint metadata are recorded in
`reference_results/checkpoint_manifest.csv`.

## Small deterministic regression

The test compared solver status, OSQP iteration count, rho-update count, and
objective value with the corresponding historical rows. It used all eight
paper configurations.

| Suite | Instances checked | Configuration rows | Status matches | Iteration matches | Rho-update matches | Largest objective absolute difference |
|---|---:|---:|---:|---:|---:|---:|
| Table 1 | first instance at one paper test dimension for each of 5 families | 40 | 40/40 | 40/40 | 40/40 | `4.44e-9` |
| Table 2 | first 3 unseen initial states (seeds 240--242) | 24 | 24/24 | 24/24 | 24/24 | `1.76e-11` |

The exact expected rows are in `reference_results/smoke_expected_table1.csv`
and `reference_results/smoke_expected_table2.csv`. Run the same check with:

```bash
python smoke_test.py --suite all
```

## What was not validated

- Full 100-instance-per-family Table 1 and full 100-instance Table 2 reruns were
  intentionally not repeated because they are slow.
- Training was not rerun because the authoritative ARC training data are not
  available locally and the data entry points retain an ARC-only absolute path.
- CUDA training/inference was not retested.
- Runtime equality is not expected across machines. The deterministic solver
  metrics above are the regression gate.

## Observed provenance discrepancies

1. The paper states batch size 16 for all benchmark-suite training, but the
   metadata inside the adaptive-rho scalar Portfolio checkpoints records batch
   size 10. This is consistent with `submit_array_failed_jobs.sh`, which reran
   that job with `--batch 10`.
2. The camera-ready Table 2 reports `0.183` s for the adaptive-rho vector
   rho-selected checkpoint. The latest local 100-instance CSV averages
   `0.173219` s while retaining exactly the same mean iteration count (125.84).
   This looks like a timing-run/provenance difference; no deterministic solver
   mismatch was found.
