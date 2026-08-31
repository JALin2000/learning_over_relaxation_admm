# Learning Over-Relaxation Policies for ADMM with Convergence Guarantees

Code and pretrained models for the paper:

> Junan Lin, Paul J. Goulart, and Luca Furieri, **“Learning
> Over-Relaxation Policies for ADMM with Convergence Guarantees,”** IEEE
> Conference on Decision and Control (CDC), 2026.

The method learns an online policy for the OSQP/ADMM relaxation parameter
`alpha`. Two MLP policies are provided:

- **Scalar policy:** predicts one relaxation parameter shared by all
  constraints.
- **Vector policy:** predicts one relaxation parameter per constraint using a
  shared row-equivariant MLP.

Both policies update `alpha` every 10 OSQP iterations, predict values in
`[1.25, 1.95]`, and stop changing `alpha` after 500 iterations. Unlike an OSQP
penalty (`rho`) update, an `alpha` update does not change the KKT matrix and
therefore does not require a refactorization.

## Release status

This directory is a minimal release assembled from the camera-ready code state
(`d67f260`, “for cdc initial submission”) and the ARC checkpoint directories
referenced by the evaluation scripts. Existing Python and Slurm files were
copied byte-for-byte; release documentation, manifests, and `smoke_test.py` are
new files.

Included:

- the learned OSQP implementation and training code;
- the problem generators required by the paper experiments;
- the two paper evaluation scripts;
- the 36 MLP checkpoints actually selected by those scripts;
- the four remembered ARC submission/retry scripts;
- compact paper-result and regression-reference CSVs.

Not included:

- `learned_osqp/data/` datasets (the local copy was about 22 GB and may not be
  identical to the data regenerated on ARC);
- unrelated checkpoint families, GRU checkpoints, plots, caches, IDE files,
  and the large benchmark datasets inherited from the upstream repository;
- full raw result directories (compact summaries and smoke-test reference rows
  are included instead).

## Environment

The original ARC scripts explicitly run `conda activate rlqp`. The local
`rlqp` environment was therefore used for release validation; it was originally
cloned from `NeuralADMM`. The validated environment was:

- Python 3.13.9
- NumPy 2.3.4
- SciPy 1.16.3
- PyTorch 2.9.1
- CVXPY 1.7.5
- pandas 2.3.3
- Matplotlib 3.10.7

Create a clean environment with:

```bash
conda create -n learned-alpha-osqp python=3.13 -y
conda activate learned-alpha-osqp
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` lists direct imports only rather than a complete environment
freeze. It also explains three legacy-only dependencies:

- `gurobipy` and `Mosek` are imported eagerly by the unchanged inherited solver
  registry, although the paper scripts do not call those solvers;
- `tables` is imported by the unchanged SuiteSparse continuation at the end of
  `test_neural_comparison.py`.

No Gurobi or MOSEK license is needed for the paper's OSQP experiments. A
license is required only if those commercial solver backends are used. Removing
their Python packages from the installation would require making the legacy
registry imports optional, which was deliberately not done in this faithful
copy.

## Quick checkpoint validation

The smoke test needs no training data. It regenerates test instances from the
original seeds and compares all eight paper solver configurations against
validated historical rows.

```bash
# Table 1: one test dimension and one instance for each of five families
python smoke_test.py --suite table1

# Table 2: the first three unseen initial states (seeds 240, 241, 242)
python smoke_test.py --suite table2

# Both suites; optionally save the detailed comparison
python smoke_test.py --suite all --output smoke_results.csv
```

On the release machine, all 64 comparisons matched exactly in solver status,
iteration count, and number of `rho` updates. Objective differences were below
`4.5e-9`. See [VALIDATION.md](VALIDATION.md) for the complete validation scope
and known discrepancies.

## Pretrained checkpoint layout

### Table 1: benchmark problem families

```text
learned_osqp/checkpoints_arc/
└── 0324_feat_unscaled_alpha_1.25_1.95_loss_softplus/
```

This directory contains six MLP checkpoints for each of Random QP, Portfolio,
Lasso, SVM, and Control:

| `rho` mode | Policy | Selection | Filename suffix |
|---|---|---|---|
| adaptive | scalar | lowest validation iterations | `.pt` |
| adaptive | scalar | fewest validation `rho` updates | `_best_rho.pt` |
| adaptive | vector | lowest validation iterations | `.pt` |
| adaptive | vector | fewest validation `rho` updates | `_best_rho.pt` |
| fixed (`rho=0.1`) | scalar | lowest validation iterations | `.pt` |
| fixed (`rho=0.1`) | vector | lowest validation iterations | `.pt` |

### Table 2: fixed MPC system, varying initial state

```text
learned_osqp/checkpoints_arc/
└── 0324_control_fixed/
```

This directory contains the same six policy/selection combinations for
`nx=100`, `dynamics_seed=0`.

The exact file list, SHA-256 hashes, checkpoint epochs, validation metrics, and
embedded training configuration are recorded in
[`reference_results/checkpoint_manifest.csv`](reference_results/checkpoint_manifest.csv).

## Reproducing Table 1

The camera-ready Table 1 script is:

```bash
python test_neural_comparison.py --mode benchmark
```

It evaluates 10 deterministic instances at each of 10 larger dimensions:

| Family | Training/validation size | Test dimensions |
|---|---:|---|
| Random QP | 250 | 500, 501, 503, 507, 515, 531, 562, 625, 750, 999 |
| Portfolio | 20 | 50, 51, 52, 54, 58, 63, 72, 87, 110, 149 |
| Lasso | 20 | 50, 51, 52, 54, 58, 63, 72, 87, 110, 149 |
| SVM | 20 | 50, 51, 52, 54, 58, 63, 72, 87, 110, 149 |
| Control | 100 | 200, 201, 203, 205, 210, 219, 235, 262, 311, 399 |

Useful options:

```bash
# Smaller legacy run: 3 instances at 5 dimensions per family
python test_neural_comparison.py --mode benchmark --small

# Parallelize instances (results can have noisier timings)
python test_neural_comparison.py --mode benchmark --parallel
```

Results are written to:

```text
results/0324_neural_comparison_<family>_solver_alpha_freeze/
```

After all five paper families finish, the unchanged script prints
`All QP types completed!` and then enters an older SuiteSparse-Lasso transfer
experiment. That experiment is **not used in the camera-ready paper**, and its
large matrices are not bundled. An expected missing-data failure after that
message does not invalidate the Table 1 outputs; stop the process at that point
unless the SuiteSparse data have been supplied separately.

Generate the compact Table 1 CSV after the five result folders exist:

```bash
python compute_statistics.py
```

The historical paper values are stored in
[`reference_results/table1_summary.csv`](reference_results/table1_summary.csv).

### Camera-ready Table 1 values

Mean iterations are shown first; mean solve time in seconds is in parentheses.

| `rho` mode | Policy/checkpoint | Random QP | Portfolio | Lasso | SVM | Control |
|---|---|---:|---:|---:|---:|---:|
| fixed | OSQP | 321.53 (0.803) | 454.64 (1.981) | 203.39 (1.972) | 654.10 (6.715) | 1331.41 (7.350) |
| fixed | scalar | 263.34 (0.653) | 372.95 (1.721) | 172.64 (1.667) | 547.07 (6.000) | 1096.20 (6.043) |
| fixed | vector | 269.56 (0.748) | 373.13 (1.886) | 172.44 (1.761) | 545.49 (6.332) | 1092.95 (6.196) |
| adaptive | OSQP | 321.53 (0.808) | 272.30 (1.645) | 175.58 (3.644) | 346.32 (7.959) | 127.48 (1.518) |
| adaptive | scalar, best iterations | 262.92 (0.662) | 233.58 (1.468) | 175.62 (3.847) | 341.45 (7.949) | 126.86 (1.551) |
| adaptive | scalar, best `rho` updates | 262.92 (0.666) | 324.26 (1.858) | 179.96 (7.042) | 462.78 (7.059) | 261.26 (2.438) |
| adaptive | vector, best iterations | 268.41 (0.755) | 251.16 (1.595) | 180.09 (3.654) | 343.53 (7.642) | 140.13 (1.657) |
| adaptive | vector, best `rho` updates | 265.65 (0.736) | 366.33 (2.038) | 174.76 (1.772) | 425.22 (7.836) | 218.13 (2.100) |

## Reproducing Table 2

The fixed-MPC system has `nx=100`, `nu=50`, horizon 10, and
`dynamics_seed=0`. Training and validation use initial-state seeds 0--239;
testing begins at seed 240.

```bash
python test_control_fixed.py \
  --mode benchmark \
  --nx 100 \
  --dynamics_seed 0 \
  --n_train 160 \
  --n_val 80 \
  --n_test 100
```

Results are written to:

```text
results/control_fixed_nx100_dseed0_alpha_freeze_new/
```

The script skips a solver if its `results.csv` already exists. Remove only the
specific generated result directory you intentionally want to rerun.

### Camera-ready Table 2 values

| `rho` mode | Policy/checkpoint | Mean iterations (solve time, s) |
|---|---|---:|
| fixed | OSQP | 890.7 (0.745) |
| fixed | scalar | 733.4 (0.621) |
| fixed | vector | 730.9 (0.691) |
| adaptive | OSQP | 141.4 (0.169) |
| adaptive | scalar, best iterations | 133.5 (0.168) |
| adaptive | scalar, best `rho` updates | 139.7 (0.176) |
| adaptive | vector, best iterations | 126.5 (0.174) |
| adaptive | vector, best `rho` updates | 125.8 (0.183) |

The latest local CSV produces the same deterministic iteration means but a
`0.173219` s mean for the last row; see [VALIDATION.md](VALIDATION.md). Runtime
differences across machines or repeated runs are expected, so compare
iterations, status, and `rho` updates before interpreting timing differences.

## Retraining

Training data are generated and cached on demand by `learned_osqp/data.py`, but
the two unchanged training entry points still contain the original absolute ARC
data path. Read [`learned_osqp/data/README.md`](learned_osqp/data/README.md)
before retraining outside ARC.

The benchmark-suite command template is:

```bash
python learned_osqp/train.py \
  --device cuda \
  --precision low \
  --types random_qp \
  --sizes 250 \
  --normalize_features \
  --alpha_mode vector \
  --ckpt learned_osqp/checkpoints/my_random_qp_vector_arho.pt
```

Change the family and size as follows:

```text
random_qp: 250
portfolio: 20
lasso:     20
svm:       20
control:   100
```

For a scalar policy, add `--alpha_mode scalar`. For fixed `rho=0.1`, add
`--adaptive_rho false`. The defaults used in the paper are 1000 epochs, learning
rate `5e-5`, stage length 10, 160 training instances, 80 validation instances,
and a two-hidden-layer MLP with 64 units per layer. Use `--ckpt` explicitly so
the output path is clear.

The fixed-system MPC template is:

```bash
python learned_osqp/train_control_fixed.py \
  --device cuda \
  --precision low \
  --nx 100 \
  --dynamics_seed 0 \
  --batch 10 \
  --normalize_features \
  --alpha_mode vector \
  --ckpt learned_osqp/checkpoints/my_control_fixed_vector_arho.pt
```

The original ARC job arrays are preserved under [`scripts/arc/`](scripts/arc/):

- `submit_array_vector_mlp_alpha_1.25_1.95_gpu.sh`
- `submit_array_scalar_mlp_alpha_1.25_1.95_gpu.sh`
- `submit_array_failed_jobs.sh`
- `submit_array_failed_jobs2.sh`

They retain the original ARC working directory, Conda environment, Slurm
resources, and command arrays for provenance. Some retry entries are GRU runs
that were not reported in the camera-ready MLP tables.

## Repository structure

```text
.
├── learned_osqp/                 # learned policy, differentiable OSQP, training
│   ├── checkpoints_arc/          # selected camera-ready MLP checkpoints
│   └── data/README.md            # why datasets are excluded
├── solvers/osqppurepy/           # modified pure-Python OSQP implementation
├── problem_classes/              # paper QP generators and legacy import closure
├── benchmark_problems/           # benchmark orchestration
├── utils/                        # result aggregation and plotting utilities
├── test_neural_comparison.py     # Table 1 experiment (plus legacy tail)
├── test_control_fixed.py         # Table 2 experiment
├── compute_statistics.py         # Table 1 aggregation
├── smoke_test.py                 # small deterministic release regression
├── scripts/arc/                  # original Slurm scripts
├── reference_results/            # compact historical values and manifests
├── requirements.txt
└── VALIDATION.md
```

## Reproducibility notes and known limitations

1. **Training data are absent.** Pretrained evaluation and the smoke test do not
   require them; exact retraining will require recovering or regenerating the
   authoritative ARC datasets.
2. **Training data paths are ARC-specific.** The current training entry points
   have no `--data-dir` option because existing source files were not modified
   during this migration.
3. **One checkpoint metadata mismatch is known.** The adaptive-rho scalar
   Portfolio checkpoints record batch size 10, while the paper describes batch
   size 16 for benchmark-suite training. The failed-job Slurm script confirms
   that this configuration was rerun with batch size 10.
4. **Runtime is not deterministic.** The camera-ready results were benchmarked
   on an Apple M4 MacBook Pro with 16 GB RAM. Use identical hardware, process
   load, thread settings, and serial execution for meaningful time comparisons.
5. **The bundled PyTorch checkpoints are trusted project artifacts.** The loader
   uses `torch.load(..., weights_only=False)` because each checkpoint contains a
   serialized `Config`. Do not replace them with untrusted checkpoint files.

## Citation

Please update this entry with the final DOI or public paper URL when available:

```bibtex
@inproceedings{lin2026learning,
  author    = {Junan Lin and Paul J. Goulart and Luca Furieri},
  title     = {Learning Over-Relaxation Policies for {ADMM} with Convergence Guarantees},
  booktitle = {Proceedings of the IEEE Conference on Decision and Control},
  year      = {2026}
}
```

## License and provenance

The Apache License 2.0 file from the source repository is retained as
[`LICENSE`](LICENSE). This project was developed from the
[OSQP benchmark repository](https://github.com/osqp/osqp_benchmarks), and the
headers in `solvers/osqppurepy/` identify code copied from the OSQP Python
implementation. The authors should complete a final third-party notice and
artifact-distribution review before making the new GitHub repository public.
