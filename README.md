# Learning Over-Relaxation Policies for ADMM with Convergence Guarantees

Code and pretrained models for the paper:

> Junan Lin, Paul J. Goulart, and Luca Furieri, **“Learning
> Over-Relaxation Policies for ADMM with Convergence Guarantees,”** IEEE
> Conference on Decision and Control (CDC), 2026.

The method learns an online policy for the [OSQP](https://osqp.org) relaxation parameter
`alpha`. Two MLP policies are provided:

- **Scalar policy:** predicts one relaxation parameter shared by all
  constraints.
- **Vector policy:** predicts one relaxation parameter per constraint using a
  shared row-equivariant MLP.

Both policies update `alpha` every 10 OSQP iterations, predict values in
`[1.25, 1.95]`, and stop changing `alpha` after 500 iterations. Unlike an OSQP
penalty (`rho`) update, an `alpha` update does not change the KKT matrix and
therefore does not require a refactorization which is time consuming.

## Files included

- learned OSQP implementation and training code;
- QP problem generators (provided by [Benchmark examples for the OSQP solver](https://github.com/osqp/osqp_benchmarks) repository);
- evaluation scripts;
- MLP checkpoints;
- compact paper-result and regression-reference CSVs.

## Environment

Create a clean environment with:

```bash
conda create -n learned-alpha-osqp python=3.13 -y
conda activate learned-alpha-osqp
python -m pip install -r requirements.txt
```


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

Run the following command:

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

# Diagnostic plot run: 1 instance at 1 dimension per family
python test_neural_comparison.py --mode plot
```

Plot mode is intended for inspecting policy behaviour rather than reproducing
the aggregate values in Table 1. It enables per-stage history recording and
uses one deterministic instance at one dimension for each problem family. For
a scalar policy, it saves a PNG showing the learned `alpha` trajectory together
with the scaled primal-to-dual residual ratio, and a PDF showing the magnitude
of each `alpha` change. For a vector policy, it saves the PDF using the maximum
absolute row-wise `alpha` change at each policy update. These files are written
under each neural solver's `alpha_plots/` result directory. The run also writes
the usual CSV results and time/iteration performance-profile plots; profiles
based on a single instance are diagnostic only.

As in benchmark mode, an existing per-dimension CSV is reused. Remove only the
corresponding generated result directory if you need to regenerate its policy
history plots.

Results are written to:

```text
results/0324_neural_comparison_<family>_solver_alpha_freeze/
```

Generate the compact Table 1 CSV after the five result folders exist:

```bash
python compute_statistics.py
```

The historical paper values are stored in
[`reference_results/table1_summary.csv`](reference_results/table1_summary.csv).

The script skips a solver if its results.csv already exists. Remove only the specific generated result directory you intentionally want to rerun.

### Table 1 values in the paper

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

### Table 2 values in the paper

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

## Retraining

Training data are generated and cached on demand by `learned_osqp/data.py`.
Both training entry points use `learned_osqp/data` by default and accept
`--data-dir` (or `--data_dir`) to select another cache location. Relative paths
are resolved from the current working directory, so run these commands from the
repository root. See
[`learned_osqp/data/README.md`](learned_osqp/data/README.md) for details.

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

Change the `types` and `sizes` as follows:

```text
random_qp: 250
portfolio: 20
lasso:     20
svm:       20
control:   100
```

For a scalar policy, change `--alpha_mode vector` to `--alpha_mode scalar`. For fixed `rho=0.1`, add
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
resources, and command arrays for provenance.

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
├── test_neural_comparison.py     # Table 1 experiment
├── test_control_fixed.py         # Table 2 experiment
├── compute_statistics.py         # Table 1 aggregation
├── scripts/arc/                  # original Slurm scripts
├── reference_results/            # compact historical values and manifests
└── requirements.txt
```


## Acknowledgements

This work used the
[University of Oxford Advanced Research Computing (ARC) facility](https://doi.org/10.5281/zenodo.22558).
Parts of this codebase were adapted from the
[OSQP benchmark examples](https://github.com/osqp/osqp_benchmarks) and the
pure-Python OSQP implementation distributed in
[osqp-python](https://github.com/osqp/osqp-python). We thank the OSQP developers
and contributors for making these projects available.
