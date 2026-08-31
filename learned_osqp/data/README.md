# Training data (not included)

The original local data directory was approximately 22 GB and is intentionally
excluded from this release. The final ARC runs also regenerated data on the
cluster, so the local data files could not be established as the authoritative
training copy.

The training code can generate and cache datasets from the problem generators
in `problem_classes/`. For the paper configuration, each benchmark family used
160 training and 80 validation instances. The fixed-control experiment used one
shared system (`dynamics_seed=0`) and varied the initial-state seed.

## Important portability note

The unchanged entry points `learned_osqp/train.py` and
`learned_osqp/train_control_fixed.py` still set `Config.data_dir` to the
original ARC path:

```text
/data/engs-goulart/sedm7756/0319_feat_pri_dua_res_scaled_alpha_1.25_1.95
```

There is currently no `--data-dir` command-line option. Before retraining on a
different machine, change that argument to `learned_osqp/data` (the commented
relative path immediately above it shows the intended local value), or add a
portable CLI option in a follow-up cleanup. This release deliberately does not
alter those existing Python files.

Generated `.pt` datasets are ignored by Git and should normally be hosted as a
separate release asset if exact training data later become available.
