# Training data (not included)

The original local data directory was approximately 22 GB and is intentionally
excluded from this release. The final ARC runs also regenerated data on the
cluster, so the local data files could not be established as the authoritative
training copy.

The training code can generate and cache datasets from the problem generators
in `problem_classes/`. For the paper configuration, each benchmark family used
160 training and 80 validation instances. The fixed-control experiment used one
shared system (`dynamics_seed=0`) and varied the initial-state seed.

## Cache location

Both training entry points default to the repository-relative cache directory
`learned_osqp/data`. Run them from the repository root so that this relative
path resolves as shown here. A different cache location can be selected with
either spelling of the command-line option:

```bash
python learned_osqp/train.py --data-dir path/to/data ...
python learned_osqp/train_control_fixed.py --data_dir path/to/data ...
```

If the expected `.pt` cache is absent, the entry point generates it and saves
it under this directory. On a later run with the same problem configuration,
the cached file is loaded instead. Pass `--regen` to replace the matching cache.

Generated `.pt` datasets are ignored by Git and should normally be hosted as a
separate release asset if exact training data later become available.
