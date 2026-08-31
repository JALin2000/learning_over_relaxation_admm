"""Small checkpoint regression test for the two camera-ready experiment suites.

This script does not use the omitted training datasets. It regenerates the test
QP instances from their original seeds and compares deterministic solver metrics
with validated historical rows stored under ``reference_results/``.
"""

from __future__ import annotations

import argparse
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_problems.example import Example
from learned_osqp.neural_osqp_solver import NeuralOSQPSolver
from problem_classes.control import ControlExample
from solvers.osqppurepy import OSQP as OSQPPythonSolver
import solvers.solvers as solver_registry


ROOT = Path(__file__).resolve().parent
TABLE1_CHECKPOINT_DIR = (
    ROOT
    / "learned_osqp"
    / "checkpoints_arc"
    / "0324_feat_unscaled_alpha_1.25_1.95_loss_softplus"
)
TABLE2_CHECKPOINT_DIR = (
    ROOT / "learned_osqp" / "checkpoints_arc" / "0324_control_fixed"
)

TABLE1_PROBLEMS = (
    ("Random QP", "random_qp", 500),
    ("Portfolio", "portfolio", 50),
    ("Lasso", "lasso", 50),
    ("SVM", "svm", 50),
    ("Control", "control", 200),
)

BASE_SETTINGS = {
    "max_iter": int(1e9),
    "eps_abs": 1e-3,
    "eps_rel": 1e-3,
    "polish": False,
    "verbose": False,
    "eps_prim_inf": 1e-15,
    "eps_dual_inf": 1e-15,
    "time_limit": 1000.0,
}


def _settings(adaptive_rho: bool) -> dict:
    settings = dict(BASE_SETTINGS)
    if not adaptive_rho:
        settings["adaptive_rho"] = False
    return settings


def _solver_specs() -> list[tuple[str, bool, str | None, str | None]]:
    specs: list[tuple[str, bool, str | None, str | None]] = [
        ("OSQP_python_arho", True, None, None),
        ("OSQP_python_no_arho", False, None, None),
    ]
    for adaptive_rho in (True, False):
        rho_label = "arho" if adaptive_rho else "no_arho"
        selections = ("best_iter", "best_rho") if adaptive_rho else ("best_iter",)
        for alpha_mode in ("scalar", "vector"):
            for selection in selections:
                name = (
                    f"OSQP_python_neural_mlp_{alpha_mode}_"
                    f"{rho_label}_{selection}"
                )
                specs.append((name, adaptive_rho, alpha_mode, selection))
    return specs


def _checkpoint_path(
    checkpoint_dir: Path,
    problem_key: str,
    adaptive_rho: bool,
    alpha_mode: str,
    selection: str,
    fixed_control: bool = False,
) -> Path:
    prefix = (
        "best_model_control_fixed_nx100_dseed0"
        if fixed_control
        else f"best_model_{problem_key}"
    )
    filename = (
        f"{prefix}_precision=low_adaptive_rho={adaptive_rho}"
        f"_alpha_mode={alpha_mode}_model_type=mlp_loss=log_convergence"
    )
    if selection == "best_rho":
        filename += "_best_rho"
    return checkpoint_dir / f"{filename}.pt"


def _register_solvers(
    checkpoint_dir: Path,
    problem_key: str,
    *,
    fixed_control: bool = False,
) -> list[str]:
    names: list[str] = []
    for name, adaptive_rho, alpha_mode, selection in _solver_specs():
        if alpha_mode is None:
            constructor = OSQPPythonSolver
        else:
            checkpoint = _checkpoint_path(
                checkpoint_dir,
                problem_key,
                adaptive_rho,
                alpha_mode,
                selection or "best_iter",
                fixed_control=fixed_control,
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            constructor = partial(
                NeuralOSQPSolver,
                checkpoint_path=str(checkpoint),
                alpha_mode=alpha_mode,
                model_type="mlp",
                record_history=False,
            )
        solver_registry.SOLVER_MAP[name] = constructor
        solver_registry.settings[name] = _settings(adaptive_rho)
        names.append(name)
    return names


def _comparison_row(
    suite: str,
    identifier: str,
    solver: str,
    status: str,
    iterations: int,
    rho_updates: float,
    objective: float,
    expected: pd.Series,
) -> dict:
    status_match = status == str(expected["status"])
    iteration_match = iterations == int(expected["iter"])
    rho_match = rho_updates == float(expected["rho_updates"])
    objective_match = bool(
        np.isclose(objective, float(expected["obj_val"]), rtol=1e-8, atol=1e-7)
    )
    passed = status_match and iteration_match and rho_match and objective_match
    return {
        "suite": suite,
        "case": identifier,
        "solver": solver,
        "status": status,
        "expected_status": str(expected["status"]),
        "iterations": iterations,
        "expected_iterations": int(expected["iter"]),
        "rho_updates": rho_updates,
        "expected_rho_updates": float(expected["rho_updates"]),
        "objective": objective,
        "expected_objective": float(expected["obj_val"]),
        "objective_abs_diff": abs(objective - float(expected["obj_val"])),
        "passed": passed,
    }


def run_table1() -> list[dict]:
    expected = pd.read_csv(
        ROOT / "reference_results" / "smoke_expected_table1.csv"
    )
    rows: list[dict] = []
    for display_name, problem_key, dimension in TABLE1_PROBLEMS:
        names = _register_solvers(TABLE1_CHECKPOINT_DIR, problem_key)
        runner = Example(display_name, [dimension], names, solver_registry.settings, "", 1)
        for name in names:
            result = runner.solve_single_example(
                dimension, 0, name, solver_registry.settings[name]
            ).iloc[0]
            reference = expected[
                (expected["problem"] == display_name)
                & (expected["n"] == dimension)
                & (expected["instance"] == 0)
                & (expected["solver"] == name)
            ]
            if len(reference) != 1:
                raise RuntimeError(
                    f"Expected one Table 1 reference row, found {len(reference)}: "
                    f"{display_name}, n={dimension}, solver={name}"
                )
            rows.append(
                _comparison_row(
                    "table1",
                    f"{display_name} n={dimension} instance=0",
                    name,
                    str(result["status"]),
                    int(result["iter"]),
                    float(result["rho_updates"]),
                    float(result["obj_val"]),
                    reference.iloc[0],
                )
            )
    return rows


def run_table2() -> list[dict]:
    expected = pd.read_csv(
        ROOT / "reference_results" / "smoke_expected_table2.csv"
    )
    names = _register_solvers(
        TABLE2_CHECKPOINT_DIR, "control_fixed", fixed_control=True
    )
    base = ControlExample(100, seed=0)
    rows: list[dict] = []
    for name in names:
        for instance in range(3):
            x0_seed = 240 + instance
            rng = np.random.default_rng(x0_seed)
            raw = rng.random(base.nx)
            x0 = 0.5 * base.xmin + raw * (0.5 * base.xmax - 0.5 * base.xmin)
            base.update_x0(x0)
            qp = base.qp_problem

            solver = solver_registry.SOLVER_MAP[name]()
            solver.setup(
                P=qp["P"],
                q=qp["q"],
                A=qp["A"],
                l=qp["l"].copy(),
                u=qp["u"].copy(),
                **solver_registry.settings[name],
            )
            result = solver.solve()
            reference = expected[
                (expected["x0_seed"] == x0_seed)
                & (expected["solver"] == name)
            ]
            if len(reference) != 1:
                raise RuntimeError(
                    f"Expected one Table 2 reference row, found {len(reference)}: "
                    f"x0_seed={x0_seed}, solver={name}"
                )
            rows.append(
                _comparison_row(
                    "table2",
                    f"Control_fixed nx=100 x0_seed={x0_seed}",
                    name,
                    str(result.status),
                    int(result.niter),
                    float(result.rho_updates),
                    float(result.obj_val),
                    reference.iloc[0],
                )
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("all", "table1", "table2"),
        default="all",
        help="Regression subset to run (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for a detailed comparison CSV.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, module=r"cvxpy.*")
    started = time.perf_counter()
    rows: list[dict] = []
    if args.suite in ("all", "table1"):
        rows.extend(run_table1())
    if args.suite in ("all", "table2"):
        rows.extend(run_table2())

    for row in rows:
        label = "PASS" if row["passed"] else "FAIL"
        print(
            f"[{label}] {row['suite']} | {row['case']} | {row['solver']} | "
            f"iter={row['iterations']} (expected {row['expected_iterations']})"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"Wrote {args.output}")

    n_passed = sum(bool(row["passed"]) for row in rows)
    elapsed = time.perf_counter() - started
    print(f"Summary: {n_passed}/{len(rows)} passed in {elapsed:.1f} s")
    return 0 if n_passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
