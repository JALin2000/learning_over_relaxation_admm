"""ARADMM parameter adaptation embedded in the repository's OSQP solver.

This is a formula-level Python reimplementation of Adaptive Relaxed ADMM
(ARADMM):

    Z. Xu, M. A. T. Figueiredo, X. Yuan, C. Studer, and T. Goldstein,
    "Adaptive Relaxed ADMM: Convergence Theory and Practical
    Implementation," CVPR 2017.

The update schedule and practical safeguards follow the authors' released
MATLAB implementation:
https://github.com/nightldj/admm_release/tree/master/2017-cvpr-aradmm

For an apples-to-apples comparison with the paper experiments in this
repository, ARADMM reuses the same scaled OSQP iterations and termination
criteria.  Unlike stock OSQP, it uses the single scalar penalty assumed by
ARADMM for every non-loose constraint (including equality constraints).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solvers.osqppurepy import _osqp
from solvers.osqppurepy import OSQP as _OSQPInterface


def _hybrid_spectral_step(sd_step: float, mg_step: float) -> float:
    """Return the hybrid BB spectral estimate used by ARADMM."""
    if 2.0 * mg_step > sd_step:
        return mg_step
    return sd_step - 0.5 * mg_step


class _ScalarPenaltyOSQP(_osqp.OSQP):
    """Pure-Python OSQP variant with one rho for every active constraint."""

    def _set_scalar_rho_values(self) -> None:
        work = self.work
        work.settings.rho = float(
            np.clip(work.settings.rho, _osqp.RHO_MIN, _osqp.RHO_MAX)
        )

        loose = np.logical_and(
            work.data.l < -_osqp.OSQP_INFTY * _osqp.MIN_SCALING,
            work.data.u > _osqp.OSQP_INFTY * _osqp.MIN_SCALING,
        )
        equality = (work.data.u - work.data.l < _osqp.RHO_TOL) & ~loose
        inequality = ~(loose | equality)

        work.constr_type[loose] = -1
        work.constr_type[equality] = 1
        work.constr_type[inequality] = 0

        work.rho_vec[loose] = _osqp.RHO_MIN
        work.rho_vec[~loose] = work.settings.rho
        work.rho_inv_vec = np.reciprocal(work.rho_vec)

    def set_rho_vec(self) -> None:
        """Initialize the scalar ARADMM penalty vector."""
        self._set_scalar_rho_values()

    def update_rho_vec(self) -> None:
        """Refresh constraint types while retaining a scalar active penalty."""
        old_constraint_types = np.copy(self.work.constr_type)
        self._set_scalar_rho_values()
        if not np.array_equal(old_constraint_types, self.work.constr_type):
            self.work.linsys_solver = _osqp.linsys_solver(self.work)

    def update_rho(self, rho_new: float) -> None:
        """Set scalar rho and refactor the KKT system."""
        if not np.isfinite(rho_new) or rho_new <= 0.0:
            raise ValueError('rho must be positive and finite')

        self.work.settings.rho = float(
            np.clip(rho_new, _osqp.RHO_MIN, _osqp.RHO_MAX)
        )
        loose = self.work.constr_type == -1
        self.work.rho_vec[loose] = _osqp.RHO_MIN
        self.work.rho_vec[~loose] = self.work.settings.rho
        self.work.rho_inv_vec = np.reciprocal(self.work.rho_vec)
        self.work.linsys_solver = _osqp.linsys_solver(self.work)


@dataclass(frozen=True)
class _ARADMMSnapshot:
    """Quantities required by the spectral update at one ADMM iteration."""

    Au: np.ndarray
    Bv: np.ndarray
    dual: np.ndarray
    intermediate_dual: np.ndarray


class _ARADMMCallback:
    """Update scalar rho and alpha between OSQP iterations."""

    def __init__(
        self,
        model: _ScalarPenaltyOSQP,
        update_interval: int = 2,
        adaptation_start: int = 2,
        adaptation_end: int = 1000,
        correlation_threshold: float = 0.2,
        minimum_value: float = 1e-20,
        record_history: bool = False,
    ):
        if update_interval <= 0:
            raise ValueError('update_interval must be positive')
        if adaptation_start < 2:
            raise ValueError('adaptation_start must be at least 2')
        if adaptation_end < adaptation_start:
            raise ValueError('adaptation_end must not precede adaptation_start')
        if not 0.0 <= correlation_threshold <= 1.0:
            raise ValueError('correlation_threshold must be in [0, 1]')

        self.model = model
        self.work = model.work
        self.update_interval = update_interval
        self.adaptation_start = adaptation_start
        self.adaptation_end = adaptation_end
        self.correlation_threshold = correlation_threshold
        self.minimum_value = minimum_value
        self.record_history = record_history

        # The callback is called at the beginning of an OSQP iteration.  Thus,
        # call 1 precedes iteration 1 and observes no completed iteration.
        self.call_count = 0
        self.reference: _ARADMMSnapshot | None = None
        self._z_before_iteration = np.copy(self.work.z)
        self._y_before_iteration = np.copy(self.work.y)
        self._rho_during_iteration = float(self.work.settings.rho)
        self.history: dict[str, list] = {
            'iter': [0],
            'rho': [float(self.work.settings.rho)],
            'alpha': [float(self.work.settings.alpha_z)],
            'rho_changed': [False],
            'h_accepted': [False],
            'g_accepted': [False],
            'h_correlation': [np.nan],
            'g_correlation': [np.nan],
        }

    def _snapshot(self) -> _ARADMMSnapshot:
        work = self.work
        n = work.data.n

        # In the OSQP KKT solve, z_tilde is A @ x_tilde.  It corresponds to
        # A u in the ARADMM notation before over-relaxation.
        Au = np.copy(work.xz_tilde[n:])
        Bv = -np.copy(work.z)  # B = -I for A x - z = 0.

        # ARADMM uses the opposite dual sign from OSQP for this splitting:
        # lambda = -y.  The OSQP solve loop overwrites work.z_prev before this
        # callback runs, so retain the start-of-iteration z, y, and rho here.
        intermediate_dual = (
            -self._y_before_iteration
            - self._rho_during_iteration * (Au - self._z_before_iteration)
        )

        return _ARADMMSnapshot(
            Au=Au,
            Bv=Bv,
            dual=-np.copy(work.y),
            intermediate_dual=np.asarray(intermediate_dual),
        )

    def _save_next_iteration_start(self) -> None:
        """Remember the state and parameters used by the next ADMM step."""
        self._z_before_iteration = np.copy(self.work.z)
        self._y_before_iteration = np.copy(self.work.y)
        self._rho_during_iteration = float(self.work.settings.rho)

    def _spectral_estimate(
        self,
        gradient_delta: np.ndarray,
        dual_delta: np.ndarray,
    ) -> tuple[bool, float, float]:
        inner = float(np.dot(gradient_delta, dual_delta))
        gradient_norm = float(np.linalg.norm(gradient_delta))
        dual_norm = float(np.linalg.norm(dual_delta))
        denominator = gradient_norm * dual_norm
        correlation = inner / denominator if denominator > 0.0 else np.nan

        accepted = (
            np.isfinite(inner)
            and np.isfinite(denominator)
            and inner
            > self.correlation_threshold * denominator + self.minimum_value
        )
        if not accepted:
            return False, np.nan, correlation

        sd_step = dual_norm * dual_norm / inner
        mg_step = inner / (gradient_norm * gradient_norm)
        estimate = _hybrid_spectral_step(sd_step, mg_step)
        accepted = np.isfinite(estimate) and estimate > 0.0
        return accepted, float(estimate) if accepted else np.nan, correlation

    def _record(
        self,
        completed_iteration: int,
        rho_changed: bool,
        h_accepted: bool,
        g_accepted: bool,
        h_correlation: float,
        g_correlation: float,
    ) -> None:
        if not self.record_history:
            return
        self.history['iter'].append(completed_iteration)
        self.history['rho'].append(float(self.work.settings.rho))
        self.history['alpha'].append(float(self.work.settings.alpha_z))
        self.history['rho_changed'].append(rho_changed)
        self.history['h_accepted'].append(h_accepted)
        self.history['g_accepted'].append(g_accepted)
        self.history['h_correlation'].append(h_correlation)
        self.history['g_correlation'].append(g_correlation)

    def __call__(self) -> None:
        self.call_count += 1
        completed_iteration = self.call_count - 1

        if completed_iteration == 0:
            return

        if completed_iteration == 1:
            self.reference = self._snapshot()
            self._save_next_iteration_start()
            return

        if (
            completed_iteration < self.adaptation_start
            or completed_iteration > self.adaptation_end
            or completed_iteration % self.update_interval != 0
        ):
            self._save_next_iteration_start()
            return

        current = self._snapshot()
        if self.reference is None:
            self.reference = current
            self._save_next_iteration_start()
            return

        delta_Au = current.Au - self.reference.Au
        delta_intermediate_dual = (
            current.intermediate_dual - self.reference.intermediate_dual
        )
        delta_Bv = current.Bv - self.reference.Bv
        delta_dual = current.dual - self.reference.dual

        h_ok, h_step, h_corr = self._spectral_estimate(
            delta_Au, delta_intermediate_dual
        )
        g_ok, g_step, g_corr = self._spectral_estimate(delta_Bv, delta_dual)

        old_rho = float(self.work.settings.rho)
        if h_ok and g_ok:
            new_rho = float(np.sqrt(h_step * g_step))
            raw_alpha = 1.0 + 2.0 * np.sqrt(h_step * g_step) / (h_step + g_step)
            # The paper's formula is uncapped, but the authors' released
            # practical implementation caps this particular branch at 1.5.
            new_alpha = min(float(raw_alpha), 1.5)
        elif h_ok:
            new_rho = h_step
            new_alpha = 1.9
        elif g_ok:
            new_rho = g_step
            new_alpha = 1.1
        else:
            new_rho = old_rho
            new_alpha = 1.5

        self.work.settings.alpha_x = float(new_alpha)
        self.work.settings.alpha_z = float(new_alpha)

        clipped_rho = float(np.clip(new_rho, _osqp.RHO_MIN, _osqp.RHO_MAX))
        rho_changed = clipped_rho != old_rho
        if rho_changed:
            self.model.update_rho(clipped_rho)
            self.work.info.rho_updates += 1

        self._record(
            completed_iteration,
            rho_changed,
            h_ok,
            g_ok,
            h_corr,
            g_corr,
        )
        # The reference implementation advances k_0 after every attempted
        # spectral update, including when neither curvature estimate passes.
        self.reference = current
        self._save_next_iteration_start()

    def get_history(self) -> dict[str, list] | None:
        return self.history if self.record_history else None


class ARADMMOSQPSolver:
    """Benchmark-compatible ARADMM solver using the pure-Python OSQP core."""

    def __init__(
        self,
        update_interval: int = 2,
        adaptation_start: int = 2,
        adaptation_end: int = 1000,
        correlation_threshold: float = 0.2,
        initial_rho: float = 0.1,
        initial_alpha: float = 1.0,
        record_history: bool = False,
    ):
        self.update_interval = update_interval
        self.adaptation_start = adaptation_start
        self.adaptation_end = adaptation_end
        self.correlation_threshold = correlation_threshold
        self.initial_rho = initial_rho
        self.initial_alpha = initial_alpha
        self.record_history = record_history

        self._osqp = _OSQPInterface()
        self._osqp._model = _ScalarPenaltyOSQP()
        self._callback: _ARADMMCallback | None = None

    def version(self):
        return self._osqp.version()

    def constant(self, name):
        return self._osqp.constant(name)

    def setup(self, P=None, q=None, A=None, l=None, u=None, **settings):
        """Set up the common OSQP iteration with ARADMM parameter updates."""
        settings['rho'] = self.initial_rho
        settings['alpha_x'] = self.initial_alpha
        settings['alpha_z'] = self.initial_alpha
        settings['adaptive_rho'] = False
        settings['learnt_component'] = 'alpha'
        settings.setdefault('scaling', 10)

        self._osqp.setup(P=P, q=q, A=A, l=l, u=u, **settings)
        model = self._osqp._model
        self._callback = _ARADMMCallback(
            model=model,
            update_interval=self.update_interval,
            adaptation_start=self.adaptation_start,
            adaptation_end=self.adaptation_end,
            correlation_threshold=self.correlation_threshold,
            record_history=self.record_history,
        )
        model.work.learnt_component_callback = self._callback

    def solve(self, total_iters=None):
        return self._osqp.solve(total_iters=total_iters)

    def get_aradmm_history(self) -> dict[str, list] | None:
        if self._callback is None:
            return None
        return self._callback.get_history()

    def warm_start(self, x=None, y=None):
        return self._osqp.warm_start(x=x, y=y)

    def update(self, **kwargs):
        return self._osqp.update(**kwargs)

    def update_settings(self, **kwargs):
        return self._osqp.update_settings(**kwargs)
