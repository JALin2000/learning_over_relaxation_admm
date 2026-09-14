"""Numerical checks for the ARADMM baseline."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sparse

from solvers.aradmm import ARADMMOSQPSolver


def _spectral_estimate(gradient_delta, dual_delta, threshold=0.2):
    """Independent transcription of the released MATLAB estimator."""
    inner = float(gradient_delta @ dual_delta)
    gradient_norm = float(np.linalg.norm(gradient_delta))
    dual_norm = float(np.linalg.norm(dual_delta))
    denominator = gradient_norm * dual_norm
    if inner <= threshold * denominator + 1e-20:
        return False, np.nan
    sd_step = dual_norm**2 / inner
    mg_step = inner / gradient_norm**2
    hybrid_step = mg_step if 2.0 * mg_step > sd_step else sd_step - 0.5 * mg_step
    return True, hybrid_step


def _reference_parameter_trace(P, q, A, lower, upper, n_iterations):
    """Run the paper equations directly, without the OSQP implementation."""
    rho = 0.1
    alpha = 1.0
    z = np.zeros(A.shape[0])
    dual = np.zeros(A.shape[0])
    reference = None
    trace = [(0, rho, alpha)]

    for iteration in range(1, n_iterations):
        z_previous = z.copy()
        dual_previous = dual.copy()
        x = np.linalg.solve(
            P + rho * A.T @ A,
            A.T @ (dual + rho * z) - q,
        )
        Au = A @ x
        relaxed_Au = alpha * Au + (1.0 - alpha) * z_previous
        z = np.clip(relaxed_Au - dual_previous / rho, lower, upper)
        dual = dual_previous + rho * (-relaxed_Au + z)
        intermediate_dual = dual_previous + rho * (-Au + z_previous)
        current = (Au.copy(), -z.copy(), dual.copy(), intermediate_dual.copy())

        if iteration == 1:
            reference = current
            continue
        if iteration % 2:
            continue

        h_ok, h_step = _spectral_estimate(
            current[0] - reference[0], current[3] - reference[3]
        )
        g_ok, g_step = _spectral_estimate(
            current[1] - reference[1], current[2] - reference[2]
        )

        if h_ok and g_ok:
            rho = float(np.sqrt(h_step * g_step))
            alpha = min(
                1.0 + 2.0 * np.sqrt(h_step * g_step) / (h_step + g_step),
                1.5,
            )
        elif h_ok:
            rho, alpha = h_step, 1.9
        elif g_ok:
            rho, alpha = g_step, 1.1
        else:
            alpha = 1.5
        rho = float(np.clip(rho, 1e-6, 1e6))
        trace.append((iteration, rho, alpha))
        reference = current

    return trace


class ARADMMTests(unittest.TestCase):
    def test_matches_direct_equation_trace(self):
        P = np.array([[3.0, 0.2], [0.2, 2.0]])
        q = np.array([-1.0, 0.4])
        A = np.array([[1.0, 2.0], [-1.0, 1.0], [0.5, -0.25]])
        lower = np.array([-np.inf, -0.5, -0.2])
        upper = np.array([1.2, 0.8, 0.7])
        n_iterations = 20

        expected = _reference_parameter_trace(
            P, q, A, lower, upper, n_iterations
        )

        solver = ARADMMOSQPSolver(record_history=True)
        solver.setup(
            P=sparse.csc_matrix(P),
            q=q,
            A=sparse.csc_matrix(A),
            l=lower,
            u=upper,
            max_iter=n_iterations,
            eps_abs=1e-12,
            eps_rel=1e-12,
            eps_prim_inf=1e-15,
            eps_dual_inf=1e-15,
            polish=False,
            verbose=False,
            scaling=0,
            sigma=0.0,
            check_termination=False,
        )
        solver.solve()
        history = solver.get_aradmm_history()
        actual = list(zip(history['iter'], history['rho'], history['alpha']))

        self.assertEqual([row[0] for row in actual], [row[0] for row in expected])
        np.testing.assert_allclose(
            [row[1] for row in actual],
            [row[1] for row in expected],
            rtol=2e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [row[2] for row in actual],
            [row[2] for row in expected],
            rtol=0.0,
            atol=1e-14,
        )

    def test_converges_on_box_qp_with_scalar_equality_penalty(self):
        solver = ARADMMOSQPSolver(record_history=True)
        solver.setup(
            P=sparse.eye(2, format='csc'),
            q=np.array([-1.0, -2.0]),
            A=sparse.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
            l=np.array([1.0, 0.0, 0.0]),
            u=np.array([1.0, np.inf, np.inf]),
            max_iter=1000,
            eps_abs=1e-7,
            eps_rel=1e-7,
            eps_prim_inf=1e-15,
            eps_dual_inf=1e-15,
            polish=False,
            verbose=False,
        )

        np.testing.assert_allclose(
            solver._osqp._model.work.rho_vec,
            np.full(3, 0.1),
        )
        result = solver.solve()
        self.assertEqual(result.status, 'optimal')
        np.testing.assert_allclose(result.x, np.array([0.0, 1.0]), atol=2e-6)
        self.assertAlmostEqual(result.obj_val, -1.5, places=6)


if __name__ == '__main__':
    unittest.main()
