"""The Numerical integration page's model (core/integration.py): the schemes' orders, limits and maps."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vib_tutorial.core.integration import (
    SCHEMES,
    STABLE_LIMIT,
    Load,
    Model,
    Scheme,
    amplification,
    exact,
    simulate,
    spectral,
    stable_limit,
)


def test_closed_form_matches_the_ode():
    m, load = Model(damping=3.0, stiff_hz=40.0), Load(0.01, 2.0, 1.7)
    M, C, K = m.matrices()
    f = load.at(m)

    def rhs(t, z):
        x, v = z[:2], z[2:]
        return np.concatenate([v, np.linalg.solve(M, f(t) - C @ v - K @ x)])

    t = np.linspace(0, 2, 801)
    ref = solve_ivp(rhs, (0, 2), [0.01, 0, 0, 0], t_eval=t, rtol=1e-11, atol=1e-14).y[0]
    assert exact(m, load, t) == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("scheme, order", [(Scheme.CENTRAL, 2), (Scheme.AVERAGE, 2), (Scheme.LINEAR, 2),
                                           (Scheme.HHT, 2), (Scheme.RK4, 4), (Scheme.EULER, 1), (Scheme.FOH, 2)])
def test_each_scheme_converges_at_its_order(scheme, order):
    m, load = Model(damping=1.0), Load(0.01, 1.0, 2.0)
    T = 1 / m.frequencies_hz()[0]
    errs = []
    for per in (80, 160):
        h = T / per
        n = round(2 * T / h)
        x = simulate(scheme, m, load, h, n, -0.1)
        errs.append(np.max(np.abs(x - exact(m, load, np.arange(n + 1) * h))))
    assert math.log2(errs[0] / errs[1]) == pytest.approx(order, abs=0.25)


@pytest.mark.parametrize("scheme", [s for s in SCHEMES if s is not Scheme.CENTRAL])
def test_the_amplification_matrix_is_the_steppers_free_map(scheme):
    m = Model(1.0, 1.0, 0.1)  # ω = 1, ζ = 0.05
    h = 0.7
    x = simulate(scheme, m, Load(1.0), h, 6, -0.2)
    G = amplification(scheme, h, 0.05, -0.2)
    z = np.array([1.0, 0.0]) if G.shape[0] == 2 else np.array([1.0, 0.0, -1.0])  # a₀ = −x₀ for ω = 1
    for i in range(1, 7):
        z = G @ z
        assert z[0] == pytest.approx(x[i], abs=1e-12)


def test_stability_limits():
    for scheme, limit in STABLE_LIMIT.items():
        assert stable_limit(scheme) == pytest.approx(limit, rel=1e-6)
    assert spectral(Scheme.FOH, 1.3, 0.05).damping == pytest.approx(0.05)
    assert spectral(Scheme.AVERAGE, 1.0).radius == pytest.approx(1.0)
    assert spectral(Scheme.HHT, 1e4, 0.0, -1 / 3).radius == pytest.approx(0.5, abs=0.01)  # ρ∞ = (1 + α)/(1 − α)
