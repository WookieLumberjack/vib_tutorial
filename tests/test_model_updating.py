"""Model updating and correlation: the test, the MAC and its pairs, the auto-MAC and COMAC, the trials' notes."""

import numpy as np
import pytest

from vib_tutorial.core.modal import modal_analysis
from vib_tutorial.core.model import ChainSystem
from vib_tutorial.core.model_updating import (
    Errors,
    Source,
    comac,
    correlate,
    mac,
    measure,
    pair,
)

CHAIN = ChainSystem.uniform(4)
ALL = np.ones(4, dtype=bool)


def errors(**factors) -> Errors:
    e = Errors.none(4)
    for key, (i, f) in factors.items():
        getattr(e, key)[i] = f
    return e


def test_mac_and_pairing():
    a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    m = mac(a * np.exp(0.4j), a)
    assert m == pytest.approx(np.array([[1.0, 0.25], [0.25, 1.0]]))
    assert mac(np.array([[1.0], [1.0]]), np.array([[1.0], [-1.0]]))[0, 0] == pytest.approx(0.0)
    # by shape, best first, whatever the frequencies' order; nothing under the threshold
    pairs = pair(np.array([[0.1, 0.9, 0.0], [0.95, 0.2, 0.0], [0.3, 0.4, 0.45]]), [1.0, 2.0, 3.0], [1.5, 0.8, 3.0])
    assert [(p.test, p.model) for p in pairs] == [(0, 1), (1, 0)]
    assert pairs[0].error == pytest.approx(0.8 - 1.0)
    shapes = np.array([[1.0, 0.5], [0.4, -1.0], [-0.3, 0.2]])
    assert comac(shapes * 2j, shapes, pair(mac(shapes, shapes), [1, 2], [1, 2])) == pytest.approx(np.ones(3))


def test_the_tests():
    hammer = measure(CHAIN, Source.HAMMER, ALL, seed=1)
    exact = measure(CHAIN, Source.EXACT, ALL, seed=1)
    assert len(hammer.modes) == 3 and len(exact.modes) == 4  # the hammer test misses the most damped
    r = modal_analysis(CHAIN)
    for test in (hammer, exact):
        c = correlate(CHAIN, Errors.none(4), test)
        assert [(p.test, p.model) for p in c.pairs] == [(i, i) for i in range(len(test.modes))]
        assert max(abs(p.error) for p in c.pairs) < 0.005
        assert [m.fn_hz for m in test.modes] == pytest.approx([m.fn_hz for m in r.modes[:len(test.modes)]], rel=0.005)
    # the sensors pick from the same measurement
    two = exact.with_sensors(np.array([1, 0, 0, 1], dtype=bool))
    assert two.shapes().shape == (2, 4) and two.modes is exact.modes


def test_uniform_errors_leave_the_shapes():
    test = measure(CHAIN, Source.EXACT, ALL, seed=2)
    base = correlate(CHAIN, Errors.none(4), test)
    soft = correlate(CHAIN, Errors(np.ones(4), np.full(4, 0.9), np.ones(4)), test)
    heavy = correlate(CHAIN, Errors(np.full(4, 1 / 0.9), np.ones(4), np.ones(4)), test)
    for c in (soft, heavy):
        assert c.macs == pytest.approx(base.macs, abs=1e-12)
        for p, q in zip(c.pairs, base.pairs):
            assert p.f_model == pytest.approx(np.sqrt(0.9) * q.f_model)


def test_the_trials_notes():
    from vib_tutorial.gui.model_updating import TRIALS

    note = {name: text for name, _, text in TRIALS}
    r = modal_analysis(CHAIN)

    def shifts(e: Errors) -> list[float]:
        m = modal_analysis(e.model(CHAIN))
        return [round(100 * (a.fn_hz / b.fn_hz - 1), 1) for a, b in zip(m.modes, r.modes)]

    assert shifts(errors(stiffness=(2, 1.3))) == [2.2, 4.1, 0.6, 6.6]
    assert "2.2, 4.1, 0.6 and 6.6 %" in note["One spring 30 % too stiff"]
    assert min(correlate(CHAIN, errors(stiffness=(2, 1.3)), measure(CHAIN, Source.EXACT, ALL)).macs.diagonal()) > 0.98
    assert shifts(errors(mass=(1, 1.2))) == [-1.8, -3.2, -0.5, -3.5]
    assert "1.8, 3.2, 0.5 and 3.5 %" in note["One mass 20 % too heavy"]
    assert shifts(Errors(np.ones(4), np.full(4, 0.9), np.ones(4))) == [-5.1] * 4
    doubled = modal_analysis(Errors(np.ones(4), np.ones(4), np.full(4, 2.0)).model(CHAIN))
    assert [m.damped.zeta / e.damped.zeta for m, e in zip(doubled.modes, r.modes)] == pytest.approx([2.0] * 4)

    two = measure(CHAIN, Source.EXACT, np.array([1, 0, 0, 1], dtype=bool), seed=1)
    c = correlate(CHAIN, errors(stiffness=(2, 1.3)), two)
    assert c.auto_mac[1, 3] == pytest.approx(0.94, abs=0.03)  # with the test's noise
    shapes = np.column_stack([m.damped.shape for m in r.modes])[[0, 3]]
    assert mac(shapes, shapes)[1, 3] == pytest.approx(0.91, abs=0.005) and "0.91 without noise" in note["Two sensors"]
    assert [(p.test, p.model) for p in c.pairs] == [(i, i) for i in range(4)]

    c = correlate(CHAIN, Errors.none(4), measure(CHAIN, Source.HAMMER, ALL, seed=1))
    assert max(abs(p.error) for p in c.pairs) < 0.003 and c.pairs[2].mac == pytest.approx(0.92, abs=0.005)
    assert "0.3 %" in note["No error: the test's own noise"] and "0.92" in note["No error: the test's own noise"]


def test_frequency_sensitivity():
    """The closed form against finite differences: springs by their strain energy, masses by their kinetic."""
    from vib_tutorial.core.model_updating import (
        Kind,
        frequency_sensitivity,
        parameters,
        updated_errors,
    )

    e = errors(stiffness=(2, 1.3))
    params = parameters(4, [Kind.SPRING, Kind.MASS, Kind.DAMPER])
    model = e.model(CHAIN)
    sens = frequency_sensitivity(model, modal_analysis(model), params)
    base = np.array([m.fn_hz for m in modal_analysis(model).modes])
    h = 1e-6
    for j in range(len(params)):
        theta = np.zeros(len(params))
        theta[j] = h
        moved = np.array([m.fn_hz for m in modal_analysis(updated_errors(e, params, theta).model(CHAIN)).modes])
        assert sens[:, j] == pytest.approx(np.log(moved / base) / h, abs=1e-5)
    # every spring together: +1/2 in every mode; every mass: −1/2
    assert sens[:, :4].sum(axis=1) == pytest.approx(np.full(4, 0.5))
    assert sens[:, 4:8].sum(axis=1) == pytest.approx(np.full(4, -0.5))


def test_updating():
    from vib_tutorial.core.model_updating import Kind, Residuals, parameters, update

    wrong = errors(stiffness=(2, 1.3))
    exact = measure(CHAIN, Source.EXACT, ALL, seed=1)
    springs = parameters(4, [Kind.SPRING])
    # four frequencies, four springs: found from the frequencies alone, to the test's noise
    u = update(CHAIN, wrong, exact, springs, Residuals(shapes=False))
    assert u.factors() == pytest.approx(np.ones(4), abs=0.025) and u.history[-1].cost < 1e-6
    assert u.factors(u.history[0].errors) == pytest.approx([1.0, 1.0, 1.3, 1.0])
    # masses as well: one direction (all scaled together) the data can't see
    u = update(CHAIN, wrong, exact, parameters(4, [Kind.SPRING, Kind.MASS]))
    sv = u.history[1].singular
    assert np.sum(sv < 1e-3 * sv[0]) == 1
    # a strong regularization holds the model where it was
    u = update(CHAIN, wrong, exact, springs, regularization=100.0)
    assert u.factors() == pytest.approx([1.0, 1.0, 1.3, 1.0], abs=0.005)
    # dampers from the damping ratios
    u = update(CHAIN, Errors(np.ones(4), np.ones(4), np.full(4, 2.0)), exact, parameters(4, [Kind.DAMPER]),
               Residuals(shapes=False, damping=True))
    assert u.factors() == pytest.approx(np.ones(4), abs=0.1)
    assert len(update(CHAIN, wrong, exact, []).history) == 1  # nothing to update


def test_the_update_trials_notes():
    from vib_tutorial.core.model_updating import Kind, Residuals, parameters, update
    from vib_tutorial.gui.model_updating import DEFAULT_UPDATE, TRIALS

    note = {name: text for name, _, text in TRIALS}
    hammer = measure(CHAIN, Source.HAMMER, ALL, seed=1)
    exact = measure(CHAIN, Source.EXACT, ALL, seed=1)
    wrong = errors(stiffness=(2, 1.3))
    what = Residuals(DEFAULT_UPDATE["shapes"], DEFAULT_UPDATE["damping"], DEFAULT_UPDATE["weight"])
    springs = parameters(4, [Kind.SPRING])

    f = update(CHAIN, wrong, hammer, springs, what).factors()
    assert f[2] == pytest.approx(1.02, abs=0.005) and np.max(np.abs(f - 1)) < 0.035
    assert "1.30 to 1.02" in note["Update: find the stiff spring"] and "3.5 %" in note["Update: find the stiff spring"]
    f = update(CHAIN, wrong, exact, parameters(4, [Kind.SPRING, Kind.MASS]), what).factors()
    assert np.all((f > 1.015) & (f < 1.045)) and "2 to 4 % high" in note["Update: masses too, not unique"]
    f = update(CHAIN, wrong, hammer, springs, what, regularization=1.0).factors()
    assert f[2] == pytest.approx(1.29, abs=0.005) and "1.29" in note["Update: regularized"]
    f = update(CHAIN, Errors(np.ones(4), np.ones(4), np.full(4, 2.0)), hammer, parameters(4, [Kind.DAMPER]),
               Residuals(shapes=False, damping=True)).factors()
    assert np.max(np.abs(f - 1)) < 0.15 and "15 %" in note["Update: the dampers"]
