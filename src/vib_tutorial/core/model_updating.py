"""Model updating and correlation: a model of the chain set against a modal test of it.

The Simulation page's chain is the structure, the thing tested. The model is a copy of it with
errors in it, element by element (a spring 30 % too stiff, a mass too heavy): `Errors` holds them
as factors on the structure's values, so the model follows the structure when it changes. In
practice one never knows the structure, only the test; here both are at hand, so the
correlation's verdict, and later the updating's, can be checked against the truth.

The test
--------
`measure` gets the structure's modes as a test would: by the virtual modal test (hammer on m1,
H1, LSCF's stable poles and LSFD's residues, as `oma.hammer_modes`), whose modes carry its noise
and its identification errors, and may miss some; or, quickly, the exact modes with noise added
to their frequencies and shapes. Only the masses with a sensor are kept (`sensors`).

Correlation
-----------
* The MAC of each test shape with each model shape (the model's undamped φ), at the sensors:
  |ψᴴφ|² / (ψᴴψ φᵀφ). 1 when they are the same shape, 0 when orthogonal.
* Pairing: each test mode with the model mode its shape matches best, best MAC first, at least
  `MIN_MAC` (`pair`). Pairing by frequency alone goes wrong once the model's errors move the
  modes past each other; the shapes don't.
* For each pair, the frequency error (f_model − f_test)/f_test.
* Auto-MAC: the test shapes against each other. With too few sensors, or badly placed ones,
  different modes look alike (spatial aliasing) and the off-diagonal terms rise.
* COMAC, per sensor: (Σ_r |ψ_ir φ_ir|)² / (Σ_r |ψ_ir|² Σ_r φ_ir²) over the pairs, each test shape
  first scaled to its model shape (the modal scale factor). Where it drops, the shapes disagree,
  which points near the error.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

from .modal import ModalResult, modal_analysis
from .model import ChainSystem
from .oma import hammer_modes

MIN_MAC = 0.5  # a test mode pairs only with a model mode whose shape it matches at least this well
NOISE_F = 0.005  # the quick test's noise: frequencies (relative), damping ratios (relative), shapes (of the peak)
NOISE_ZETA = 0.05
NOISE_SHAPE = 0.02


class Source(enum.Enum):
    HAMMER = "Virtual modal test (hammer)"
    EXACT = "Exact modes + noise"


@dataclass(frozen=True)
class Errors:
    """The model's errors: factors on the structure's masses, springs and dampers, element by element."""

    mass: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray

    @classmethod
    def none(cls, n: int) -> Errors:
        return cls(np.ones(n), np.ones(n), np.ones(n))

    def resized(self, n: int) -> Errors:
        def fit(a: np.ndarray) -> np.ndarray:
            return a[:n] if n <= a.size else np.concatenate([a, np.ones(n - a.size)])

        return Errors(fit(self.mass), fit(self.stiffness), fit(self.damping))

    def model(self, structure: ChainSystem) -> ChainSystem:
        """The structure with these errors (friction left out: the model is linear)."""
        return ChainSystem(structure.masses * self.mass, structure.stiffness * self.stiffness,
                           structure.damping * self.damping)


@dataclass(frozen=True)
class TestMode:
    """One measured mode: its frequency, damping and complex shape at every mass (the sensors pick from it)."""

    fn_hz: float
    zeta: float
    shape: np.ndarray


@dataclass(frozen=True)
class Test:
    source: Source
    modes: list[TestMode]  # by frequency
    sensors: np.ndarray  # (N,) bool: which masses have a sensor

    def with_sensors(self, sensors: np.ndarray) -> Test:
        """The same test read at other sensors (the virtual test records every mass; this picks)."""
        return Test(self.source, self.modes, np.asarray(sensors, dtype=bool))

    def shapes(self) -> np.ndarray:
        """(sensors, modes) complex: the measured shapes at the sensors."""
        if not self.modes:
            return np.zeros((int(self.sensors.sum()), 0), dtype=complex)
        return np.column_stack([m.shape[self.sensors] for m in self.modes])


def sample_rate(result: ModalResult) -> float:
    """The virtual modal test page's: a passband that clears the highest mode by 25 %."""
    from .measurement import AA_CUTOFF

    f_top = max((m.fn_hz for m in result.complex_modes if m.is_oscillatory), default=1.0)
    need = 1.25 * f_top * 2.0 / AA_CUTOFF
    scale = 10.0 ** (math.floor(math.log10(need)) - 1)
    return math.ceil(need / scale) * scale


def measure(structure: ChainSystem, source: Source, sensors: np.ndarray, seed: int = 1,
            result: ModalResult | None = None) -> Test:
    """The structure's modes as a test sees them; correlation reads them at the sensors only."""
    linear = ChainSystem(structure.masses, structure.stiffness, structure.damping)
    result = result or modal_analysis(linear)
    sensors = np.asarray(sensors, dtype=bool)
    if source is Source.HAMMER:
        found, _ = hammer_modes(linear, result, sample_rate(result), seed=seed)
        modes = [(m.fn_hz, m.zeta, m.shape) for m in found]
    else:
        rng = np.random.default_rng(seed)
        modes = []
        for m in result.modes:
            if m.damped is None:
                continue
            psi = m.damped.shape
            noise = NOISE_SHAPE * (rng.standard_normal(psi.size) + 1j * rng.standard_normal(psi.size)) / math.sqrt(2)
            modes.append((m.damped.fn_hz * (1 + NOISE_F * rng.standard_normal()),
                          m.damped.zeta * (1 + NOISE_ZETA * rng.standard_normal()), psi + noise))
    out = [TestMode(float(fn), float(zeta), np.asarray(shape, dtype=complex))
           for fn, zeta, shape in sorted(modes, key=lambda t: t[0])]
    return Test(source, out, sensors)


def mac(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MAC of each column of a with each column of b: (a's columns, b's columns)."""
    num = np.abs(a.conj().T @ b) ** 2
    den = np.outer(np.sum(np.abs(a) ** 2, axis=0), np.sum(np.abs(b) ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0)


@dataclass(frozen=True)
class Pair:
    test: int  # 0-based, into Test.modes
    model: int  # 0-based undamped model mode
    mac: float
    f_test: float
    f_model: float

    @property
    def error(self) -> float:
        """(f_model − f_test) / f_test."""
        return self.f_model / self.f_test - 1.0


@dataclass(frozen=True)
class Correlation:
    model: ChainSystem
    result: ModalResult  # the model's modes
    test: Test
    macs: np.ndarray  # (test modes, model modes)
    pairs: list[Pair]  # by test mode
    auto_mac: np.ndarray  # (test, test)
    comac: np.ndarray  # (sensors,)

    def model_shapes(self) -> np.ndarray:
        """(sensors, model modes): the model's undamped shapes at the sensors."""
        return np.column_stack([m.shape for m in self.result.modes])[self.test.sensors]


def pair(macs: np.ndarray, f_test: list[float], f_model: list[float], min_mac: float = MIN_MAC) -> list[Pair]:
    """Each test mode with the model mode it matches best, best MAC first; none under min_mac."""
    out: list[Pair] = []
    used_t, used_m = set(), set()
    for i, j in sorted(np.ndindex(macs.shape), key=lambda ij: -macs[ij]):
        if macs[i, j] < min_mac:
            break
        if i in used_t or j in used_m:
            continue
        out.append(Pair(i, j, float(macs[i, j]), f_test[i], f_model[j]))
        used_t.add(i)
        used_m.add(j)
    return sorted(out, key=lambda p: p.test)


def comac(test: np.ndarray, model: np.ndarray, pairs: list[Pair]) -> np.ndarray:
    """COMAC at each sensor over the pairs: each test shape scaled to its model shape first."""
    if not pairs:
        return np.full(test.shape[0], np.nan)
    psi = np.column_stack([test[:, p.test] for p in pairs])
    phi = np.column_stack([model[:, p.model] for p in pairs])
    msf = np.sum(psi.conj() * phi, axis=0) / np.sum(np.abs(psi) ** 2, axis=0)  # least-squares scale onto φ
    psi = psi * msf
    num = np.sum(np.abs(psi * phi), axis=1) ** 2  # |ψ φ|: a test shape's phase scatter doesn't count against it
    den = np.sum(np.abs(psi) ** 2, axis=1) * np.sum(phi**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def correlate(structure: ChainSystem, errors: Errors, test: Test, min_mac: float = MIN_MAC) -> Correlation:
    model = errors.model(structure)
    result = modal_analysis(model)
    shapes = test.shapes()
    model_shapes = np.column_stack([m.shape for m in result.modes])[test.sensors]
    macs = mac(shapes, model_shapes)
    # the model's |λ|/2π, as the test measures it (the undamped frequency where a mode is overdamped)
    f_model = [m.damped.fn_hz if m.damped is not None else m.fn_hz for m in result.modes]
    pairs = pair(macs, [m.fn_hz for m in test.modes], f_model, min_mac)
    return Correlation(model, result, test, macs, pairs, mac(shapes, shapes), comac(shapes, model_shapes, pairs))
