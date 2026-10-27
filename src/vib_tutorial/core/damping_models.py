"""Four ways a structure loses energy, matched to one oscillator: viscous, hysteretic, Coulomb and Zener.

The *Damping models* page. Each model is matched to the oscillator m, k at its natural frequency: the same
energy per cycle as a viscous damper of ratio ζ there (at the amplitude X, for friction), and for the
viscoelastic (Zener, standard linear solid) element the same stiffness at f_n too, so every model rings at
nearly the same frequency. Then: the force over a harmonic cycle (the hysteresis loop) and the energy it
encloses, the dynamic stiffness K*(ω) = K' + iK'' and loss factor η = K''/K' against frequency, the
steady response to a harmonic force (friction by its describing function), and the free decay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import brentq

from .sdof import TWO_PI, Oscillator, free_response, respond


class Model(Enum):
    VISCOUS = "Viscous"
    HYSTERETIC = "Hysteretic"
    COULOMB = "Coulomb friction"
    ZENER = "Viscoelastic (Zener)"


MODELS = tuple(Model)


@dataclass(frozen=True)
class Matched:
    """The four models on m and k, matched at f_n to the damping ratio `zeta`.

    `amplitude` is where friction is matched (it dissipates πcω_nX² per cycle there), and `relax` puts
    the Zener element's loss peak at relax × f_n.
    """

    mass: float = 1.0
    stiffness: float = 400.0
    zeta: float = 0.05
    amplitude: float = 0.01  # m
    relax: float = 1.0

    def __post_init__(self) -> None:
        if not (self.mass > 0 and self.stiffness > 0 and self.zeta > 0 and self.amplitude > 0 and self.relax > 0):
            raise ValueError("m, k, ζ, X and the relaxation ratio must be positive")

    @property
    def omega_n(self) -> float:
        return math.sqrt(self.stiffness / self.mass)

    @property
    def natural_hz(self) -> float:
        return self.omega_n / TWO_PI

    @property
    def c(self) -> float:
        """The viscous damper: c = 2ζ√(km)."""
        return 2.0 * self.zeta * math.sqrt(self.stiffness * self.mass)

    @property
    def eta(self) -> float:
        """The hysteretic loss factor: η = 2ζ (the same energy per cycle at f_n)."""
        return 2.0 * self.zeta

    @property
    def friction(self) -> float:
        """The friction force F_f: 4F_fX = πcω_nX², so F_f = πcω_nX/4."""
        return math.pi * self.c * self.omega_n * self.amplitude / 4.0

    def zener(self) -> tuple[float, float, float]:
        """(k0, k1, τ): k0 in parallel with k1 in series with a dashpot c1 = k1τ.

        η(ω) = aωτ/(1 + (1 + a)ω²τ²) with a = k1/k0 peaks at ωτ = 1/√(1 + a); that is put at relax × ω_n,
        a is chosen so η(ω_n) = 2ζ, and k0 so the storage stiffness at ω_n is k.
        """
        target = self.eta

        def eta_at_wn(a: float) -> float:
            u = 1.0 / (self.relax * math.sqrt(1.0 + a))  # ω_n τ
            return a * u / (1.0 + (1.0 + a) * u * u) - target

        hi = 1.0
        while eta_at_wn(hi) < 0:
            hi *= 2.0
            if hi > 1e12:
                raise ValueError("no Zener element reaches that loss factor")
        a = brentq(eta_at_wn, 1e-12, hi, xtol=1e-14, rtol=1e-12)
        u = 1.0 / (self.relax * math.sqrt(1.0 + a))
        k0 = self.stiffness / (1.0 + a * u * u / (1.0 + u * u))
        return k0, a * k0, u / self.omega_n

    def dynamic_stiffness(self, model: Model, omega: np.ndarray, amplitude: float | None = None) -> np.ndarray:
        """K*(ω) = K' + iK'' of the spring and the damping element together (friction's at `amplitude`)."""
        w = np.asarray(omega, dtype=float)
        k = self.stiffness
        if model is Model.VISCOUS:
            return k + 1j * w * self.c
        if model is Model.HYSTERETIC:
            return k * (1.0 + 1j * self.eta) + 0 * w
        if model is Model.COULOMB:  # the describing function: c_eq = 4F_f/πωX, so K'' = 4F_f/πX
            X = self.amplitude if amplitude is None else amplitude
            return k + 1j * 4.0 * self.friction / (math.pi * X) + 0 * w
        k0, k1, tau = self.zener()
        s = 1j * w * tau
        return k0 + k1 * s / (1.0 + s)

    def loss_factor(self, model: Model, omega: np.ndarray, amplitude: float | None = None) -> np.ndarray:
        K = self.dynamic_stiffness(model, omega, amplitude)
        return K.imag / K.real

    def energy_per_cycle(self, model: Model, omega: np.ndarray, amplitude: float | None = None) -> np.ndarray:
        """The energy lost in one cycle of x = X sin ωt: πK''X² (exactly 4F_fX for friction)."""
        X = self.amplitude if amplitude is None else amplitude
        K = self.dynamic_stiffness(model, omega, X)
        return math.pi * K.imag * X * X

    def loop(self, model: Model, omega: float, points: int = 400,
             amplitude: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One steady cycle of x = X sin ωt: (x, the force the element and the spring carry, the damping part).

        Friction's is its true force, F_f sign(ẋ), not the describing function's ellipse; the Zener element's
        damping part is its force less k x.
        """
        X = self.amplitude if amplitude is None else amplitude
        th = np.linspace(0.0, TWO_PI, points)
        x = X * np.sin(th)
        k = self.stiffness
        if model is Model.COULOMB:
            damp = self.friction * np.sign(np.cos(th))
            damp[np.cos(th) == 0] = 0.0
        else:
            K = complex(self.dynamic_stiffness(model, np.array([omega]), X)[0])
            total = (K * X * np.exp(1j * (th - math.pi / 2))).real  # Re(K X e^{iωt}), x = Re(X e^{i(ωt − π/2)})
            damp = total - k * x
        return x, k * x + damp, damp

    def receptance(self, model: Model, freqs_hz: np.ndarray, force: float) -> np.ndarray:
        """|X|/F under F cos ωt (complex, from K* − ω²m); friction's by its describing function, magnitude only.

        With friction alone, X = √(F² − (4F_f/π)²)/|k − ω²m|: none where F ≤ 4F_f/π (it stays stuck, to this
        approximation) and infinite at f_n, which friction cannot bound.
        """
        w = TWO_PI * np.asarray(freqs_hz, dtype=float)
        if model is Model.COULOMB:
            push = force * force - (4.0 * self.friction / math.pi) ** 2
            with np.errstate(divide="ignore"):
                X = np.where(push > 0, math.sqrt(max(push, 0.0)) / np.abs(self.stiffness - w * w * self.mass), 0.0)
            return X / force + 0j
        K = self.dynamic_stiffness(model, w)
        return 1.0 / (K - w * w * self.mass)

    def free_decay(self, model: Model, x0: float, t: np.ndarray) -> np.ndarray:
        """The free vibration released from rest at x0 (after being held there long).

        Hysteretic damping has no time-domain model that is causal (constant K'' at every frequency, negative
        ones included, isn't); its curve is the viscous one of the same loss at f_n, as is usual.
        """
        t = np.asarray(t, dtype=float)
        if model in (Model.VISCOUS, Model.HYSTERETIC):
            x, _ = free_response(Oscillator(self.mass, self.stiffness, self.c), x0, 0.0, t)
            return x
        if model is Model.COULOMB:
            return coulomb_decay(self.mass, self.stiffness, self.friction, x0, t)
        k0, k1, tau = self.zener()
        c1 = k1 * tau
        m = self.mass
        A = np.array([[0.0, 1.0, 0.0], [-(k0 + k1) / m, 0.0, k1 / m], [k1 / c1, 0.0, -k1 / c1]])
        B = np.zeros((3, 1))
        h = t[1] - t[0]
        z = respond(A, B, h, np.zeros(t.size), np.array([x0, 0.0, x0]))  # held long: the arm relaxed
        return z[:, 0]


def coulomb_decay(mass: float, stiffness: float, friction: float, x0: float, t: np.ndarray) -> np.ndarray:
    """A mass on a spring sliding on friction F_f, released from rest at x0, in closed form.

    Each half cycle is a swing about ±F_f/k (the side the friction pushes it to), so the amplitude falls by
    2F_f/k each half cycle, 4F_f/k each cycle, a straight line; it stops at the first turn within F_f/k of
    rest.
    """
    t = np.asarray(t, dtype=float)
    wn = math.sqrt(stiffness / mass)
    half = math.pi / wn
    d = friction / stiffness
    x = np.empty_like(t)
    start, pos = 0.0, x0
    done = np.zeros(t.size, dtype=bool)
    for _ in range(100000):
        if abs(pos) <= d:  # stuck where it turned
            x[~done] = pos
            break
        e = math.copysign(d, pos)  # it moves towards zero; friction pushes back to +e side
        inside = (~done) & (t <= start + half)
        x[inside] = (pos - e) * np.cos(wn * (t[inside] - start)) + e
        done |= inside
        pos = 2 * e - pos
        start += half
        if done.all():
            break
    return x


def coulomb_stop(stiffness: float, friction: float, x0: float) -> tuple[int, float]:
    """(half cycles, where it stops) for a mass released from rest at x0 against friction F_f."""
    d = friction / stiffness
    pos, n = x0, 0
    while abs(pos) > d:
        pos = 2.0 * math.copysign(d, pos) - pos
        n += 1
    return n, pos
