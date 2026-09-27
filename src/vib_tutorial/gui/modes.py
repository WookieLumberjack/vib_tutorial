"""The two modal-analysis methods the user can switch between, as rows for the GUI.

Both methods are reduced to the same list of ModeEntry rows so the table,
mode-shape plot, phasor plot, release button and tune/fit combos can treat
them alike.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from ..core import ModalResult
from .style import colors


class Method(enum.Enum):
    CLASSICAL = "Classical: N real modes (undamped Kφ = ω²Mφ)"
    STATE_SPACE = "State-space: 2N complex modes (damped ż = Az)"


@dataclass(frozen=True)
class ModeEntry:
    key: str  # row label, e.g. "Mode 2" or "λ3"; also identifies it for the fit combo
    shape: np.ndarray  # complex displacement shape, largest |entry| = 1 (real for classical)
    spin: int  # +1 / -1: the shape rotates as e^{+iωt} / e^{-iωt}; 0 if non-oscillatory
    freq_hz: float  # frequency to tune / fit to; 0 if none (rigid or non-oscillatory)
    color: str
    legend: str | None  # mode-shape legend text; None hides it (the conjugate of a pair)
    state0: np.ndarray  # real [x, v] released by "Release selected mode", largest |x| = 1

    @property
    def listed(self) -> bool:
        """Offered in the tune / fit combos (skips rigid modes and conjugate duplicates)."""
        return self.freq_hz > 0 and self.legend is not None


def mode_entries(result: ModalResult, method: Method) -> list[ModeEntry]:
    if method is Method.CLASSICAL:
        return [
            ModeEntry(
                key=f"Mode {m.index}",
                shape=m.shape.astype(complex),
                spin=1,
                freq_hz=m.fn_hz,
                color=colors.mode[r % len(colors.mode)],
                legend=f"Mode {m.index}: {m.fn_hz:.3g} Hz",
                state0=np.concatenate([m.shape, np.zeros_like(m.shape)]),
            )
            for r, m in enumerate(result.modes)
        ]

    entries = []
    group = -1  # one color per conjugate pair or real root
    for m in result.complex_modes:
        first = m.conjugate is None or m.conjugate > m.index
        if first:
            group += 1
        if m.is_oscillatory:
            legend = f"λ{m.index},{m.conjugate}: {m.fd_hz:.3g} Hz" if first else None
        else:
            legend = f"λ{m.index}: {time_constant_text(m.eigenvalue.real)}"
        entries.append(
            ModeEntry(
                key=f"λ{m.index}",
                shape=m.shape,
                spin=int(np.sign(m.eigenvalue.imag)) if m.is_oscillatory else 0,
                freq_hz=m.fd_hz if m.is_oscillatory else 0.0,
                color=colors.mode[group % len(colors.mode)],
                legend=legend,
                # x(t) = Re(psi e^{lambda t}) needs x0 = Re(psi) and v0 = Re(lambda psi).
                # Re is the same for psi and its conjugate, so both release the same motion.
                state0=m.state_vector.real.copy(),
            )
        )
    return entries


def time_constant_text(root: float) -> str:
    """Describe a real eigenvalue by its decay time constant tau = -1/lambda."""
    return "rigid (τ = ∞)" if abs(root) < 1e-9 else f"τ = {-1.0 / root:.3g} s"
