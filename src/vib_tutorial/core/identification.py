"""Modal parameter extraction from a measured FRF column H(w) (force at one mass, every response).

Every method returns identified modes in pole-residue form,

    H_j(w) ~ sum_r [ R+_jr / (iw - lambda_r) + R-_jr / (iw - lambda_r*) ] + U_j - L_j / w^2

with lambda_r = -zeta_r w_r + i w_r sqrt(1 - zeta_r^2). The upper residual U_j
(a constant) stands in for modes above the fitted band, the lower residual
L_j (mass-like) for modes below it. For a single input the residues of one
mode are proportional to its mode shape, R+_jr ~ psi_jr psi_kr.

* Peak picking: a mode at each peak of the summed FRF magnitude. Its natural
  frequency is the peak's frequency line, its damping comes from the half-power
  bandwidth, zeta = (f2 - f1) / (2 f_r), and its shape from the FRF at the peak
  (at resonance H_j = A_j / (2 i zeta w_r^2)). Simple, but limited by the line
  spacing, and biased when modes are close or heavily damped.
* Circle fit (Kennedy-Pancu): near a resonance the mobility iwH traces a circle.
  The angle each point makes about the centre gives w_r and zeta together,
  between the frequency lines; each response's modal constant comes from a
  local fit with a constant for the other modes.
* LSCF + LSFD: a common-denominator rational fraction fitted to every response
  at once (least-squares complex frequency domain, as in PolyMAX), at a range
  of model orders. Physical poles stay put as the order rises; noise poles do
  not, which the stabilization diagram shows. The chosen poles are then fitted
  with residues and residual terms by linear least squares (LSFD).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import numpy as np

from .modal import TWO_PI, ModalResult

PEAK_PROMINENCE = 0.05  # decades of the mode indicator a peak must rise above its surroundings
MIN_CIRCLE_POINTS = 5
STABLE_FREQ = 0.01  # relative change allowed between orders for a "stable" pole
STABLE_DAMPING = 0.05
AUTO_RUN = 5  # orders a column of stable poles must span for automatic selection
MAX_ZETA = 0.5  # poles more damped than this are not taken as modes
MATCH_FREQ = 0.1  # an identified mode further than this from an exact one is not that mode


class Method(enum.Enum):
    PEAK = "Peak picking (half-power)"
    CIRCLE = "Circle fit"
    LSCF = "LSCF + LSFD (stabilization)"


class Stability(enum.Enum):
    NEW = "new"  # no match at the previous order
    FREQUENCY = "frequency"  # frequency within STABLE_FREQ
    STABLE = "stable"  # frequency and damping within their limits


@dataclass(frozen=True)
class IdentifiedMode:
    pole: complex  # lambda, Im > 0
    plus: np.ndarray  # (n,) residues R+ of lambda
    minus: np.ndarray  # (n,) residues R- of lambda*

    @property
    def fn_hz(self) -> float:
        return abs(self.pole) / TWO_PI

    @property
    def fd_hz(self) -> float:
        return self.pole.imag / TWO_PI

    @property
    def zeta(self) -> float:
        return -self.pole.real / abs(self.pole)

    @property
    def shape(self) -> np.ndarray:
        """Complex shape from the residues, largest entry normalized to 1+0j."""
        return self.plus / self.plus[np.argmax(np.abs(self.plus))]

    def shifted(self, sigma: float) -> IdentifiedMode:
        """The same mode with its pole moved sigma to the right (undoing an exponential window)."""
        return IdentifiedMode(complex(self.pole.real + sigma, self.pole.imag), self.plus, self.minus)


@dataclass
class Identification:
    method: Method
    modes: list[IdentifiedMode]
    upper: np.ndarray  # (n,) upper residual U
    lower: np.ndarray  # (n,) lower residual L
    notes: list[str] = field(default_factory=list)

    def synthesize(self, freqs_hz: np.ndarray) -> np.ndarray:
        """The fitted FRF, shape (len(freqs), n)."""
        w = TWO_PI * np.asarray(freqs_hz, dtype=float)
        iw = 1j * w[:, None]
        H = np.zeros((w.size, self.upper.size), dtype=complex) + self.upper[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            H = H - self.lower[None, :] / (w**2)[:, None]
        for m in self.modes:
            H = H + m.plus[None, :] / (iw - m.pole) + m.minus[None, :] / (iw - m.pole.conjugate())
        return H

    def shifted(self, sigma: float) -> Identification:
        """Every pole moved sigma to the right; the residuals are kept."""
        modes = [m.shifted(sigma) for m in self.modes]
        return Identification(self.method, modes, self.upper, self.lower, list(self.notes))


def band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return (freqs > 0) & (freqs >= lo) & (freqs <= hi)


def mode_indicator(H: np.ndarray) -> np.ndarray:
    """Sum over responses of |H_j| / max |H_j|: every response counts equally, peaks at every mode."""
    mag = np.abs(H)
    peak = mag.max(axis=0)
    return (mag / np.where(peak > 0, peak, 1.0)).sum(axis=1)


def pole_from(fn_hz: float, zeta: float) -> complex:
    wn = TWO_PI * fn_hz
    return complex(-zeta * wn, wn * math.sqrt(max(0.0, 1.0 - zeta**2)))


def sdof_mode(pole: complex, A: np.ndarray) -> IdentifiedMode:
    """The mode A / (w_r^2 - w^2 + 2i zeta w_r w) = A / ((iw - lambda)(iw - lambda*)) in residue form."""
    d = pole - pole.conjugate()
    return IdentifiedMode(pole, A / d, -A / d)


def _empty(method: Method, n: int, modes: list[IdentifiedMode], notes: list[str]) -> Identification:
    return Identification(method, modes, np.zeros(n, dtype=complex), np.zeros(n, dtype=complex), notes)


# ------------------------------------------------------------ peak picking
@dataclass(frozen=True)
class Peak:
    index: int  # into the full frequency array
    channel: int  # response with the largest |H| there
    fn_hz: float
    zeta: float


def find_peaks(freqs: np.ndarray, H: np.ndarray, band: tuple[float, float], max_modes: int) -> list[Peak]:
    """Peaks of the mode indicator in the band, with half-power damping; at most max_modes, lowest first."""
    import scipy.signal  # deferred: scipy.signal adds ~0.9 s to startup

    idx = np.flatnonzero(band_mask(freqs, band))
    if idx.size < 3:
        return []
    mif = np.log10(np.maximum(mode_indicator(H[idx]), 1e-300))
    found, props = scipy.signal.find_peaks(mif, prominence=PEAK_PROMINENCE)
    strongest = np.argsort(props["prominences"])[::-1][:max_modes]
    peaks = []
    for p in sorted(found[strongest]):
        g = int(idx[p])
        c = int(np.argmax(np.abs(H[g])))
        zeta = half_power_damping(freqs[idx], np.abs(H[idx, c]), p)
        if zeta is not None:
            peaks.append(Peak(g, c, float(freqs[g]), zeta))
    return peaks


def half_power_damping(f: np.ndarray, mag: np.ndarray, p: int) -> float | None:
    """zeta = (f2 - f1) / (2 f_p) from where |H| falls to peak / sqrt(2), interpolated between lines.

    With only one side inside the data, that side's half-bandwidth is doubled.
    """
    level = mag[p] / math.sqrt(2.0)
    f1 = f2 = None
    for i in range(p, 0, -1):
        if mag[i - 1] < level:
            f1 = f[i - 1] + (level - mag[i - 1]) * (f[i] - f[i - 1]) / (mag[i] - mag[i - 1])
            break
    for i in range(p, mag.size - 1):
        if mag[i + 1] < level:
            f2 = f[i] + (mag[i] - level) * (f[i + 1] - f[i]) / (mag[i] - mag[i + 1])
            break
    if f1 is None and f2 is None:
        return None
    width = (f2 - f1) if f1 is not None and f2 is not None else 2.0 * ((f2 or f[p]) - (f1 or f[p]))
    return abs(width) / (2.0 * f[p])


def peak_picking(freqs: np.ndarray, H: np.ndarray, band: tuple[float, float], max_modes: int) -> Identification:
    n = H.shape[1]
    modes = []
    for pk in find_peaks(freqs, H, band, max_modes):
        pole = pole_from(pk.fn_hz, pk.zeta)
        wr = TWO_PI * pk.fn_hz
        A = 2j * pk.zeta * wr**2 * H[pk.index]  # at resonance H = A / (2i zeta w_r^2)
        modes.append(sdof_mode(pole, A))
    notes = [] if modes else ["No peaks found in the band."]
    return _empty(Method.PEAK, n, modes, notes)


# -------------------------------------------------------------- circle fit
def fit_circle(z: np.ndarray) -> tuple[complex, float]:
    """Least-squares (Kasa) circle through complex points: (centre, radius)."""
    x, y = z.real, z.imag
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    (a, b, c), *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    return complex(a, b), math.sqrt(max(c + a**2 + b**2, 0.0))


def circle_fit(freqs: np.ndarray, H: np.ndarray, band: tuple[float, float], max_modes: int) -> Identification:
    """Each peak's w_r and zeta from the angles on the mobility circle; each response's constant locally."""
    n = H.shape[1]
    modes, notes = [], []
    df = float(np.median(np.diff(freqs))) if freqs.size > 1 else 1.0
    mask = band_mask(freqs, band)
    for pk in find_peaks(freqs, H, band, max_modes):
        half = max(3.0 * pk.zeta * pk.fn_hz, 2.5 * df)
        sel = np.flatnonzero(mask & (np.abs(freqs - pk.fn_hz) <= half))
        if sel.size < MIN_CIRCLE_POINTS:
            notes.append(f"{pk.fn_hz:.3g} Hz: only {sel.size} lines on the circle; kept the peak-picking values.")
            wr, zeta = TWO_PI * pk.fn_hz, pk.zeta
        else:
            wr, zeta = _circle_parameters(TWO_PI * freqs[sel], H[sel, pk.channel], TWO_PI * pk.fn_hz, pk.zeta)
            if not band[0] <= wr / TWO_PI <= band[1]:
                notes.append(f"{pk.fn_hz:.3g} Hz: the circle fit left the band; kept the peak-picking values.")
                wr, zeta = TWO_PI * pk.fn_hz, pk.zeta
        pole = pole_from(wr / TWO_PI, zeta)
        # Each response: H_j = A_j / (w_r^2 - w^2 + 2i zeta w_r w) + B_j over the same lines.
        pts = sel if sel.size >= 2 else np.array([pk.index])
        w = TWO_PI * freqs[pts]
        basis = np.column_stack([1.0 / (wr**2 - w**2 + 2j * zeta * wr * w), np.ones(w.size)])
        coef = np.linalg.lstsq(basis, H[pts], rcond=None)[0] if pts.size >= 2 else None
        A = coef[0] if coef is not None else 2j * zeta * wr**2 * H[pk.index]
        modes.append(sdof_mode(pole, A))
    if not modes:
        notes.append("No peaks found in the band.")
    return _empty(Method.CIRCLE, n, modes, notes)


def _circle_parameters(w: np.ndarray, h: np.ndarray, wr0: float, zeta0: float) -> tuple[float, float]:
    """(w_r, zeta) from the angles of the mobility points about the fitted circle's centre.

    For Y = iwH near an isolated mode, Y - centre turns by -2 atan(x) from the
    resonance point, x = (w^2 - w_r^2) / (2 zeta w_r w). The resonance
    direction, w_r and zeta are fitted to the measured angles.
    """
    import scipy.optimize  # deferred like scipy.signal in find_peaks

    Y = 1j * w * h
    centre, _ = fit_circle(Y)
    phi = np.angle(Y - centre)
    k = int(np.argmin(np.abs(w - wr0)))

    def residual(p):
        phi_r, wr, zeta = p
        x = (w**2 - wr**2) / (2.0 * zeta * wr * w)
        return np.angle(np.exp(1j * (phi - phi_r + 2.0 * np.arctan(x))))

    lower = [-np.inf, 0.5 * wr0, 1e-6]
    upper = [np.inf, 1.5 * wr0, 1.0]
    start = [phi[k], wr0, min(max(zeta0, 1e-4), 0.99)]  # a noise "peak" can be wider than critical
    fit = scipy.optimize.least_squares(residual, start, bounds=(lower, upper))
    return float(fit.x[1]), float(fit.x[2])


# ------------------------------------------------------------------- LSCF
@dataclass
class Stabilization:
    """Poles of the LSCF fit at every model order (only physical ones: Im > 0, stable, in the band)."""

    orders: list[int]
    poles: list[np.ndarray]  # per order, complex
    status: list[list[Stability]]
    runs: list[np.ndarray]  # per order and pole: consecutive lower orders it has stayed stable over
    band: tuple[float, float]

    def pole(self, order: int, i: int) -> complex:
        return complex(self.poles[self.orders.index(order)][i])


def lscf(freqs: np.ndarray, H: np.ndarray, band: tuple[float, float], max_order: int) -> Stabilization:
    """Fit a common denominator of every order 1..max_order; keep each order's physical poles.

    The basis is Omega^r = e^{i w dt r} with dt = 1 / (2 f_max): the band maps onto
    half the unit circle, so the normal equations stay well conditioned. For
    order p, the denominator coefficients alpha (alpha_p = 1) minimize
    sum_j sum_f |W_j (N_j - H_j D)|^2 after the numerators N_j are eliminated.
    A root z of the denominator is the pole lambda = ln(z) / dt.
    """
    mask = band_mask(freqs, band)
    f, h = freqs[mask], H[mask]
    lo, hi = band
    f_top = float(f.max()) if f.size else hi
    dt = 1.0 / (2.0 * f_top)
    P = min(max_order, f.size - 1)  # order p needs at least p + 1 lines
    Om = np.exp(1j * TWO_PI * f[:, None] * dt * np.arange(P + 1)[None, :])  # (F, P+1)
    R0 = (Om.conj().T @ Om).real
    rms = np.sqrt(np.mean(np.abs(h) ** 2, axis=0))
    weights = 1.0 / np.where(rms > 0, rms, 1.0)
    S_all, T_all = [], []
    for j in range(h.shape[1]):
        hj = h[:, j] * weights[j]
        S_all.append(-(Om.conj().T @ (hj[:, None] * Om)).real)
        T_all.append((Om.conj().T @ ((np.abs(hj) ** 2)[:, None] * Om)).real)

    orders, poles, status, runs = [], [], [], []
    for p in range(1, P + 1):
        k = p + 1
        R = R0[:k, :k]
        M = np.zeros((k, k))
        try:
            for S, T in zip(S_all, T_all):
                Sp = S[:k, :k]
                M += T[:k, :k] - Sp.T @ np.linalg.solve(R, Sp)
            alpha = np.append(-np.linalg.solve(M[:p, :p], M[:p, p]), 1.0)
        except np.linalg.LinAlgError:
            continue
        z = np.roots(alpha[::-1])
        z = z[np.abs(z) > 0]
        lam = np.log(z.astype(complex)) / dt
        fn = np.abs(lam) / TWO_PI
        keep = (lam.imag > 0) & (lam.real < 0) & (fn >= lo) & (fn <= hi)
        lam = lam[keep]
        lam = lam[np.argsort(np.abs(lam))]  # by frequency
        st, run = _classify(lam, poles[-1] if poles else None, runs[-1] if runs else None)
        orders.append(p)
        poles.append(lam)
        status.append(st)
        runs.append(run)
    return Stabilization(orders, poles, status, runs, band)


def _classify(lam: np.ndarray, prev: np.ndarray | None, prev_runs: np.ndarray | None):
    status, runs = [], np.zeros(lam.size, dtype=int)
    for i, l in enumerate(lam):
        if prev is None or not prev.size:
            status.append(Stability.NEW)
            continue
        j = int(np.argmin(np.abs(np.abs(prev) - abs(l))))
        df = abs(abs(prev[j]) - abs(l)) / abs(l)
        z0, z1 = -prev[j].real / abs(prev[j]), -l.real / abs(l)
        dz = abs(z1 - z0) / max(z1, 1e-12)
        if df > STABLE_FREQ:
            status.append(Stability.NEW)
        elif dz > STABLE_DAMPING:
            status.append(Stability.FREQUENCY)
        else:
            status.append(Stability.STABLE)
            runs[i] = prev_runs[j] + 1
    return status, runs


def auto_select(stab: Stabilization) -> list[tuple[int, int]]:
    """(order, index) of one pole per column of the diagram that is stable over at least AUTO_RUN orders.

    Stable poles (frequency and damping) within STABLE_FREQ of each other are one
    column, wherever in the range of orders they appear: with noisy data a mode
    is often clearest at middle orders, and noise poles crowd in at high ones.
    Each column is represented by its pole with the longest run of stable
    orders (the highest such order on a tie).
    """
    stable = [
        (abs(l), stab.runs[k][i], order, i)
        for k, order in enumerate(stab.orders)
        for i, (l, st) in enumerate(zip(stab.poles[k], stab.status[k]))
        if st is Stability.STABLE and -l.real / abs(l) <= MAX_ZETA
    ]
    stable.sort()
    columns: list[list[tuple]] = []
    for item in stable:
        if columns and item[0] - columns[-1][-1][0] <= STABLE_FREQ * item[0]:
            columns[-1].append(item)
        else:
            columns.append([item])
    chosen = []
    for col in columns:
        if len({order for _, _, order, _ in col}) < AUTO_RUN:
            continue
        _, _, order, i = max(col, key=lambda t: (t[1], t[2]))
        chosen.append((order, i))
    return chosen


def lsfd(freqs: np.ndarray, H: np.ndarray, band: tuple[float, float], poles: list[complex]) -> Identification:
    """Residues of the given poles and residual terms, by linear least squares over the band.

    The measured system is real, so a pole's conjugate has the conjugate
    residue: per response the unknowns are Re R, Im R for each pole, U and L.
    """
    mask = band_mask(freqs, band)
    f, h = freqs[mask], H[mask]
    n = H.shape[1]
    if not poles or f.size < 2 * len(poles) + 2:
        note = "No poles selected." if not poles else "Too few lines in the band for the poles selected."
        return _empty(Method.LSCF, n, [], [note])
    iw = 1j * TWO_PI * f
    cols = []
    for lam in poles:
        a, b = 1.0 / (iw - lam), 1.0 / (iw - np.conj(lam))
        cols += [a + b, 1j * (a - b)]
    cols += [np.ones_like(iw), -1.0 / (TWO_PI * f) ** 2]
    B = np.column_stack(cols)
    A = np.vstack([B.real, B.imag])
    rhs = np.vstack([h.real, h.imag])
    scale = np.linalg.norm(A, axis=0)
    x = np.linalg.lstsq(A / scale, rhs, rcond=None)[0] / scale[:, None]
    modes = []
    for r, lam in enumerate(poles):
        R = x[2 * r] + 1j * x[2 * r + 1]
        modes.append(IdentifiedMode(complex(lam), R, R.conj()))
    m = len(poles)
    return Identification(Method.LSCF, modes, x[2 * m].astype(complex), x[2 * m + 1].astype(complex))


# ------------------------------------------------------------- comparison
@dataclass(frozen=True)
class ModeMatch:
    mode: int | None  # 1-based exact mode number; None: spurious (no exact mode matched)
    exact: complex | None  # exact pole
    identified: IdentifiedMode | None  # None: missed
    mac: float


def mac(a: np.ndarray, b: np.ndarray) -> float:
    num = abs(np.vdot(a, b)) ** 2
    den = np.vdot(a, a).real * np.vdot(b, b).real
    return float(num / den) if den > 0 else 0.0


def mac_matrix(modes: list[IdentifiedMode], result: ModalResult) -> np.ndarray:
    """MAC of each identified shape (rows) with each exact damped mode shape (columns)."""
    exact = [m.damped.shape for m in result.modes if m.damped is not None]
    return np.array([[mac(m.shape, e) for e in exact] for m in modes]).reshape(len(modes), len(exact))


def match_modes(modes: list[IdentifiedMode], result: ModalResult, f_max: float) -> list[ModeMatch]:
    """Pair identified modes with exact ones (below f_max): best MAC first, frequencies within MATCH_FREQ.

    Exact modes left over are missed; identified ones left over are spurious.
    """
    exact = [m for m in result.modes if m.damped is not None]
    macs = mac_matrix(modes, result)
    pairs: dict[int, int] = {}
    order = sorted(
        ((macs[i, j], i, j) for i in range(len(modes)) for j in range(len(exact))),
        key=lambda t: -t[0],
    )
    used_i, used_j = set(), set()
    for value, i, j in order:
        if i in used_i or j in used_j:
            continue
        fe = exact[j].damped.fn_hz
        if abs(modes[i].fn_hz - fe) > MATCH_FREQ * fe:
            continue
        pairs[j] = i
        used_i.add(i)
        used_j.add(j)
    out = []
    for j, m in enumerate(exact):
        if j in pairs:
            i = pairs[j]
            out.append(ModeMatch(m.index, m.damped.eigenvalue, modes[i], float(macs[i, j])))
        elif m.damped.fn_hz <= f_max:
            out.append(ModeMatch(m.index, m.damped.eigenvalue, None, 0.0))
    for i, m in enumerate(modes):
        if i not in used_i:
            out.append(ModeMatch(None, None, m, 0.0))
    out.sort(key=lambda r: abs(r.exact) if r.exact is not None else abs(r.identified.pole))
    return out
