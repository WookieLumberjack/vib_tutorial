import numpy as np
import pytest

from vib_tutorial.core import (
    Acquisition,
    ChainResponse,
    ChainSystem,
    Estimator,
    Excitation,
    FrfEstimator,
    Processing,
    MeasurementSettings,
    Response,
    Window,
    foh_discretize,
    frf,
    from_receptance,
    impact_spectrum,
    modal_analysis,
    state_space,
    to_receptance,
    transfer,
)

SYSTEM = ChainSystem.uniform(4)


def measure(system: ChainSystem, **settings) -> Acquisition:
    acq = Acquisition(system, MeasurementSettings(seed=1, **settings))
    while not acq.done:
        acq.step()
    return acq


def band_error(system: ChainSystem, acq: Acquisition, processing: Processing, exact=None) -> float:
    """Largest |H - exact| inside the excitation band, relative to the exact peak."""
    est = FrfEstimator(acq, processing).estimate()
    s = acq.settings
    keep = (est.freqs > 0.05) & (est.freqs < s.band)
    ref = frf(system, est.freqs, s.input_dof) if exact is None else exact(est.freqs)
    return float(np.abs(est.H[keep] - ref[keep]).max() / np.abs(ref[keep]).max())


def test_fast_response_matches_stepping_the_foh_update():
    h = 1 / 160
    f = np.random.default_rng(0).standard_normal(3000)
    fast = ChainResponse(SYSTEM, h, 3)
    x = np.vstack([fast.run(f[:1000]), fast.run(f[1000:])])  # state carries across calls
    A, B = state_space(SYSTEM)
    Phi, G0, G1 = foh_discretize(A, B, h)
    z, f0, ref = np.zeros(8), 0.0, []
    for f1 in f:
        z = Phi @ z + G0[:, 3] * f0 + G1[:, 3] * f1
        ref.append(z[:4])
        f0 = f1
    np.testing.assert_allclose(x, ref, atol=1e-10 * np.abs(ref).max())


@pytest.mark.parametrize("h", [1 / 100, 1 / 2000])
def test_fast_response_of_a_free_chain(h):
    # The rigid-body lambda = 0 is double and defective. At h = 1/100 the eigenvectors of Phi are
    # still independent enough to diagonalize; at h = 1/2000 they are not, and it steps directly.
    free = ChainSystem([1.0] * 3, [0.0, 400.0, 400.0], [0.0, 2.0, 2.0])
    f = np.random.default_rng(0).standard_normal(5000)
    x = ChainResponse(free, h, 2).run(f)
    A, B = state_space(free)
    Phi, G0, G1 = foh_discretize(A, B, h)
    z, f0, ref = np.zeros(6), 0.0, []
    for f1 in f:
        z = Phi @ z + G0[:, 2] * f0 + G1[:, 2] * f1
        ref.append(z[:3])
        f0 = f1
    np.testing.assert_allclose(x, ref, atol=1e-8 * np.abs(ref).max())


@pytest.mark.parametrize("system", [SYSTEM, ChainSystem([1.0] * 3, [0.0, 400.0, 400.0], [0.0, 2.0, 2.0])])
def test_acceleration_is_the_equation_of_motion_at_each_sample(system):
    # a = M^-1 (f - C v - K x) from the same states: exact, not a difference of displacements.
    h, j = 1 / 160, system.n - 1
    f = np.random.default_rng(0).standard_normal(2000)
    a = ChainResponse(system, h, j, Response.ACCELERATION).run(f)
    A, B = state_space(system)
    Phi, G0, G1 = foh_discretize(A, B, h)
    n = system.n
    z, f0, ref = np.zeros(2 * n), 0.0, []
    for f1 in f:
        z = Phi @ z + G0[:, j] * f0 + G1[:, j] * f1
        ref.append(A[n:] @ z + B[n:, j] * f1)
        f0 = f1
    np.testing.assert_allclose(a, ref, atol=1e-8 * np.abs(ref).max())


def test_accelerance_and_receptance_convert_both_ways():
    f = np.linspace(0.0, 8.0, 41)
    H = frf(SYSTEM, f, 1)
    A = from_receptance(f, H, Response.ACCELERATION)
    np.testing.assert_allclose(A[1:], -((2 * np.pi * f[1:]) ** 2)[:, None] * H[1:], rtol=1e-12)
    back = to_receptance(f, A, Response.ACCELERATION)
    assert np.isnan(back[0]).all()  # 0 Hz: an accelerance says nothing about the static receptance
    np.testing.assert_allclose(back[1:], H[1:], rtol=1e-12)
    assert to_receptance(f, H, Response.DISPLACEMENT) is H


@pytest.mark.parametrize("excitation", [Excitation.PERIODIC_RANDOM, Excitation.STEPPED_SINE, Excitation.IMPACT])
def test_noise_free_acceleration_measurement_recovers_the_accelerance(excitation):
    acq = measure(SYSTEM, excitation=excitation, input_dof=3, averages=3, sine_points=30,
                  response=Response.ACCELERATION)
    est = FrfEstimator(acq, Processing(window=Window.RECTANGULAR)).estimate()
    assert est.response is Response.ACCELERATION
    err = band_error(SYSTEM, acq, Processing(window=Window.RECTANGULAR),
                     lambda f: from_receptance(f, frf(SYSTEM, f, 3), Response.ACCELERATION))
    assert err < 5e-3


@pytest.mark.parametrize(
    "excitation", [Excitation.PERIODIC_RANDOM, Excitation.CHIRP, Excitation.STEPPED_SINE, Excitation.IMPACT]
)
def test_noise_free_measurement_recovers_the_exact_frf(excitation):
    acq = measure(SYSTEM, excitation=excitation, input_dof=3, averages=3, sine_points=30)
    err = band_error(SYSTEM, acq, Processing(window=Window.RECTANGULAR))
    assert err < 5e-3  # transients are waited out to SETTLE_TOL = 0.1%, not to zero


def test_leakage_needs_a_window_for_continuous_random():
    # Leakage scatters the estimate at every line, not just at the peaks: compare the typical error.
    acq = measure(SYSTEM, excitation=Excitation.RANDOM, input_dof=3, averages=20, overlap=0.5)
    s = acq.settings

    def typical_error(window):
        est = FrfEstimator(acq, Processing(window=window)).estimate()
        keep = (est.freqs > 0.05) & (est.freqs < s.band)
        exact = frf(SYSTEM, est.freqs[keep], 3)[:, 3]
        return float(np.median(np.abs(est.H[keep, 3] - exact) / np.abs(exact)))

    assert typical_error(Window.HANN) < 0.2 * typical_error(Window.RECTANGULAR)


def test_exponential_window_adds_known_damping():
    # A short block: the hit is still ringing at its end. The exponential window fixes the leakage,
    # and what it measures is exactly H(iw + 1/tau): every pole moved left by 1/tau.
    acq = measure(SYSTEM, excitation=Excitation.IMPACT, input_dof=3, block=256, averages=3)
    p = Processing(window=Window.FORCE_EXPONENTIAL, exp_end=0.01)
    tau = p.exp_tau(acq.settings)
    shifted = band_error(SYSTEM, acq, p, lambda f: transfer(SYSTEM, 2j * np.pi * f + 1 / tau, 3))
    assert shifted < 5e-3
    assert band_error(SYSTEM, acq, p) > 0.2  # far from the undamped-by-window FRF at the peaks
    assert band_error(SYSTEM, acq, Processing(window=Window.RECTANGULAR)) > 0.1  # leakage


def test_transfer_on_the_imaginary_axis_is_the_receptance():
    f = np.linspace(0.1, 8.0, 50)
    np.testing.assert_allclose(transfer(SYSTEM, 2j * np.pi * f, 1), frf(SYSTEM, f, 1), rtol=1e-12)


def test_force_noise_biases_h1_low_and_leaves_h2():
    acq = measure(SYSTEM, excitation=Excitation.PERIODIC_RANDOM, input_dof=3, averages=60, block=512)
    fd = modal_analysis(SYSTEM).modes[1].damped.fd_hz
    est = {}
    for e in Estimator:
        est[e] = FrfEstimator(acq, Processing(window=Window.RECTANGULAR, estimator=e, force_noise=0.3)).estimate()
    i = int(np.argmin(np.abs(est[Estimator.H1].freqs - fd)))
    exact = abs(frf(SYSTEM, est[Estimator.H1].freqs[i : i + 1], 3)[0, 3])
    h1 = abs(est[Estimator.H1].H[i, 3]) / exact
    h2 = abs(est[Estimator.H2].H[i, 3]) / exact
    coh = est[Estimator.H1].coherence[i, 3]
    assert h1 < 0.8 and abs(h2 - 1) < 0.1
    assert h1 / h2 == pytest.approx(coh, rel=1e-9)  # coherence = H1 / H2


def test_noise_free_coherence_is_one_and_noise_lowers_it():
    acq = measure(SYSTEM, excitation=Excitation.PERIODIC_RANDOM, input_dof=3, averages=5)
    keep = slice(1, int(acq.settings.band / acq.settings.resolution))
    clean = FrfEstimator(acq, Processing(window=Window.RECTANGULAR)).estimate()
    noisy = FrfEstimator(acq, Processing(window=Window.RECTANGULAR, response_noise=0.05)).estimate()
    assert clean.coherence[keep].min() > 1 - 1e-6
    assert noisy.coherence[keep].min() < 0.9


def test_stepped_sine_noise_shows_in_the_samples_and_averages_out_of_the_fit():
    acq = measure(SYSTEM, excitation=Excitation.STEPPED_SINE, input_dof=3, sine_points=20)
    clean = FrfEstimator(acq, Processing()).estimate()
    noisy = FrfEstimator(acq, Processing(force_noise=0.2, response_noise=0.2)).estimate()
    # The samples carry the full 20% of the channel's peak...
    residual = np.std(noisy.f - clean.f) / np.abs(clean.f).max()
    assert residual == pytest.approx(0.2, rel=0.3)
    # ...but fitting a sine to N samples leaves about 20% x sqrt(2 / N) in its amplitude.
    ratio = noisy.H[:, 3] / clean.H[:, 3]
    assert 0.0 < np.median(np.abs(ratio - 1)) < 0.1
    # The fitted sine passes through the clean samples.
    t, f_fit, x_fit = clean.fit
    np.testing.assert_allclose(np.interp(clean.t, t, f_fit), clean.f, atol=0.02 * np.abs(clean.f).max())
    assert x_fit.shape == (t.size, SYSTEM.n)


def test_fewer_averages_restart_the_sums():
    acq = measure(SYSTEM, excitation=Excitation.RANDOM, input_dof=3, averages=6, block=256)
    est = FrfEstimator(acq, Processing(response_noise=0.05))
    six = est.estimate()
    acq.settings.averages = 2
    two = est.estimate()
    fresh = FrfEstimator(acq, Processing(response_noise=0.05)).estimate()
    assert (six.count, two.count) == (6, 2)
    np.testing.assert_allclose(two.H, fresh.H)


def test_overlap_gives_more_blocks_from_less_data():
    none = measure(SYSTEM, excitation=Excitation.RANDOM, averages=8, block=256, overlap=0.0)
    half = measure(SYSTEM, excitation=Excitation.RANDOM, averages=8, block=256, overlap=0.5)
    assert none.samples == 8 * 256 and half.samples == 256 + 7 * 128
    # Overlapping blocks share their samples, noise included.
    f0, _, n0, _ = half.block(0)
    f1, _, n1, _ = half.block(1)
    np.testing.assert_array_equal(f0[128:], f1[:128])
    np.testing.assert_array_equal(n0[128:], n1[:128])


def test_aliasing_without_the_anti_alias_filter():
    # One mass at 14 Hz, sampled at 20 Hz: without the filter its resonance folds to 6 Hz.
    sdof = ChainSystem([1.0], [(2 * np.pi * 14.0) ** 2], [2.0])
    common = dict(excitation=Excitation.IMPACT, fs=20.0, block=512, averages=1, tip_width=0.15)
    ratios = {}
    for aa in (True, False):
        acq = measure(sdof, anti_alias=aa, **common)
        est = FrfEstimator(acq, Processing(window=Window.RECTANGULAR)).estimate()
        i = int(np.argmin(np.abs(est.freqs - 6.0)))
        ratios[aa] = abs(est.H[i, 0]) / abs(frf(sdof, est.freqs[i : i + 1], 0)[0, 0])
    assert ratios[True] == pytest.approx(1.0, rel=0.02)
    assert ratios[False] > 1.5


def test_impact_spectrum_of_a_half_sine():
    width = 0.01
    f = np.array([0.0, 1 / (2 * width), 1.5 / width])
    np.testing.assert_allclose(impact_spectrum(width, f), [1.0, np.pi / 4, 0.0], atol=1e-12)
    # Against the FFT of a finely sampled pulse.
    fs = 1e5
    t = np.arange(round(width * fs) + 1) / fs
    pulse = np.sin(np.pi * t / width)
    F = np.abs(np.fft.rfft(pulse, 1 << 16))
    freqs = np.fft.rfftfreq(1 << 16, 1 / fs)
    keep = freqs < 1.2 / width
    np.testing.assert_allclose(F[keep] / F[0], impact_spectrum(width, freqs[keep]), atol=2e-3)


# ------------------------------------------------------------------ friction
FRICTION = ChainSystem.uniform(4, friction=0.5)


def test_drive_follows_the_simulator_and_skips_rest():
    from vib_tutorial.core import ForceController, ForceKind, ForceSettings, Simulator

    # A pulse at m4, then nothing: the chain slides, sticks and stays stuck. drive()
    # on the recorded force matches advance() (which keeps the ledger), and skips the rest.
    force = ForceController(ForceSettings(target=3, kind=ForceKind.PULSE, amplitude=20.0, pulse_duration=0.2))
    sim = Simulator(FRICTION, force)
    force.switch_on()
    _, xs, vs, fs = sim.advance(8.0)
    driven = Simulator(FRICTION, ForceController(ForceSettings(target=3)))
    z, a = driven.drive(fs, sim.step_size())
    np.testing.assert_allclose(z, np.hstack([xs, vs]), rtol=0, atol=1e-12)
    assert np.all(vs[-100:] == 0.0)  # at rest at the end
    # Accelerations: the equation of motion with friction, zero while stuck.
    _, C, K = FRICTION.matrices()  # unit masses
    pull = -xs @ K - vs @ C
    pull[:, 3] += fs
    stuck = (vs == 0) & (np.abs(pull) <= 0.5)
    way = np.where(vs != 0, np.sign(vs), np.sign(pull))
    np.testing.assert_allclose(a, np.where(stuck, 0.0, pull - 0.5 * way), rtol=0, atol=1e-9)
    assert stuck.any() and (~stuck).any()


def test_force_level_leaves_a_linear_frf_alone():
    settings = dict(excitation=Excitation.IMPACT, input_dof=3, averages=2)
    a, b = (measure(SYSTEM, force_level=level, **settings) for level in (1.0, 10.0))
    processing = Processing(window=Window.FORCE_EXPONENTIAL)
    ha, hb = (FrfEstimator(acq, processing).estimate().H for acq in (a, b))
    np.testing.assert_allclose(hb, ha, rtol=1e-9, atol=0)
    assert np.abs(b.block(0)[0]).max() == pytest.approx(10 * np.abs(a.block(0)[0]).max())


def test_friction_frf_depends_on_the_force_level():
    # Stepped sine at m4 on a chain with 0.5 N of friction on every mass: the first
    # peak is flattened by a light force and close to the frictionless one under a strong force.
    ratios = []
    for level in (0.2, 3.0):
        acq = measure(FRICTION, excitation=Excitation.STEPPED_SINE, input_dof=3, fs=19.0, sine_points=6,
                      force_level=level)
        est = FrfEstimator(acq, Processing()).estimate()
        exact = frf(FRICTION, est.freqs, 3)[:, 3]  # the chain without friction
        ratios.append(np.abs(est.H[:, 3]).max() / np.abs(exact).max())
    assert ratios[0] < 0.25 and 0.9 < ratios[1] < 1.0


def test_friction_lowers_the_coherence_of_random_excitation():
    def coherence(system):
        acq = measure(system, excitation=Excitation.RANDOM, input_dof=3, fs=19.0, averages=4, force_level=0.1)
        est = FrfEstimator(acq, Processing(window=Window.HANN)).estimate()
        keep = (est.freqs > 0.2) & (est.freqs < acq.settings.band)
        return est.coherence[keep, 3].mean()

    assert coherence(SYSTEM) > 0.99 and coherence(FRICTION) < 0.95


def test_work_measures_in_slices_and_gives_the_same_record():
    settings = MeasurementSettings(excitation=Excitation.RANDOM, input_dof=3, fs=19.0, averages=2, seed=3)
    whole = Acquisition(FRICTION, settings)
    whole.step()
    sliced = Acquisition(FRICTION, MeasurementSettings(**{**settings.__dict__}))
    calls = 0
    while not sliced.work(0.0):  # one piece of analog signal per call
        calls += 1
    assert calls > 3 and sliced.blocks == whole.blocks == 1
    for x, y in zip(whole.block(0), sliced.block(0)):
        np.testing.assert_array_equal(x, y)
