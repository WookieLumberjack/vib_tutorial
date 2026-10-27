"""The themes' colours: legible on their backgrounds, and the masses told apart by colour-blind eyes too."""

import itertools

import numpy as np
import pytest

from vib_tutorial.gui import style

# Machado, Oliveira & Fernandes (2009), severity 1: linear RGB -> what a dichromat sees.
DEUTERANOPIA = np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                         [-0.011820, 0.042940, 0.968881]])
PROTANOPIA = np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                       [-0.003882, -0.048116, 1.051998]])
SIGHT = {"normal": np.eye(3), "deuteranopia": DEUTERANOPIA, "protanopia": PROTANOPIA}
MIN_DELTA_E = 10.0  # CIE76; the tab10 green and red of masses 3 and 4 were 7 apart under deuteranopia
MASSES = 4  # the default chain's: the ones a student meets first


def _linear(color: str) -> np.ndarray:
    c = np.array(style.rgb(color)) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(linear: np.ndarray) -> np.ndarray:
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ np.clip(linear, 0, 1)
    t = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


@pytest.mark.parametrize("theme", style.THEMES.values(), ids=lambda t: t.name)
@pytest.mark.parametrize("sight", SIGHT)
def test_the_first_masses_stay_apart(theme, sight):
    colours = theme.mass[:MASSES]
    seen = [_lab(SIGHT[sight] @ _linear(c)) for c in colours]
    for (i, a), (j, b) in itertools.combinations(enumerate(seen), 2):
        assert np.linalg.norm(a - b) >= MIN_DELTA_E, (theme.name, sight, f"m{i + 1}", f"m{j + 1}")


@pytest.mark.parametrize("theme", style.THEMES.values(), ids=lambda t: t.name)
def test_curves_read_on_the_background(theme):
    for c in (*theme.mass[:MASSES], *theme.mode[:MASSES]):
        assert style.contrast(c, theme.background) >= 3.0, (theme.name, c)


@pytest.mark.parametrize("theme", style.THEMES.values(), ids=lambda t: t.name)
def test_mode_1_is_not_the_reference_curve(theme):
    """Mode 1's term and the full solution (strong) are drawn together, both 1 px. Light's old slate
    mode 1 (#264653) was 31 from its black and read as black; Nord's mauve on white is 38."""
    assert np.linalg.norm(_lab(_linear(theme.mode[0])) - _lab(_linear(theme.strong))) >= 35.0
