"""Shared limits, and the colour themes for the GUI.

Every colour the GUI draws with comes from the current Theme, read through the
`colors` proxy at the time something is drawn (``colors.mass[i]``,
``colors.force``). Switching themes (``set_theme``) swaps the Theme behind the
proxy; the widgets then redraw with it (see ``MainWindow.set_theme``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_DOF = 8


@dataclass(frozen=True)
class Theme:
    name: str
    dark: bool
    # --- Qt widgets. None keeps the platform's own palette (the Light theme).
    window: str | None = None
    base: str | None = None  # text entry, tables, text browsers
    alt_base: str | None = None
    button: str | None = None
    text: str | None = None
    highlight: str | None = None
    highlighted_text: str | None = None
    link: str | None = None
    disabled: str | None = None  # disabled text
    # --- plots and text
    background: str = "#ffffff"  # plots, animation, energy bars
    foreground: str = "#000000"  # axes, tick labels, plot titles
    muted: str = "#666666"  # secondary text: notes, captions
    faint: str = "#bbbbbb"  # reference lines: zero lines, unit circle, cut lines
    grey: str = "#888888"  # secondary curves (windows, envelopes), greyed-out entries
    strong: str = "#000000"  # the reference curve: full solution, exact FRF
    structure: str = "#333333"  # springs, dampers, wall, mass outlines
    legend_alpha: int = 215  # legend background: `background` at this alpha
    cell_border: str = "#cccccc"
    diagonal_tint: str = "#f1f3f8"
    band_shade: tuple[int, int, int, int] = (0, 0, 0, 18)  # above the anti-alias passband
    # --- data. One color per mass, used consistently in the animation, time plots and FRF.
    mass: tuple[str, ...] = ()
    # Mode shapes use a separate palette so they are not confused with masses.
    mode: tuple[str, ...] = ()
    sub: tuple[str, ...] = ()  # substructures A, B, C, ... in turn (bands, springs, labels)
    roots: tuple[str, ...] = ()  # non-oscillatory (overdamped) roots in the FRF matrix
    force: str = "#c1121f"  # the force, and anything drawn to stand out (selection, sums)
    energy: dict[str, str] = field(default_factory=dict)  # kinetic, potential, stored, added, dissipated
    fit: str = "#2a9d8f"  # fitted modal model and its band
    good: str = "#2a7d2a"
    fair: str = "#b36b00"
    poor: str = "#c1121f"
    stab_new: str = "#bbbbbb"  # stabilization diagram: new, stable in f, stable in f and ζ
    stab_freq: str = "#e76f51"
    stab_stable: str = "#264653"
    mac_high: str = "#264653"  # MAC matrix colour at 1 (0 is the background)
    block_tints: dict[tuple[str, str], str] = field(default_factory=dict)  # CMS matrix partitions
    dim_tint: str = "#f4f4f4"  # CMS matrix: discarded columns

    @property
    def energy_colors(self) -> dict[str, str]:
        """Energy bars and curves: stored (kinetic, potential), and the ledger of energy in and out."""
        return {**self.energy, "work": self.force}

    @property
    def method(self) -> dict[str, str]:
        """Component mode synthesis methods, on the Substructuring page's comparison plot."""
        return {"craig-bampton": self.strong, "rubin": self.mode[1], "macneal": self.mode[3]}

    def legend_brush(self) -> tuple[int, int, int, int]:
        r, g, b = rgb(self.background)
        return r, g, b, self.legend_alpha


LIGHT = Theme(
    name="Light",
    dark=False,
    fair="#a35f00",
    mass=("#1f77b4", "#ec7000", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#d35db1", "#0e9fae"),
    mode=("#264653", "#2a9d8f", "#b08900", "#e76f51", "#6a4c93", "#1982c4", "#62a01a", "#ff595e"),
    sub=("#3a6ea5", "#d17a00", "#4f8f3a", "#8a4fa8"),
    roots=("#888888", "#b0b0b0", "#606060"),
    energy={"kinetic": "#1982c4", "potential": "#2a9d8f", "stored": "#7b889e", "added": "#6a4c93",
            "dissipated": "#e76f51"},
    block_tints={("i", "i"): "#dbe8f5", ("b", "b"): "#dcefd8", ("b", "i"): "#fbe7d3",
                 ("q", "q"): "#ebe2f5", ("b", "q"): "#f6efcc", ("i", "q"): "#efe6f7"},
)

DARK = Theme(
    name="Dark",
    dark=True,
    window="#26282c",
    base="#1c1e21",
    alt_base="#24272b",
    button="#30333a",
    text="#e4e6e9",
    highlight="#3d6fb4",
    highlighted_text="#ffffff",
    link="#6cb6ff",
    disabled="#6e737a",
    background="#1c1e21",
    foreground="#d2d5da",
    muted="#a2a7ae",
    faint="#4a4e55",
    grey="#8a9099",
    strong="#f2f3f5",
    structure="#c3c7cd",
    cell_border="#43474e",
    diagonal_tint="#262b33",
    band_shade=(255, 255, 255, 16),
    mass=("#4c9fea", "#ff9a3c", "#4cc46a", "#ff6161", "#b996e6", "#d0a07c", "#f27fca", "#35d0e0"),
    mode=("#b9a4ff", "#3ccab8", "#e3b82c", "#f58f70", "#e07be0", "#4fb3f5", "#9ed84a", "#ff7c82"),
    sub=("#6fa6e6", "#f2a33a", "#7ccb66", "#c792ea"),
    roots=("#9a9a9a", "#6f6f6f", "#c4c4c4"),
    force="#ff4f5e",
    energy={"kinetic": "#4fb3f5", "potential": "#3ccab8", "stored": "#a9b4c6", "added": "#a88cf0",
            "dissipated": "#f58f70"},
    fit="#3ccab8",
    good="#62c962",
    fair="#eaa53e",
    poor="#ff6b6b",
    stab_new="#5c6168",
    stab_freq="#f58f70",
    stab_stable="#9cc3d5",
    mac_high="#9cc3d5",
    block_tints={("i", "i"): "#1d3349", ("b", "b"): "#1f3a26", ("b", "i"): "#43301c",
                 ("q", "q"): "#31264a", ("b", "q"): "#3d3718", ("i", "q"): "#382a47"},
    dim_tint="#26282b",
)

# Solarized (Ethan Schoonover) light, with the accents darkened where they were too
# pale to read on the cream background.
SOLARIZED = Theme(
    name="Solarized",
    dark=False,
    window="#eee8d5",
    base="#fdf6e3",
    alt_base="#f5efdc",
    button="#e6dfca",
    text="#34474f",
    highlight="#268bd2",
    highlighted_text="#fdf6e3",
    link="#1a6fb0",
    disabled="#93a1a1",
    background="#fdf6e3",
    foreground="#3b4f57",
    muted="#586e75",
    faint="#d3cbb7",
    grey="#879696",
    strong="#073642",
    structure="#34474f",
    cell_border="#d8d0bb",
    diagonal_tint="#f3ead2",
    band_shade=(88, 110, 117, 22),
    mass=("#268bd2", "#cb4b16", "#6c7d00", "#dc322f", "#6c71c4", "#9a7400", "#d33682", "#1f8a82"),
    mode=("#6c71c4", "#1f8a82", "#8a6d00", "#bd3613", "#a0307e", "#1d6fa5", "#5f7000", "#b0185a"),
    sub=("#1d6fa5", "#cb4b16", "#5f7000", "#6c71c4"),
    roots=("#93a1a1", "#b4bcb6", "#657b83"),
    force="#dc322f",
    energy={"kinetic": "#1d6fa5", "potential": "#1f8a82", "stored": "#7f8f8f", "added": "#6c71c4",
            "dissipated": "#cb4b16"},
    fit="#1f8a82",
    good="#5f7000",
    fair="#a05000",
    poor="#c62a27",
    stab_new="#c9c2ad",
    stab_freq="#cb4b16",
    stab_stable="#073642",
    mac_high="#0b4f6c",
    block_tints={("i", "i"): "#e3ecef", ("b", "b"): "#e8eed6", ("b", "i"): "#f7e3cf",
                 ("q", "q"): "#ece6f1", ("b", "q"): "#f3eac5", ("i", "q"): "#efe6ef"},
    dim_tint="#f2ecd9",
)

# Nord (Arctic Ice Studio): polar-night backgrounds, frost and aurora accents.
NORD = Theme(
    name="Nord",
    dark=True,
    window="#3b4252",
    base="#2e3440",
    alt_base="#353c4a",
    button="#434c5e",
    text="#eceff4",
    highlight="#5e81ac",
    highlighted_text="#eceff4",
    link="#88c0d0",
    disabled="#7b8394",
    background="#2e3440",
    foreground="#d8dee9",
    muted="#aab3c5",
    faint="#4c566a",
    grey="#8e99ae",
    strong="#eceff4",
    structure="#d8dee9",
    cell_border="#4c566a",
    diagonal_tint="#353c4a",
    band_shade=(236, 239, 244, 18),
    mass=("#81a1c1", "#d08770", "#a3be8c", "#e06c75", "#b48ead", "#ebcb8b", "#e39ec1", "#88c0d0"),
    mode=("#b48ead", "#8fbcbb", "#dfb65f", "#e5957a", "#8f9cff", "#6fc3e6", "#b4d38f", "#f08c93"),
    sub=("#81a1c1", "#d08770", "#a3be8c", "#b48ead"),
    roots=("#8e99ae", "#6b758a", "#b8c0cf"),
    force="#ef6f7a",
    energy={"kinetic": "#6fb7e6", "potential": "#8fbcbb", "stored": "#aab3c5", "added": "#c3a0d9",
            "dissipated": "#e5957a"},
    fit="#8fbcbb",
    good="#a3be8c",
    fair="#ebcb8b",
    poor="#f28b94",
    stab_new="#5d677c",
    stab_freq="#e5957a",
    stab_stable="#c5cedd",
    mac_high="#88c0d0",
    block_tints={("i", "i"): "#34465c", ("b", "b"): "#374a3c", ("b", "i"): "#4c3f37",
                 ("q", "q"): "#443a50", ("b", "q"): "#4b4634", ("i", "q"): "#473d4f"},
    dim_tint="#353a45",
)

THEMES = {t.name: t for t in (LIGHT, DARK, SOLARIZED, NORD)}
SYSTEM = "System"  # follow the desktop: Light or Dark


class _Current:
    """The current theme's colours: ``colors.force``, ``colors.mass[i]``."""

    theme: Theme = LIGHT

    def __getattr__(self, name: str):
        return getattr(type(self).theme, name)


colors = _Current()


def current() -> Theme:
    return _Current.theme


def set_theme(theme: Theme) -> None:
    _Current.theme = theme


def rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def luminance(color: str) -> float:
    """WCAG relative luminance of a #rrggbb colour."""

    def lin(v: int) -> float:
        s = v / 255
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = rgb(color)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two #rrggbb colours (1 to 21)."""
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def text_on(fill: str) -> str:
    """Black or white, whichever reads better on `fill` (e.g. a mass's label)."""
    return "#000000" if contrast(fill, "#000000") >= contrast(fill, "#ffffff") else "#ffffff"
